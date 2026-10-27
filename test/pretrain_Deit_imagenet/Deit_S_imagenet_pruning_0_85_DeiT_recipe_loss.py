import sys
import os
import random
import logging
import time
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torchvision import transforms
from thop import profile

import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


# ===================== 常量定义 =====================
SEED = 42

# 剪枝参数
GROUP_SPARSITY = 0.85
PRUNE_LR = 1e-4
WARMUP_EPOCHS_PRUNE = 5
PRUNING_EPOCHS = 150
BATCH_SIZE_PRUNE = 128

# 微调参数
FINETUNE_LR = 3e-4
WARMUP_EPOCHS_FINETUNE = 10
FINETUNE_EPOCHS = 510
BATCH_SIZE_FINETUNE = 256

# DropPath（剪枝/微调阶段可单独调）
DROP_PATH_RATE_PRUNE = 0.00
DROP_PATH_RATE_FINETUNE = 0.00

# ImageNet / DeiT 基本设置
NUM_CLASSES = 1000
IMG_SIZE = 224

# DataLoader
NUM_WORKERS = 16
PIN_MEMORY = True

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("deit_prune_finetune")


# ===================== 路径处理（统一）=====================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
MODEL_DEFINITION_ROOT = CURRENT_FILE.parent / "model"
DATA_MODULE_ROOT = Path("/workspace/data")

for path in [PROJECT_ROOT, MODEL_DEFINITION_ROOT, DATA_MODULE_ROOT]:
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

# ===================== 依赖导入 =====================
from only_train_once import OTO
from imagenet_dataloader import load_prepared_imagenet_data
from only_train_once.optimizer.alpha_scheduler import prepare_alpha_eval_batches


# ===================== RepeatAugSampler 兼容导入/回退实现 =====================
RepeatAugSampler = None
try:
    from timm.data import RepeatAugSampler as _RAS
    RepeatAugSampler = _RAS
except Exception:
    try:
        from timm.data.distributed_sampler import RepeatAugSampler as _RAS2
        RepeatAugSampler = _RAS2
    except Exception:
        class RepeatAugSampler(torch.utils.data.Sampler):
            """
            轻量回退版 RepeatAugSampler：
            - 每个 epoch 生成 base_indices（可 shuffle）
            - 重复 num_repeats 次后再打乱
            - pad / truncate 到 total_size
            - 按 rank 分片
            """
            def __init__(
                self,
                dataset,
                num_repeats: int = 3,
                rank: int = 0,
                num_replicas: int = 1,
                shuffle: bool = True,
            ):
                self.dataset = dataset
                self.num_repeats = int(num_repeats)
                self.rank = int(rank)
                self.num_replicas = int(num_replicas)
                self.shuffle = bool(shuffle)
                self.epoch = 0

                self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
                self.total_size = int(self.num_samples * self.num_replicas)

            def set_epoch(self, epoch: int):
                self.epoch = int(epoch)

            def __iter__(self):
                g = torch.Generator()
                g.manual_seed(self.epoch)

                if self.shuffle:
                    base_indices = torch.randperm(len(self.dataset), generator=g).tolist()
                else:
                    base_indices = list(range(len(self.dataset)))

                # repeat
                repeated = base_indices * self.num_repeats

                # shuffle after repeat（关键）
                if self.shuffle:
                    rng = random.Random(self.epoch)
                    rng.shuffle(repeated)

                # pad/truncate to total_size
                if len(repeated) < self.total_size:
                    pad = self.total_size - len(repeated)
                    repeated += repeated[:pad]
                else:
                    repeated = repeated[:self.total_size]

                # shard by rank
                start = self.rank * self.num_samples
                end = start + self.num_samples
                return iter(repeated[start:end])

            def __len__(self):
                return self.num_samples

# ===================== 增强配置 =====================
PRUNE_AUG: Dict[str, Any] = dict(
    input_size=IMG_SIZE,
    auto_augment='rand-m9-mstd0.5-inc1',
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
    interpolation='bicubic',
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],

    # mixup/cutmix args（剪枝阶段减弱）
    mixup_alpha=0,
    cutmix_alpha=0,
    mixup_prob=0,
    cutmix_switch_prob=0,
    label_smoothing=0.1,
)

# FINETUNE_AUG: Dict[str, Any] = dict(
#     input_size=IMG_SIZE,
#     auto_augment='rand-m9-mstd0.5-inc1',
#     re_prob=0.25,
#     re_mode='pixel',
#     re_count=1,
#     interpolation='bicubic',
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225],

#     mixup_alpha=0.8,
#     cutmix_alpha=1.0,
#     mixup_prob=1.0,
#     cutmix_switch_prob=0.5,
#     label_smoothing=0.1,
# )

FINETUNE_AUG: Dict[str, Any] = dict(
    input_size=IMG_SIZE,
    auto_augment='rand-m7-mstd0.5-inc1',
    re_prob=0.10,
    re_mode='pixel',
    re_count=1,
    interpolation='bicubic',
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],

    mixup_alpha=0.4,
    cutmix_alpha=0.5,
    mixup_prob=0.8,
    cutmix_switch_prob=0.5,
    label_smoothing=0.1,
)

# Eval / Test transform（通常两阶段共用；你若想分离也可以再拆两份）
EVAL_AUG: Dict[str, Any] = dict(
    resize_size=256,
    crop_size=IMG_SIZE,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ===================== 工具函数 =====================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(aug_cfg: Dict[str, Any]):
    """基于 timm.create_transform 构建训练增强（支持 RandAug + RandomErasing 等）"""
    return create_transform(
        input_size=aug_cfg["input_size"],
        is_training=True,
        color_jitter=aug_cfg.get("color_jitter", None),
        auto_augment=aug_cfg["auto_augment"],
        re_prob=aug_cfg["re_prob"],
        re_mode=aug_cfg["re_mode"],
        re_count=aug_cfg["re_count"],
        interpolation=aug_cfg["interpolation"],
        mean=aug_cfg["mean"],
        std=aug_cfg["std"],
    )


def build_eval_transform(eval_cfg: Dict[str, Any]):
    """构建 ImageNet eval 增强（Resize+CenterCrop+Normalize）"""
    return transforms.Compose([
        transforms.Resize(eval_cfg["resize_size"], interpolation=eval_cfg["interpolation"]),
        transforms.CenterCrop(eval_cfg["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_cfg["mean"], std=eval_cfg["std"]),
    ])


def build_mixup_fn(aug_cfg: Dict[str, Any], num_classes: int) -> Optional[Mixup]:
    """构建 Mixup/Cutmix（若 mixup_prob<=0 且 mixup_alpha/cutmix_alpha<=0 可返回 None）"""
    prob = float(aug_cfg.get("mixup_prob", 0.0))
    mixup_alpha = float(aug_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(aug_cfg.get("cutmix_alpha", 0.0))
    if prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=None,
        prob=prob,
        switch_prob=float(aug_cfg.get("cutmix_switch_prob", 0.5)),
        mode='batch',
        label_smoothing=float(aug_cfg.get("label_smoothing", 0.0)),
        num_classes=num_classes
    )


def load_local_deit() -> torch.nn.Module:
    """加载本地 DeiT-small 预训练权重"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "deit_small_patch16_224"
    local_model_weights_path = '/workspace/OTOv2_v2/test/pretrain_Deit_imagenet/model/deit_small_patch16_244/pytorch_model.bin'

    logger.info(f"创建 DeiT 模型: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    logger.info(f"加载权重: {local_model_weights_path}")
    ckpt = torch.load(local_model_weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_ema"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                logger.info(f"从 checkpoint['{key}'] 加载权重")
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"未知 checkpoint 格式: {type(ckpt)}")

    new_state = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)

    if missing:
        logger.warning(f"缺失参数 {len(missing)} 个：{missing[:5]}")
    if unexpected:
        logger.warning(f"多余参数 {len(unexpected)} 个：{unexpected[:5]}")

    model.to(device).eval()
    logger.info("DeiT 权重加载完成")
    return model


@torch.no_grad()
def check_accuracy(model, loader, amp_enabled: bool = True) -> Tuple[float, float]:
    """Top-1 / Top-5 accuracy（AMP 可开关）"""
    model.eval()
    num_correct, num_samples, top5_correct = 0, 0, 0

    model_device = next(model.parameters()).device
    amp_device_type = model_device.type if model_device.type in {"cuda", "cpu"} else "cuda"

    for i, (x, y) in enumerate(loader):
        x = x.to(model_device, non_blocking=True)
        y = y.to(model_device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            scores = model(x)

        _, preds = scores.max(1)
        num_correct += (preds == y).sum().item()

        _, top5_preds = scores.topk(5, dim=1)
        top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()

        num_samples += preds.size(0)
        if (i + 1) % 100 == 0:
            logger.info(f"准确率评估：已处理 {i+1}/{len(loader)} 批次")

    if num_samples == 0:
        return 0.0, 0.0

    top1_acc = num_correct / num_samples * 100
    top5_acc = top5_correct / num_samples * 100
    return top1_acc, top5_acc


def compute_flops_with_thop(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE)) -> Tuple[float, float]:
    """thop: 返回 (MFLOPs, MParams)"""
    if isinstance(model, torch.nn.DataParallel):
        model_to_copy = model.module
    else:
        model_to_copy = model

    model_cpu = copy.deepcopy(model_to_copy).to("cpu").eval()
    dummy_input = torch.randn(*input_size).to("cpu")
    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)

    flops_m = macs * 2 / 1e6
    params_m = params / 1e6

    del model_cpu, dummy_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return flops_m, params_m


def build_imagenet_loaders(
    batch_size: int,
    train_transform,
    test_transform,
    use_repeat_aug: bool = True,
    num_repeats: int = 3,
):
    """统一封装 ImageNet loaders（可开 RepeatAug）"""
    trainloader, testloader, trainset, testset = load_prepared_imagenet_data(
        base_imagenet_path="/workspace/data/imagenet",
        batch_size=batch_size,
        workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    if use_repeat_aug and RepeatAugSampler is not None:
        try:
            ra_sampler = RepeatAugSampler(
                trainset,
                num_repeats=num_repeats,
                num_replicas=1,
                rank=0,
                shuffle=True
            )
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                sampler=ra_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=True
            )
            logger.info(f"启用 RepeatAugSampler (num_repeats={num_repeats})")
        except Exception as e:
            logger.warning(f"RepeatAug 启用失败: {e}，回退标准 DataLoader")

    return trainloader, testloader, trainset, testset


# ===================== 工具函数 =====================
def format_eta(seconds: float) -> str:
    """把秒格式化成 H:MM:SS"""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"


# ===================== 剪枝阶段 =====================
def prune_model(seed: int = SEED):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    output_cache_dir = CURRENT_FILE.parent / f"cache_deit_small_imagenet_pruned_{GROUP_SPARSITY}"
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) 模型
    logger.info("\n=== [Prune] Step1: 加载预训练 DeiT ===")
    model = load_local_deit().to(device)

    # DropPath 调整（剪枝阶段）
    try:
        from timm.layers import DropPath
        logger.info(f"调整 DropPath 率 (Prune) 为 {DROP_PATH_RATE_PRUNE}")
        for m in model.modules():
            if isinstance(m, DropPath):
                m.drop_prob = DROP_PATH_RATE_PRUNE
    except ImportError:
        try:
            from timm.models.layers import DropPath
            logger.info(f"调整 DropPath 率 (Prune, legacy import) 为 {DROP_PATH_RATE_PRUNE}")
            for m in model.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = DROP_PATH_RATE_PRUNE
        except Exception as e:
            logger.warning(f"DropPath 调整跳过: {e}")
    except Exception as e:
        logger.warning(f"DropPath 调整跳过: {e}")

    # 2) 数据增强（✅ 使用 PRUNE_AUG）
    logger.info("\n=== [Prune] Step2: 构建剪枝阶段增强（PRUNE_AUG）===")
    transform_train_prune = build_train_transform(PRUNE_AUG)
    transform_test = build_eval_transform(EVAL_AUG)

    logger.info("\n=== [Prune] Step3: 加载 ImageNet DataLoader ===")
    trainloader, testloader, trainset, _ = build_imagenet_loaders(
        batch_size=BATCH_SIZE_PRUNE,
        train_transform=transform_train_prune,
        test_transform=transform_test,
        use_repeat_aug=True,
        num_repeats=3
    )

    # 3) Alpha 子集
    logger.info("\n=== [Prune] Step4: 准备 Alpha 更新子集 ===")
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=trainset,
        device=device,
        batch_size=BATCH_SIZE_PRUNE,
        num_workers=NUM_WORKERS,
    )
    logger.info(f"Alpha 子集：样本数={alpha_info['n_alpha_val']} | 批大小={alpha_info['alpha_batch_size']}")

    # 4) OTO 初始化
    logger.info("\n=== [Prune] Step5: 初始化 OTO ===")
    dummy_input = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=output_cache_dir)
    nodes_to_keep_unpruned = ['node-163']
    print(f"尝试标记节点 {nodes_to_keep_unpruned} 为不可剪枝...")
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)

    # 5) 剪枝前评估
    logger.info("\n=== [Prune] Step6: 剪枝前评估 ===")
    acc1_before, acc5_before = check_accuracy(model, testloader, amp_enabled=True)
    flops_before, params_before = compute_flops_with_thop(model)
    logger.info(f"剪枝前 | Top-1: {acc1_before:.2f}% | Top-5: {acc5_before:.2f}%")
    logger.info(f"剪枝前 | FLOPs: {flops_before:.2f}M | Params: {params_before:.2f}M")

    # 6) 训练配置（Mixup/Cutmix ✅ 使用 PRUNE_AUG）
    logger.info("\n=== [Prune] Step7: 配置剪枝优化器/调度器 ===")
    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = WARMUP_EPOCHS_PRUNE * num_update_steps_per_epoch
    pruning_steps = PRUNING_EPOCHS * num_update_steps_per_epoch
    total_steps = warmup_steps + pruning_steps

    mixup_fn = build_mixup_fn(PRUNE_AUG, NUM_CLASSES)
    if mixup_fn is None:
        criterion_train = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    else:
        criterion_train = SoftTargetCrossEntropy().to(device)
    criterion_oto = nn.CrossEntropyLoss().to(device)

    optimizer_prune = oto.hesso(
        variant='adamw',
        lr=PRUNE_LR,
        first_momentum=0.9,
        second_momentum=0.999,
        dampening=0.0,
        weight_decay=1e-4,
        target_group_sparsity=GROUP_SPARSITY,
        start_pruning_step=warmup_steps,
        pruning_periods=PRUNING_EPOCHS,
        pruning_steps=pruning_steps,
        device=device,
    )

    warmup_scheduler = LinearLR(
        optimizer_prune,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer_prune,
        T_max=total_steps - warmup_steps,
    )
    lr_scheduler_prune = SequentialLR(
        optimizer_prune,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # 7) 剪枝训练循环
    logger.info("\n=== [Prune] Step8: 开始剪枝训练 ===")
    model.train()
    optimizer_prune.zero_grad(set_to_none=True)
    warned_odd_batch = False

    total_epochs_prune = WARMUP_EPOCHS_PRUNE + PRUNING_EPOCHS
    prune_train_t0 = time.time()

    for epoch in range(total_epochs_prune):
        epoch_t0 = time.time()

        if hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        logger.info(f"\n[Prune] Epoch {epoch+1}/{total_epochs_prune}")
        epoch_loss, epoch_steps = 0.0, 0
        epoch_lr = 0.0
        t0 = time.time()

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                bs = inputs.shape[0]
                if bs % 2 != 0:
                    if bs == 1:
                        if not warned_odd_batch:
                            logger.warning("Batch size=1，跳过该批次")
                            warned_odd_batch = True
                        continue
                    inputs = inputs[:-1]
                    targets = targets[:-1]
                    if not warned_odd_batch:
                        logger.warning(f"奇数 batch size={bs}，丢弃 1 个样本")
                        warned_odd_batch = True
                inputs, targets = mixup_fn(inputs, targets)

            outputs = model(inputs)
            loss = criterion_train(outputs, targets)
            loss.backward()

            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion_oto,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )

            optimizer_prune.step()
            optimizer_prune.zero_grad(set_to_none=True)
            lr_scheduler_prune.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1
            epoch_lr += optimizer_prune.param_groups[0]["lr"]

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_lr = epoch_lr / max(1, epoch_steps)
        it_per_sec = epoch_steps / dt

        # 每个 epoch 结束打印一次 ETA
        epoch_time = time.time() - epoch_t0
        elapsed = time.time() - prune_train_t0
        avg_epoch = elapsed / (epoch + 1)
        remaining = avg_epoch * (total_epochs_prune - (epoch + 1))

        logger.info(
            f"[Prune] Epoch {epoch+1}/{total_epochs_prune} 评估\n"
            f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
            f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
            f"估计剩余: {format_eta(remaining)}"
        )

        # 输出剪枝指标（Sparsity + Alpha）
        try:
            metrics = optimizer_prune.compute_metrics()
            logger.info(f"Sparsity: {metrics.group_sparsity:.4f}")
            alpha_vals = dict(zip(optimizer_prune._alpha_keys,
                                  optimizer_prune.alpha_scheduler.get_alpha().tolist()))
            logger.info(str(alpha_vals))
        except Exception as e:
            logger.warning(f"剪枝指标输出失败: {e}")

        model.eval()
        acc1, acc5 = check_accuracy(model, testloader, amp_enabled=True)
        logger.info(f"[Prune] Epoch {epoch+1} Eval | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")
        model.train()

    # 8) 构建子网络
    logger.info("\n=== [Prune] Step9: 构建剪枝子网络 ===")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    logger.info(f"压缩模型保存路径: {compressed_model_path}")

    # 9) 剪枝后评估
    logger.info("\n=== [Prune] Step10: 剪枝后评估 ===")
    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device).eval()

    acc1_after, acc5_after = check_accuracy(compressed_model, testloader, amp_enabled=True)
    flops_after, params_after = compute_flops_with_thop(compressed_model)

    pruned_flops = flops_before - flops_after
    pruned_params = params_before - params_after
    flops_prune_ratio = pruned_flops / flops_before * 100
    params_prune_ratio = pruned_params / params_before * 100

    logger.info("\n=== [Prune] Summary ===")
    logger.info(f"FLOPs: {flops_before:.2f}M -> {flops_after:.2f}M (↓{flops_prune_ratio:.2f}%)")
    logger.info(f"Params: {params_before:.2f}M -> {params_after:.2f}M (↓{params_prune_ratio:.2f}%)")
    logger.info(f"Top-1: {acc1_before:.2f}% -> {acc1_after:.2f}% ({acc1_after - acc1_before:+.2f}%)")
    logger.info(f"Top-5: {acc5_before:.2f}% -> {acc5_after:.2f}% ({acc5_after - acc5_before:+.2f}%)")


# ===================== 微调阶段 =====================
def finetune_model(seed: int = SEED):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    pruned_dir = CURRENT_FILE.parent / f"cache_deit_small_imagenet_pruned_{GROUP_SPARSITY}"
    finetuned_dir = CURRENT_FILE.parent / f"finetuned_deit_small_sparsity_{GROUP_SPARSITY}"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    finetuned_model_path = finetuned_dir / "deit_small_imagenet_finetuned.pt"

    # 1) 加载剪枝后模型
    logger.info("\n=== [FT] Step1: 加载剪枝后模型 ===")
    compressed_model_candidates = sorted(pruned_dir.glob("*_compressed.pt"))
    if not compressed_model_candidates:
        logger.error("未找到剪枝后的模型，请先运行 prune_model()")
        return
    compressed_model_path = compressed_model_candidates[-1]
    logger.info(f"加载剪枝模型: {compressed_model_path}")

    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)

    # 2) DropPath 调整（训练期有效）
    try:
        # 修改导入路径以修复 FutureWarning
        from timm.layers import DropPath
        logger.info(f"调整 DropPath 率为 {DROP_PATH_RATE_FINETUNE}")
        for m in model.modules():
            if isinstance(m, DropPath):
                m.drop_prob = DROP_PATH_RATE_FINETUNE
    except ImportError:
        # 兼容旧版本 timm
        try:
            from timm.models.layers import DropPath
            logger.info(f"调整 DropPath 率 (legacy import) 为 {DROP_PATH_RATE_FINETUNE}")
            for m in model.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = DROP_PATH_RATE_FINETUNE
        except Exception as e:
            logger.warning(f"DropPath 调整跳过: {e}")
    except Exception as e:
        logger.warning(f"DropPath 调整跳过: {e}")

    # 3) 数据增强（✅ 使用 FINETUNE_AUG）
    logger.info("\n=== [FT] Step2: 构建微调阶段增强（FINETUNE_AUG）===")
    transform_train_ft = build_train_transform(FINETUNE_AUG)
    transform_test = build_eval_transform(EVAL_AUG)

    logger.info("\n=== [FT] Step3: 加载 ImageNet DataLoader ===")
    trainloader, testloader, trainset, _ = build_imagenet_loaders(
        batch_size=BATCH_SIZE_FINETUNE,
        train_transform=transform_train_ft,
        test_transform=transform_test,
        use_repeat_aug=True,
        num_repeats=3
    )

    # 4) 微调前评估
    logger.info("\n=== [FT] Step4: 微调前评估 ===")
    acc1_before, acc5_before = check_accuracy(model, testloader, amp_enabled=True)
    flops_after, params_after = compute_flops_with_thop(model)
    logger.info(f"微调前 | Top-1: {acc1_before:.2f}% | Top-5: {acc5_before:.2f}%")
    logger.info(f"剪枝后 | FLOPs: {flops_after:.2f}M | Params: {params_after:.2f}M")

    # 5) 微调优化器/调度器
    logger.info("\n=== [FT] Step5: 初始化优化器/调度器 ===")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_LR,
        weight_decay=5e-5,
        betas=(0.9, 0.999)
    )

    warmup_steps = WARMUP_EPOCHS_FINETUNE * len(trainloader)
    total_steps = FINETUNE_EPOCHS * len(trainloader)

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        lr_scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    mixup_fn = build_mixup_fn(FINETUNE_AUG, NUM_CLASSES)
    criterion = SoftTargetCrossEntropy().to(device)

    model_ema = ModelEmaV2(model, decay=0.9999, device=device)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 6) 微调循环
    logger.info("\n=== [FT] Step6: 开始微调 ===")
    best_top1 = -float("inf")
    best_source = "none"
    warned_odd_batch = False

    enable_mixup_decay = False
    decay_start = 240

    ft_train_t0 = time.time()

    for epoch in range(FINETUNE_EPOCHS):
        epoch_t0 = time.time()

        if hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        logger.info(f"\n[FT] Epoch {epoch+1}/{FINETUNE_EPOCHS}")
        model.train()

        # Mixup 衰减（只改 FINETUNE_AUG 的 mixup，不影响 label_smoothing）
        if enable_mixup_decay and (mixup_fn is not None) and epoch >= decay_start:
            ratio = max(0.0, 1.0 - (epoch - decay_start) / max(1, (FINETUNE_EPOCHS - decay_start)))
            mixup_fn.mixup_alpha = float(FINETUNE_AUG["mixup_alpha"]) * ratio
            mixup_fn.cutmix_alpha = float(FINETUNE_AUG["cutmix_alpha"]) * ratio
            mixup_fn.prob = float(FINETUNE_AUG["mixup_prob"]) * ratio

        epoch_loss, epoch_steps = 0.0, 0
        t0 = time.time()

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                bs = inputs.shape[0]
                if bs % 2 != 0:
                    if bs == 1:
                        if not warned_odd_batch:
                            logger.warning("Batch size=1，跳过该批次")
                            warned_odd_batch = True
                        continue
                    inputs = inputs[:-1]
                    targets = targets[:-1]
                    if not warned_odd_batch:
                        logger.warning(f"奇数 batch size={bs}，丢弃 1 个样本")
                        warned_odd_batch = True
                inputs, targets = mixup_fn(inputs, targets)

            with torch.amp.autocast(device_type=device.type, enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            model_ema.update(model)
            lr_scheduler.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / max(1, epoch_steps)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[FT] Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} "
            f"| Time: {dt:.2f}s | Throughput: {epoch_steps/dt:.2f} it/s"
        )

        # 每个 epoch 结束打印一次 ETA
        epoch_time = time.time() - epoch_t0
        elapsed = time.time() - ft_train_t0
        avg_epoch = elapsed / (epoch + 1)
        remaining = avg_epoch * (FINETUNE_EPOCHS - (epoch + 1))

        logger.info(
            f"[FT][Epoch {epoch+1}/{FINETUNE_EPOCHS}] "
            f"epoch_time={epoch_time:.1f}s | elapsed={format_eta(elapsed)} | ETA={format_eta(remaining)}"
        )

        # Eval（原模型 + EMA）
        model.eval()
        acc1_model, _ = check_accuracy(model, testloader, amp_enabled=True)
        acc1_ema, _ = check_accuracy(model_ema.module, testloader, amp_enabled=True)
        logger.info(f"[FT] Epoch {epoch+1} | Model Top-1: {acc1_model:.2f}% | EMA Top-1: {acc1_ema:.2f}%")

        if acc1_model >= acc1_ema:
            candidate_top1 = acc1_model
            candidate_source = "model"
            candidate_model = model
        else:
            candidate_top1 = acc1_ema
            candidate_source = "ema"
            candidate_model = model_ema.module

        if candidate_top1 > best_top1:
            best_top1 = candidate_top1
            best_source = candidate_source
            logger.info(f"[FT] 更新最佳 Top-1: {best_top1:.2f}% (source={best_source}) -> 保存 {finetuned_model_path}")
            to_save = candidate_model.module if isinstance(candidate_model, torch.nn.DataParallel) else candidate_model
            torch.save(to_save, finetuned_model_path)

    # 7) 最终评估
    logger.info("\n=== [FT] Step7: 微调完成 - 最终评估 ===")
    best_model = torch.load(finetuned_model_path, map_location=device, weights_only=False)
    best_model.to(device).eval()
    top1_final, top5_final = check_accuracy(best_model, testloader, amp_enabled=True)

    baseline_model = load_local_deit().to(device).eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, testloader, amp_enabled=True)

    logger.info("\n=== [FT] Final Summary ===")
    logger.info(f"剪枝+微调后 Top-1: {top1_final:.2f}% | Top-5: {top5_final:.2f}%")
    logger.info(f"未剪枝基线 Top-1: {baseline_top1:.2f}% | Top-5: {baseline_top5:.2f}%")
    logger.info(f"Top-1 变化: {top1_final - baseline_top1:+.2f}% | Top-5 变化: {top5_final - baseline_top5:+.2f}%")


def main():
    # prune_model()
    finetune_model()


if __name__ == "__main__":
    main()
