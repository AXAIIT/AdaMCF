import sys
import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.amp import GradScaler
import time
from torchvision import transforms
from thop import profile
import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DEFINITION_ROOT = CURRENT_FILE.parent / "model"
if str(MODEL_DEFINITION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_DEFINITION_ROOT))

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

from only_train_once import OTO
from imagenet_dataloader import load_prepared_imagenet_data
from only_train_once.optimizer.alpha_scheduler import prepare_alpha_eval_batches


def load_local_deit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "deit_small_patch16_224"
    local_model_weights_path = "/workspace/OTOv2_v2/test/pretrain_Deit_imagenet/model/deit_small_patch16_244/pytorch_model.bin"

    print(f"正在创建 DeiT 模型: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
    )
    print(f"从本地加载 DeiT 权重: {local_model_weights_path}")
    ckpt = torch.load(local_model_weights_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_ema"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"从 checkpoint['{key}'] 加载权重")
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"未知 checkpoint 格式: {type(ckpt)}")

    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[Warning] 缺失参数 {len(missing)} 个，例如: {missing[:5]}")
    if unexpected:
        print(f"[Warning] 多余参数 {len(unexpected)} 个，例如: {unexpected[:5]}")

    model.to(device).eval()
    print("成功从本地加载 DeiT 预训练权重。")
    return model


def check_accuracy(model, loader, amp_enabled: bool = True):
    """
    计算模型在给定数据加载器上的 Top-1 和 Top-5 准确率。
    amp_enabled: True -> 使用 autocast；False -> 纯 FP32
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    top5_num_correct = 0
    model_device = next(model.parameters()).device
    amp_device_type = model_device.type if model_device.type in {"cuda", "cpu"} else "cuda"

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(model_device, non_blocking=True)
            y = y.to(model_device, non_blocking=True)

            if amp_enabled:
                with torch.amp.autocast(device_type=amp_device_type, enabled=True):
                    scores = model(x)
            else:
                scores = model(x)
            
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

            _, top5_indices = scores.topk(5, dim=1)
            y_reshaped = y.view(-1, 1)
            top5_correct_batch = (top5_indices == y_reshaped).any(dim=1).sum().item()
            top5_num_correct += top5_correct_batch
            
            if (i + 1) % 100 == 0:
                print(f"  Accuracy check: processed batch {i+1}/{len(loader)}")

    if num_samples == 0:
        print("警告: check_accuracy 接收到的样本数量为0。")
        return 0.0, 0.0

    acc = float(num_correct) / num_samples * 100
    top5_acc = float(top5_num_correct) / num_samples * 100
    
    print(f"  Accuracy check: Total samples: {num_samples}, Top-1 Correct: {num_correct}, Top-5 Correct: {top5_num_correct}")
    return acc, top5_acc


def compute_flops_with_thop(model, input_size=(1, 3, 224, 224)):
    """
    使用 thop 计算模型 FLOPs 和参数量。
    返回:
        flops_m: MFLOPs
        params_m: M 参数量
    """
    import copy

    if isinstance(model, torch.nn.DataParallel):
        model_to_copy = model.module
    else:
        model_to_copy = model
    
    model_cpu = copy.deepcopy(model_to_copy).to("cpu").eval()
    dummy_input = torch.randn(*input_size).to("cpu")

    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)

    flops_m = macs * 2 / 1e6        # MACs -> FLOPs
    params_m = params / 1e6         # 参数量 -> M
    return flops_m, params_m


def prune_model(seed: int = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size = 128
    num_workers = 16
    local_data_path = "/workspace/data/imagenet"
    base_dir = Path(__file__).parent
    output_cache_dir = base_dir / "cache_deit_small_imagenet_pruned_0_8"
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 第 1 步：加载数据和预训练模型 ---")
    print("正在加载 DeiT 模型...")
    model = load_local_deit()
    model.to(device).eval()

    print(f"正在从 {local_data_path} 加载 ImageNet 数据...")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainloader, testloader, trainset, testset = load_prepared_imagenet_data(
        base_imagenet_path=local_data_path,
        batch_size=batch_size,
        workers=num_workers,
        pin_memory=True,
        train_transform=transform_train,
        test_transform=transform_test,
    )
    print(f"成功加载 ImageNet 训练集，包含 {len(trainset)} 张图片。")
    print(f"成功加载 ImageNet 测试集，包含 {len(testset)} 张图片。")

    # ===== α 子集 =====
    print("\n--- 准备 Alpha 更新子集 ---")
    val_batches, alpha_info = prepare_alpha_eval_batches(
        trainset=trainset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(
        f"[α-Subset] 样本数={alpha_info['n_alpha_val']}, "
        f"batch_size={alpha_info['alpha_batch_size']}, "
        f"微批数={alpha_info['num_val_microbatches']}"
    )

    print("\n--- 第 2 步：初始化 OTO ---")
    dummy_input = torch.rand(1, 3, 224, 224, device=device, dtype=torch.float32)

    oto = OTO(model=model, dummy_input=dummy_input)
    print("正在生成剪枝前的依赖图...")
    oto.visualize(view=False, out_dir=output_cache_dir)
    print("依赖图已生成。")

    nodes_to_keep_unpruned = ['node-163']
    print(f"尝试标记节点 {nodes_to_keep_unpruned} 为不可剪枝...")
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)
   
    print("\n--- 第 3 步：剪枝前评估（FP32，无 AMP） ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader, amp_enabled=False)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m, params_before_m = compute_flops_with_thop(model, input_size=(1, 3, 224, 224))
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    print("\n--- 第 4 步：使用 OTO 进行剪枝 ---")
    learning_rate = 1e-4
    group_sparsity = 0.8
    warmup_epochs = 5
    pruning_epochs = 150
    pruning_periods = pruning_epochs
    total_epochs = warmup_epochs + pruning_epochs

    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch
    start_pruning_step = warmup_steps
    total_steps = (warmup_epochs + pruning_epochs) * num_update_steps_per_epoch

    print(f"目标组稀疏率: {group_sparsity}")
    print(f"剪枝将在 step={start_pruning_step} 开始，持续 {pruning_steps} 步。")

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=learning_rate,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=1e-4,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        device=device,
    )

    warmup_scheduler = LinearLR(
        optimizer_prune,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer_prune,
        T_max=total_steps - warmup_steps,
        eta_min=8e-6,
    )

    lr_scheduler_prune = SequentialLR(
        optimizer_prune,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    print("\n--- 阶段一：预热与剪枝")
    model.train()
    optimizer_prune.zero_grad(set_to_none=True)
    for epoch in range(warmup_epochs + pruning_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )

            optimizer_prune.step()
            optimizer_prune.zero_grad(set_to_none=True)

            # 新增：每个 batch 更新一次 LR（放在 optimizer.step() 之后）
            lr_scheduler_prune.step()

            current_lr = optimizer_prune.param_groups[0]["lr"]
            epoch_loss += loss.item()
            epoch_lr += current_lr
            epoch_steps += 1

        # 删除：lr_scheduler_prune.step()  # 不要再在 epoch 末尾 step

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (total_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"[阶段一] Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        metrics = optimizer_prune.compute_metrics()
        print(f"Sparsity: {metrics.group_sparsity:.4f}")
        print(dict(zip(optimizer_prune._alpha_keys,
                       optimizer_prune.alpha_scheduler.get_alpha().tolist())))

        model.eval()
        acc1_epoch, acc5_epoch = check_accuracy(model, testloader, amp_enabled=False)
        print(
            f"--- [阶段一] Epoch {epoch + 1}/{warmup_epochs + pruning_epochs} 测试:"
            f" Top-1 {acc1_epoch:.2f}%, Top-5 {acc5_epoch:.2f}%",
            file=sys.stdout,
            flush=True
        )
        model.train()

    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    print(f"剪枝后的压缩模型保存在: {compressed_model_path}")
    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device)
    compressed_model.eval()

    print("\n--- 剪枝阶段完成，模型结构已固定 ---")
    print("\n--- 第 5 步：剪枝后评估（FP32） ---")
    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, testloader, amp_enabled=False)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m, params_after_m = compute_flops_with_thop(compressed_model, input_size=(1, 3, 224, 224))
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    eps = 1e-8
    pruned_flops_m = max(flops_before_m - flops_after_m, 0.0)
    pruned_params_m = max(params_before_m - params_after_m, 0.0)
    pruned_flops_percent = pruned_flops_m / (flops_before_m + eps) * 100
    pruned_params_percent = pruned_params_m / (params_before_m + eps) * 100
    flops_compression_ratio = (flops_before_m / (flops_after_m + eps)) if flops_after_m > 0 else float("inf")
    params_compression_ratio = (params_before_m / (params_after_m + eps)) if params_after_m > 0 else float("inf")

    print("\n--- 第 6 步：压缩总结 ---")
    print(f"减少的 FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少的参数量: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")

    if flops_before_m > 0 and flops_after_m > 0:
        print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x "
              f"({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if params_before_m > 0 and params_after_m > 0:
        print(f"参数量压缩比: {params_compression_ratio:.2f}x "
              f"({params_before_m:.2f}M -> {params_after_m:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")



def finetune_model(seed: int = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size = 256
    num_workers = 16
    local_data_path = "/workspace/data/imagenet"
    base_dir = Path(__file__).parent
    pruned_dir = base_dir / "cache_deit_small_imagenet_pruned_0_8"
    finetuned_dir = base_dir / "finetuned_deit_small_sparsity_0_8"
    finetuned_dir.mkdir(parents=True, exist_ok=True)

    compressed_model_path = pruned_dir / "VisionTransformer_compressed.pt"
    if not compressed_model_path.exists():
        candidates = sorted(pruned_dir.glob("*_compressed.pt"))
        if candidates:
            compressed_model_path = candidates[-1]
    if not compressed_model_path.exists():
        print("错误：未找到剪枝后的压缩模型，请先运行 prune_model。")
        return

    finetuned_model_path = finetuned_dir / "deit_small_imagenet_finetuned.pt"
    print(f"剪枝模型路径: {compressed_model_path}")
    print(f"微调模型将保存在: {finetuned_model_path}")

    print(f"正在从 {local_data_path} 加载 ImageNet 数据...")
    
    print("配置增强: RandAugment(m9-mstd0.5-inc1), RandomErasing(0.25)")
    transform_train = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.10,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainloader, testloader, trainset, testset = load_prepared_imagenet_data(
        base_imagenet_path=local_data_path,
        batch_size=batch_size,
        workers=num_workers,
        pin_memory=True,
        train_transform=transform_train,
        test_transform=transform_test,
    )
    
    RepeatAugSampler = None
    try:
        from timm.data import RepeatAugSampler
    except ImportError:
        try:
            from timm.data.distributed_sampler import RepeatAugSampler
        except ImportError:
            pass

    # 3. 如果都找不到，使用本地简易实现
    if RepeatAugSampler is None:
        print("[Info] 未找到 timm.data.RepeatAugSampler，使用本地实现。")
        import math
        class RepeatAugSampler(torch.utils.data.Sampler):
            """ 本地简易实现，仅支持单机单卡模式 (rank=0, replicas=1) """
            def __init__(self, dataset, num_repeats=3, rank=0, num_replicas=1, shuffle=True):
                self.dataset = dataset
                self.num_repeats = num_repeats
                self.shuffle = shuffle
                self.epoch = 0
                # 仅处理单卡逻辑
                self.num_samples = math.ceil(len(self.dataset) * self.num_repeats / num_replicas)
                self.total_size = self.num_samples * num_replicas

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                # 确定性种子，保证可复现
                g = torch.Generator()
                g.manual_seed(self.epoch)
                indices = []
                for _ in range(self.num_repeats):
                    if self.shuffle:
                        indices.append(torch.randperm(len(self.dataset), generator=g))
                    else:
                        indices.append(torch.arange(len(self.dataset)))
                return iter(torch.cat(indices).tolist())

            def __len__(self):
                return self.num_samples

    try:
        print("启用 Repeated Augmentation (factor=3)...")
        ra_sampler = RepeatAugSampler(
            trainset, 
            num_repeats=3, 
            num_replicas=1, 
            rank=0,
            shuffle=True
        )
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            sampler=ra_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        print(f"警告: 启用 Repeated Augmentation 失败 ({e})，使用标准 Loader。")

    print(f"成功加载 ImageNet 训练集，包含 {len(trainset)} 张图片。")
    print(f"成功加载 ImageNet 测试集，包含 {len(testset)} 张图片。")

    print(f"加载压缩后的模型从: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    print("子网络加载完成，将用此模型进行微调。")

    try:
        from timm.models.layers import DropPath
        new_drop_path_rate = 0.0
        print(f"调整 DropPath rate 为 {new_drop_path_rate} (适应剪枝后的模型容量)...")
        for m in model.modules():
            if isinstance(m, DropPath):
                m.drop_prob = new_drop_path_rate
    except Exception as e:
        print(f"警告: DropPath 调整失败，已跳过 ({e})。")
    # ----------------------------------------------------------------------

    print("\n--- 微调前评估 ---")
    acc1_before, acc5_before = check_accuracy(model, testloader, amp_enabled=True)
    print(f"微调前准确率: Top-1: {acc1_before:.2f}%, Top-5: {acc5_before:.2f}%")
    flops_after_m, params_after_m = compute_flops_with_thop(model, input_size=(1, 3, 224, 224))
    print(f"剪枝后 FLOPs(thop): {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    finetune_lr = 5e-4
    num_train_epochs = 300
    warmup_epochs = 0
    total_epochs = num_train_epochs

    finetuned_model_dir = os.path.join(base_dir, 'finetuned_deit_small_sparsity_0_8')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(finetuned_model_dir, 'deit_small_imagenet_finetuned.pt')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=1e-4,
    )

    print("初始化 Model EMA (decay=0.99996)...")
    model_ema = ModelEmaV2(model, decay=0.99996, device=device)

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=1e-6
        )

    print("配置 Mixup(0.8) 和 CutMix(1.0)...")
    mixup_fn = Mixup(
        mixup_alpha=0.6,
        cutmix_alpha=0.6,
        cutmix_minmax=None,
        prob=0.7,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000
    )

    enable_mixup_decay = True
    decay_start = 240

    criterion = SoftTargetCrossEntropy().to(device)

    amp_enabled = device.type == "cuda"
    amp_device_type = device.type if device.type in {"cuda", "cpu"} else "cuda"
    scaler_finetune = GradScaler(enabled=amp_enabled)

    print("\n--- 阶段二：微调（使用 AMP） ---")
    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    warned_odd_batch = False

    model.train()
    for epoch in range(num_train_epochs):
        if hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        in_warmup = epoch < warmup_epochs
        phase = "[阶段二-预热]" if in_warmup else "[阶段二-微调]"
        print(f"\n{phase} Epoch {epoch+1}/{total_epochs} 正在训练...")

        if enable_mixup_decay and epoch >= decay_start:
            denom = max(1, (total_epochs - decay_start))
            ratio = max(0.0, 1.0 - (epoch - decay_start) / denom)
            mixup_fn.mixup_alpha = 0.8 * ratio
            mixup_fn.cutmix_alpha = 1.0 * ratio
            mixup_fn.prob = 1.0 * ratio
            mixup_fn.label_smoothing = 0.1 * ratio

        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                bs = inputs.shape[0]
                if (bs % 2) == 1:
                    if bs == 1:
                        if not warned_odd_batch:
                            print("[Warning] 遇到 batch_size=1 的最后批次，已跳过以避免 Mixup 断言错误。")
                            warned_odd_batch = True
                        continue
                    inputs = inputs[:-1]
                    targets = targets[:-1]
                    if not warned_odd_batch:
                        print(f"[Warning] 遇到奇数 batch_size={bs}，已丢弃 1 个样本以满足 Mixup 要求。")
                        warned_odd_batch = True

                inputs, targets = mixup_fn(inputs, targets)

            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler_finetune.scale(loss).backward()
            scaler_finetune.step(optimizer)
            scaler_finetune.update()

            model_ema.update(model)
            # -------------------------------

            epoch_loss += loss.item()
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_lr += cur_lr
            epoch_steps += 1

        lr_scheduler.step()

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_lr = epoch_lr / max(1, epoch_steps)
        it_per_sec = epoch_steps / dt
        eta = (total_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"{phase} Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        model.eval()
        # 1) 评估普通模型
        acc1_model, acc5_model = check_accuracy(model, testloader, amp_enabled=True)
        print(f"Epoch {epoch+1} [Model] Top-1: {acc1_model:.2f}%, Top-5: {acc5_model:.2f}%")

        # 2) 评估 EMA 模型
        acc1_ema, acc5_ema = check_accuracy(model_ema.module, testloader, amp_enabled=True)
        print(f"Epoch {epoch+1} [EMA]   Top-1: {acc1_ema:.2f}%, Top-5: {acc5_ema:.2f}%")

        if acc1_ema > acc1_model:
            current_acc1, current_acc5 = acc1_ema, acc5_ema
            current_model_ref = model_ema.module
            source_str = "EMA"
        else:
            current_acc1, current_acc5 = acc1_model, acc5_model
            current_model_ref = model
            source_str = "Model"

        if current_acc1 > best_top1 or (current_acc1 == best_top1 and current_acc5 > best_top5):
            best_top1 = current_acc1
            best_top5 = current_acc5
            best_epoch = epoch + 1
            print(f"当前最佳Top-1准确率: {best_top1:.2f}% (Epoch {best_epoch}, Source: {source_str})")
            print(f"  - 保存模型到: {finetuned_model_path}")

            model_to_save = current_model_ref.module if isinstance(current_model_ref, torch.nn.DataParallel) else current_model_ref
            torch.save(model_to_save, finetuned_model_path)

        model.train()

    print("剪枝和微调过程完成。")
    print(f"\n--- 最终微调模型保存在: {finetuned_model_path} ---")

    model = torch.load(finetuned_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    top1_final, top5_final = check_accuracy(model, testloader, amp_enabled=True)
    print(f"微调后最终准确率: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")

    print("\n--- 评估未剪枝 DeiT baseline ---")
    baseline_model = load_local_deit()
    baseline_model.to(device).eval()
    baseline_top1, baseline_top5 = check_accuracy(baseline_model, testloader, amp_enabled=True)
    print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
    print(f"相较未剪枝模型: Top-1 变化 {top1_final - baseline_top1:.2f}%, "
          f"Top-5 变化 {top5_final - baseline_top5:.2f}%")

def main():
    # prune_model()
    finetune_model()


if __name__ == "__main__":
    main()
