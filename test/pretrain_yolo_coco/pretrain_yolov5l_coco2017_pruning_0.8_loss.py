from __future__ import annotations

import os
import re
import sys
import time
import math
import random
import subprocess
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast, GradScaler
from thop import profile

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]
YOLO_REPO = BASE_DIR / "model" / "yolov5-master"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(YOLO_REPO) not in sys.path:
    sys.path.insert(0, str(YOLO_REPO))

from utils.torch_utils import smart_optimizer
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.loss import ComputeLoss

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG


def _extract_last_match(pattern: str, text: str) -> str | None:
    m = None
    for mm in re.finditer(pattern, text, flags=re.MULTILINE):
        m = mm
    return m.group(0) if m else None


def run_yolov5_val(repo_dir: Path, weights: Path, data_yaml: Path, imgsz: int, batch: int, device: str, conf_thres: float = 0.001,
    iou_thres: float = 0.6, max_det: int = 300, half: bool = False, augment: bool = False, ) -> tuple[float | None, float | None, str | None, str | None]:
    """调用 YOLOv5 官方 val.py，输出只保留必要行，并返回 mAP。"""
    cmd = [
        sys.executable,
        str(repo_dir / "val.py"),
        "--weights",
        str(weights),
        "--data",
        str(data_yaml),
        "--img",
        str(imgsz),
        "--batch",
        str(batch),
        "--task",
        "val",
        "--device",
        str(device),
        "--conf-thres",
        str(conf_thres),
        "--iou-thres",
        str(iou_thres),
        "--max-det",
        str(max_det),
    ]
    if half:
        cmd.append("--half")
    if augment:
        cmd.append("--augment")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["YOLOv5_AUTOINSTALL"] = "0"

    p = subprocess.run(cmd, cwd=str(repo_dir), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stdout.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)

    combined = (p.stdout or "") + "\n" + (p.stderr or "")
    lines = [ln.rstrip("\n") for ln in combined.splitlines() if ln.strip()]

    map50 = None
    map5095 = None
    all_line = None
    for ln in reversed(lines):
        s = ln.strip()
        if s.startswith("all "):
            all_line = s
            parts = re.split(r"\s+", s)
            if len(parts) >= 7:
                try:
                    map50 = float(parts[-2])
                    map5095 = float(parts[-1])
                except Exception:
                    pass
            break

    speed_line = _extract_last_match(r"^Speed:.*$", combined)
    results_line = _extract_last_match(r"^Results saved to .*?$", combined)

    print("[yolov5-val]")
    if all_line:
        print(all_line)
    if speed_line:
        print(speed_line)
    if results_line:
        print(results_line)

    return map50, map5095, all_line, results_line


def load_hyp(hyp_path: Path) -> dict[str, Any]:
    if not hyp_path.exists():
        raise FileNotFoundError(f"hyp not found: {hyp_path}")
    with hyp_path.open("r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)
    if not isinstance(hyp, dict):
        raise ValueError("hyp yaml format invalid")
    return hyp


def load_yolov5_model_from_pt(weights: Path, device: torch.device) -> torch.nn.Module:
    """
    尽量兼容：
    - YOLOv5 官方 .pt（dict，含 'model'/'ema'）
    - 直接 torch.save(model) 的 .pt（nn.Module）
    """
    ckpt = torch.load(weights, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ("ema", "model"):
            if key in ckpt and hasattr(ckpt[key], "state_dict"):
                model = ckpt[key]
                break
        else:
            raise ValueError(f"Unsupported YOLOv5 ckpt keys: {list(ckpt.keys())[:20]}")
    elif hasattr(ckpt, "state_dict"):
        model = ckpt
    else:
        raise ValueError(f"Unsupported weights format: {type(ckpt)}")

    model = model.float().to(device)
    return model


def save_yolov5_ckpt_dict(model: torch.nn.Module, out_path: Path) -> None:
    """保存成 YOLOv5 常见 ckpt dict，val.py 加载更稳。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model,
        "ema": None,
        "optimizer": None,
        "epoch": -1,
        "best_fitness": None,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, out_path)


def prepare_dataloaders(
    data_yaml: Path,
    imgsz: int,
    batch: int,
    workers: int,
    model_stride: int,
    hyp: dict[str, Any],
):
    data_dict = check_dataset(str(data_yaml))  # 解析 path/train/val
    train_path = data_dict["train"]
    val_path = data_dict["val"]

    train_loader, train_dataset = create_dataloader(
        path=train_path,
        imgsz=imgsz,
        batch_size=batch,
        stride=model_stride,
        single_cls=False,
        hyp=hyp,
        augment=True,
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix="train: ",
    )

    val_loader, val_dataset = create_dataloader(
        path=val_path,
        imgsz=imgsz,
        batch_size=batch,
        stride=model_stride,
        single_cls=False,
        hyp=hyp,
        augment=False,
        cache=False,
        pad=0.5,
        rect=True,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix="val: ",
    )

    return (train_loader, train_dataset), (val_loader, val_dataset)


def _alpha_cfg_for_yolo(*, train_batch: int, trainset_len: int) -> tuple[int, int, dict[str, int]]:
    """
    将 only_train_once.optimizer.alpha_scheduler.ALPA_CFG 的“样本级超参”
    映射为 YOLO 侧要抓取的 alpha batches 数量。

    - n_alpha_val: alpha 更新评估用的样本数（不是 batch 数）
    - eval_batch_limit: 最多允许使用多少个 batch 做 alpha 评估（限速）
    - alpha_batch_size: 如果配置里有，优先用它作为 alpha 评估 batch_size（否则用 train_batch）
    """
    n_alpha_val = int(getattr(ALPHA_CFG, "n_alpha_val", 0) or 0)
    eval_batch_limit = int(getattr(ALPHA_CFG, "eval_batch_limit", 0) or 0)
    alpha_batch_size = int(getattr(ALPHA_CFG, "alpha_batch_size", 0) or 0) or int(train_batch)

    # 将“样本数”换算成“batch 数”
    if n_alpha_val <= 0:
        alpha_batches_n = 1
    else:
        n_imgs = min(n_alpha_val, int(trainset_len) if trainset_len > 0 else n_alpha_val)
        alpha_batches_n = int(math.ceil(n_imgs / float(alpha_batch_size)))
        alpha_batches_n = max(1, alpha_batches_n)

    # 限制最大 batch 数（如果配置有）
    if eval_batch_limit > 0:
        alpha_batches_n = min(alpha_batches_n, eval_batch_limit)

    info = {
        "n_alpha_val": n_alpha_val,
        "eval_batch_limit": eval_batch_limit,
        "alpha_batch_size": alpha_batch_size,
        "alpha_batches_n": alpha_batches_n,
    }
    return alpha_batch_size, alpha_batches_n, info


@torch.no_grad()
def _grab_alpha_batches(train_loader, n_batches: int):
    """
    从 YOLO 的 train_loader 抓取 n_batches 个 batch，保存在 CPU。
    注意：这里只做 float/255 的归一化，但不搬 GPU；用到时再搬，避免显存占用。
    """
    batches = []
    it = iter(train_loader)
    for _ in range(max(1, int(n_batches))):
        try:
            imgs, targets, *_ = next(it)
        except StopIteration:
            break
        imgs = imgs.float() / 255.0  # YOLOv5 训练习惯
        batches.append((imgs, targets))
    return batches


def train_one_epoch_yolov5(
    model: torch.nn.Module,
    train_loader,
    loss_fn: ComputeLoss,
    optimizer,
    device: torch.device,
    alpha_batches=None,
    *,
    epoch: int | None = None,
    epochs: int | None = None,
    hyp: dict[str, Any] | None = None,
    ni0: int = 0,
    # --- finetune 对齐 YOLOv5 train.py 需要的参数 ---
    batch_size: int | None = None,
    nbs: int = 64,  # nominal batch size in YOLOv5
    accumulate_base: int = 1,  # 基础梯度累积步数（max(round(nbs/bs),1)）
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
) -> tuple[float, int]:
    """
    返回：(avg_loss, ni_end)
    - ni0: global step 起始值（用于 warmup 按 step 插值）
    - batch_size/nbs/accumulate_base: 用于对齐官方 accumulate + warmup-accumulate 插值
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total = 0.0
    total_lr = 0.0  # 新增：记录总 LR
    n = 0
    t0 = time.time()

    nb = len(train_loader)
    ni = ni0  # global step

    # YOLOv5 官方 cosine（lf 按 epoch）
    def lf(e: int):
        lrf = float(hyp.get("lrf", 0.1)) if hyp else 0.1
        E = max(1, int(epochs or 1))
        return ((1.0 + math.cos(math.pi * e / E)) / 2.0) * (1.0 - lrf) + lrf

    # YOLOv5 官方 warmup 参数
    if hyp and epoch is not None and epochs is not None:
        warmup_epochs = float(hyp.get("warmup_epochs", 0.0))
        warmup_momentum = float(hyp.get("warmup_momentum", 0.8))
        warmup_bias_lr = float(hyp.get("warmup_bias_lr", 0.1))
        momentum = float(hyp.get("momentum", 0.937))
        nw = max(int(round(warmup_epochs * nb)), 100) if warmup_epochs > 0 else 0
    else:
        warmup_momentum = 0.0
        warmup_bias_lr = 0.0
        momentum = 0.0
        nw = 0

    # 记录每个 param group 的“目标 lr”（base_lr * lf(epoch)）
    if hyp and epoch is not None and epochs is not None:
        lrfactor = lf(epoch)
        target_lrs = [pg.get("initial_lr", pg["lr"]) * lrfactor for pg in optimizer.param_groups]
    else:
        target_lrs = None

    for imgs, targets, *_ in train_loader:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device, non_blocking=True)

        # --- 对齐官方：warmup 期间 accumulate 也按 step 插值 ---
        # 官方：accumulate = max(1, round(interp(ni, [0,nw], [1, nbs/bs])))
        if batch_size and nw > 0 and ni <= nw and hyp and epoch is not None and epochs is not None:
            xi = [0, nw]
            accumulate_now = int(
                max(
                    1,
                    round(float(np.interp(ni, xi, [1.0, float(nbs) / float(batch_size)]))),
                )
            )
        else:
            accumulate_now = int(max(1, accumulate_base))

        # ---- 官方 warmup：按 step 覆盖 lr/momentum（你已有，这里保留并要求 target_lrs 存在）----
        if nw > 0 and ni <= nw and hyp and epoch is not None and epochs is not None and target_lrs is not None:
            xi = [0, nw]
            for j, pg in enumerate(optimizer.param_groups):
                is_bias_group = (j == (len(optimizer.param_groups) - 1))
                start_lr = warmup_bias_lr if is_bias_group else 0.0
                end_lr = target_lrs[j]
                pg["lr"] = float(np.interp(ni, xi, [start_lr, end_lr]))
                if "momentum" in pg:
                    pg["momentum"] = float(np.interp(ni, xi, [warmup_momentum, momentum]))
        # ----------------------------------------------------------------------------------------

        # 修改：使用 torch.amp.autocast，并根据 amp_enabled 控制是否开启
        with torch.amp.autocast("cuda", enabled=bool(amp_enabled and device.type == "cuda")):
            pred = model(imgs)
            loss, _ = loss_fn(pred, targets)

            # 对齐官方 accumulate：loss 要除以 accumulate
            loss = loss / float(accumulate_now)

        if amp_enabled and scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # OTO(HESSO) 的 alpha 更新（剪枝阶段用，finetune 不会走到这里）
        if hasattr(optimizer, "will_enter_new_period_on_next_step") and optimizer.will_enter_new_period_on_next_step():
            if alpha_batches:
                idx = int(ni) % len(alpha_batches)
                a_imgs_cpu, a_targets_cpu = alpha_batches[idx]
                a_imgs = a_imgs_cpu.to(device, non_blocking=True)
                a_targets = a_targets_cpu.to(device, non_blocking=True)

                optimizer.alpha_scheduler.update(
                    model=model,
                    criterion=loss_fn,
                    inputs=a_imgs,
                    targets=a_targets,
                    t_in_period=0,
                )

        # 对齐官方：按 accumulate_now 才 step
        do_step = ((ni + 1) % accumulate_now == 0) or ((n + 1) == nb)
        if do_step:
            if amp_enabled and scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_lr = optimizer.param_groups[0]["lr"]
        total_lr += current_lr

        total += float(loss.detach().item()) * float(accumulate_now)
        n += 1
        ni += 1

    dt = max(time.time() - t0, 1e-8)
    avg_loss = total / max(n, 1)
    avg_lr = total_lr / max(n, 1)
    it_per_sec = n / dt
    
    eta_str = "?"
    epoch_str = f"{epoch+1}/{epochs}" if (epoch is not None and epochs is not None) else "?"
    
    if epoch is not None and epochs is not None:
        eta_seconds = (epochs - epoch - 1) * dt
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_str = f"{eta_h}h:{eta_m}m"

    print(f"Epoch {epoch_str} 评估\n"
          f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
          f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
          f"估计剩余: {eta_str}")

    return avg_loss, ni


class _ThopYOLOv5Wrapper(torch.nn.Module):
    """thop 需要 Tensor 输出；YOLOv5 eval() 常返回 (pred, aux)，这里仅返回 pred。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            return y[0]
        return y


def compute_flops_with_thop(model: torch.nn.Module, input_size=(1, 3, 640, 640)) -> tuple[float, float]:
    """
    使用 thop 计算 FLOPs 和参数量（与 mobilenet 脚本一致的口径）：
      - 返回 flops_m(MFLOPs), params_m(M Params)
      - FLOPs = MACs * 2
    """
    if isinstance(model, torch.nn.DataParallel):
        model_to_copy = model.module
    else:
        model_to_copy = model

    model_cpu = copy.deepcopy(model_to_copy).to("cpu").eval()
    wrapped = _ThopYOLOv5Wrapper(model_cpu).eval()
    dummy = torch.randn(*input_size).to("cpu")

    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(dummy,), verbose=False)

    flops_m = float(macs) * 2.0 / 1e6
    params_m = float(params) / 1e6
    return flops_m, params_m


def prune_model(seed: int = 1) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    repo_dir = YOLO_REPO
    data_yaml = BASE_DIR / "coco2017_yolov5.yaml"
    weights = BASE_DIR / "model" / "yolov5l.pt"

    device_str = "0"  # "cpu" / "0" / "cuda:0"
    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")

    hyp = load_hyp(repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml")

    imgsz = 640
    batch = 16
    workers = 8
    learning_rate = 1e-3
    group_sparsity = 0.8
    warmup_epochs = 5
    pruning_epochs = 150
    total_epochs = warmup_epochs + pruning_epochs

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_8"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== [0] baseline eval ===")
    map50_before, map5095_before, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=weights,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        half=False,
        augment=False,
    )
    print(f"剪枝前map50={map50_before:.4f} 剪枝前map5095={map5095_before:.4f}")

    print("\n=== [0.1] baseline flops/params (thop, cpu) ===")
    baseline_model_for_cost = load_yolov5_model_from_pt(weights, device=torch.device("cpu")).eval()
    flops_before_m, params_before_m = compute_flops_with_thop(
        baseline_model_for_cost,
        input_size=(1, 3, imgsz, imgsz),
    )
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    print("\n=== [1] load model & dataloaders (prune) ===")
    model = load_yolov5_model_from_pt(weights, device=device)

    model.hyp = hyp
    model.nc = 80
    model.gr = 1.0

    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    (train_loader, train_dataset), _ = prepare_dataloaders(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        model_stride=stride,
        hyp=hyp,
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    pruning_steps = int(pruning_epochs * steps_per_epoch)
    start_pruning_step = int(warmup_steps)

    # alpha 超参：直接使用 ALPHA_CFG（样本数/limit），YOLO 只负责抓 batch
    alpha_batch_size, alpha_batches_n, alpha_info = _alpha_cfg_for_yolo(
        train_batch=batch,
        trainset_len=len(train_dataset),
    )
    print(
        "[alpha-cfg] "
        f"n_alpha_val={alpha_info['n_alpha_val']} "
        f"alpha_batch_size={alpha_info['alpha_batch_size']} "
        f"eval_batch_limit={alpha_info['eval_batch_limit']} "
        f"-> alpha_batches_n={alpha_info['alpha_batches_n']}"
    )

    alpha_batches = _grab_alpha_batches(train_loader, n_batches=alpha_batches_n)

    print("\n=== [2] init OTO ===")
    dummy_input = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=out_dir)

    # 确保参数可训练
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    # 确保 dummy_input 尺寸与设备正确，并先跑一遍前向，防止构图失败
    dummy_input = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = model(dummy_input)

    # 重新初始化 OTO（如果已有同名变量，直接覆盖）
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=out_dir)

    # 调试看下 OTO 识别到的可剪枝参数数目
    try:
        n_prunable = len(oto.graph.prunable_parameters)
    except Exception:
        n_prunable = None
    print(f"[debug] prunable params = {n_prunable}")

    loss_fn = ComputeLoss(model)

    print("\n=== [3] pruning phase (OTO/HESSO) ===")
    optimizer_prune = oto.hesso(
        variant="sgd",
        lr=learning_rate,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=5e-4,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        device=device,
    )
    optimizer_prune.zero_grad(set_to_none=True)

    ni = 0
    for epoch in range(total_epochs):
        print(f"\n[prune] epoch {epoch+1}/{total_epochs}")
        _, ni = train_one_epoch_yolov5(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer_prune,
            device=device,
            alpha_batches=alpha_batches,
            epoch=epoch,
            epochs=total_epochs,
            hyp=None,
            ni0=ni,
            amp_enabled=False,
            scaler=None
        )
        if hasattr(optimizer_prune, "compute_metrics"):
            m = optimizer_prune.compute_metrics()
            print(f"Sparsity: {getattr(m, 'group_sparsity', 0.0):.4f}")

            if hasattr(optimizer_prune, "_alpha_keys") and hasattr(optimizer_prune, "alpha_scheduler"):
                try:
                    alpha_vals = optimizer_prune.alpha_scheduler.get_alpha().tolist()
                    print(dict(zip(optimizer_prune._alpha_keys, alpha_vals)))
                except Exception:
                    pass


        temp_ckpt = out_dir / "temp_prune_epoch.pt"
        save_yolov5_ckpt_dict(model, temp_ckpt)

        map50, map5095, _, _ = run_yolov5_val(
            repo_dir=repo_dir,
            weights=temp_ckpt,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device_str,
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=300,
        )
        print(
            f"--- [Prune] Epoch {epoch + 1}/{total_epochs} Test: "
            f"mAP@0.5 {map50}, mAP@0.5:0.95 {map5095}",
            file=sys.stdout,
            flush=True
        )

    print("\n=== [4] construct subnet ===")
    oto.construct_subnet(out_dir=out_dir)
    compressed_path = Path(oto.compressed_model_path)
    print(f"[info] oto compressed model: {compressed_path}")

    compressed_model = torch.load(compressed_path, map_location=device, weights_only=False)
    compressed_model = compressed_model.float().to(device)
    compressed_model.hyp = hyp
    compressed_model.nc = 80
    compressed_model.gr = 1.0

    print("\n=== [4.1] pruned flops/params (thop, cpu) ===")
    compressed_model_cpu = copy.deepcopy(compressed_model).to("cpu").eval()
    flops_after_m, params_after_m = compute_flops_with_thop(
        compressed_model_cpu,
        input_size=(1, 3, imgsz, imgsz),
    )
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    eps = 1e-8
    pruned_flops_m = max(flops_before_m - flops_after_m, 0.0)
    pruned_params_m = max(params_before_m - params_after_m, 0.0)
    pruned_flops_percent = pruned_flops_m / (flops_before_m + eps) * 100.0
    pruned_params_percent = pruned_params_m / (params_before_m + eps) * 100.0
    flops_compression_ratio = (flops_before_m / (flops_after_m + eps)) if flops_after_m > 0 else float("inf")
    params_compression_ratio = (params_before_m / (params_after_m + eps)) if params_after_m > 0 else float("inf")

    print("\n=== [4.2] compression summary ===")
    print(f"减少的 FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少的参数量: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")
    print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x ({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    print(f"参数量压缩比: {params_compression_ratio:.2f}x ({params_before_m:.2f}M -> {params_after_m:.2f}M)")

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    save_yolov5_ckpt_dict(compressed_model, pruned_ckpt_path)
    print(f"[info] saved yolov5 ckpt: {pruned_ckpt_path}")

    print("\n=== [prune] eval pruned (before finetune) ===")
    map50_after, map5095_after, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=pruned_ckpt_path,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        half=False,
        augment=False,
    )
    print(f"map50变化: 从 {map50_before:.2f}% 变为 {map50_after:.2f}% (变化: {map50_after - map50_before:+.2f}%)")
    print(f"\nEM变化: 从 {map5095_before:.2f}% 变为 {map5095_after:.2f}% (变化: {map5095_after - map5095_before:+.2f}%)")
    

def finetune_model(seed: int = 1) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    repo_dir = YOLO_REPO
    data_yaml = BASE_DIR / "coco2017_yolov5.yaml"

    device_str = "0"
    imgsz = 640
    batch = 16
    nbs = 64
    workers = 8
    finetune_epochs = 200

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")
    hyp = load_hyp(repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml")

    baseline_weights = BASE_DIR / "model" / "yolov5l.pt"
    print("\n=== [baseline] eval unpruned (before finetune) ===")
    map50_before, map5095_before, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=baseline_weights,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        half=False,
        augment=False,
    )
    print(f"微调前(未剪枝) mAP@0.5={map50_before:.4f} mAP@0.5:0.95={map5095_before:.4f}")
    # -----------------------------------

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    if not pruned_ckpt_path.exists():
        candidates = sorted(out_dir.glob("*pruned*ckpt*.pt"))
        if candidates:
            pruned_ckpt_path = candidates[-1]
    if not pruned_ckpt_path.exists():
        raise FileNotFoundError(f"未找到剪枝后的模型 ckpt，请先运行 prune_model(): {out_dir}")

    print(f"\n=== [5] finetune phase (load: {pruned_ckpt_path}) ===")
    model = load_yolov5_model_from_pt(pruned_ckpt_path, device=device)
    model.hyp = hyp
    model.nc = 80
    model.gr = 1.0

    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    (train_loader, _), _ = prepare_dataloaders(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        model_stride=stride,
        hyp=hyp,
    )

    lr0 = float(hyp.get("lr0", 0.01))
    momentum = float(hyp.get("momentum", 0.937))
    weight_decay = float(hyp.get("weight_decay", 5e-4))
    lrf = float(hyp.get("lrf", 0.1))

    accumulate_base = max(round(nbs / batch), 1)
    scaled_wd = weight_decay * batch * accumulate_base / nbs

    loss_fn = ComputeLoss(model)

    optimizer = smart_optimizer(model, name="SGD", lr=lr0, momentum=momentum, decay=scaled_wd)

    def lf_epoch(e: int):
        E = max(1, int(finetune_epochs))
        return ((1.0 + math.cos(math.pi * e / E)) / 2.0) * (1.0 - lrf) + lrf

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_epoch)

    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"[finetune-config] bs={batch} nbs={nbs} accumulate_base={accumulate_base} weight_decay={weight_decay} scaled_wd={scaled_wd}")

    best_map50 = 0.0
    best_map5095 = 0.0
    best_epoch = -1
    best_ckpt_path = out_dir / "yolov5l_pruned_finetuned_best.pt"

    ni = 0
    for epoch in range(max(1, int(finetune_epochs))):
        print(f"\n[finetune] epoch {epoch+1}/{max(1, int(finetune_epochs))} lr={optimizer.param_groups[0]['lr']:.6g}")

        warmup_epochs_val = float(hyp.get("warmup_epochs", 0.0))
        if epoch < warmup_epochs_val:
            print(f"  [Warmup] 正在进行预热 (Epoch {epoch+1} < {warmup_epochs_val})")

        _, ni = train_one_epoch_yolov5(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            alpha_batches=None,
            epoch=epoch,
            epochs=finetune_epochs,
            hyp=hyp,
            ni0=ni,
            batch_size=batch,
            nbs=nbs,
            accumulate_base=accumulate_base,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        # 官方：每个 epoch 末 scheduler.step()
        scheduler.step()

        temp_ckpt = out_dir / "temp_finetune_epoch.pt"
        save_yolov5_ckpt_dict(model, temp_ckpt)
        
        map50, map5095, _, _ = run_yolov5_val(
            repo_dir=repo_dir,
            weights=temp_ckpt,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device_str,
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=300,
        )
        print(
            f"--- [Finetune] Epoch {epoch + 1}/{max(1, int(finetune_epochs))} Test: "
            f"mAP@0.5 {map50}, mAP@0.5:0.95 {map5095}",
            file=sys.stdout,
            flush=True
        )

        # ---- 新增：保存最佳模型逻辑 (包含 mAP@0.5:0.95 对比) ----
        current_map50 = map50 if map50 is not None else 0.0
        current_map5095 = map5095 if map5095 is not None else 0.0
        
        is_best = False
        # 优先比较 mAP@0.5
        if current_map50 > best_map50:
            is_best = True
        # 若 mAP@0.5 相等，则比较 mAP@0.5:0.95
        elif current_map50 == best_map50 and current_map50 > 0:
            if current_map5095 > best_map5095:
                is_best = True

        if is_best:
            best_map50 = current_map50
            best_map5095 = current_map5095
            best_epoch = epoch + 1
            print(f"★ 当前最佳模型更新: mAP@0.5={best_map50:.4f}, mAP@0.5:0.95={best_map5095:.4f} (Epoch {best_epoch})")
            print(f"  - 保存最佳模型到: {best_ckpt_path}")
            save_yolov5_ckpt_dict(model, best_ckpt_path)

    finetuned_ckpt_path = out_dir / "yolov5l_pruned_finetuned_ckpt.pt"
    save_yolov5_ckpt_dict(model, finetuned_ckpt_path)
    print(f"[info] saved last finetuned ckpt: {finetuned_ckpt_path}")
    print(f"[info] best finetuned ckpt: {best_ckpt_path} (Epoch {best_epoch}, mAP@0.5={best_map50:.4f})")

    print("\n=== [final] eval finetuned (Last Epoch) ===")
    map50, map5095, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=finetuned_ckpt_path,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        half=False,
        augment=False,
    )
    print("\n[metrics]")
    print(f"mAP@0.5      : {map50}")
    print(f"mAP@0.5:0.95 : {map5095}")

    if map50_before is not None and map5095_before is not None:
        print("\n[compare]")
        print(f"mAP@0.5      : {map50_before:.4f} -> {map50:.4f} (变化: {map50 - map50_before:+.4f})")
        print(f"mAP@0.5:0.95 : {map5095_before:.4f} -> {map5095:.4f} (变化: {map5095 - map5095_before:+.4f})")


def main() -> None:
    # prune_model()
    finetune_model()


if __name__ == "__main__":
    main()


