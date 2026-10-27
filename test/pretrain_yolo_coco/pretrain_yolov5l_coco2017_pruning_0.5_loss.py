from __future__ import annotations

import os
import re
import sys
import time
import math
import random
import subprocess
import copy
import argparse
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
try:
    from utils.torch_utils import ModelEMA  # yolov5 常见实现
except Exception:
    ModelEMA = None

EMA_AVAILABLE = ModelEMA is not None
if EMA_AVAILABLE:
    print("[info] ModelEMA 已找到：将启用 EMA（在代码中实例化 ModelEMA 的阶段生效）")
else:
    print("[warn] ModelEMA 未找到：将不使用 EMA（评估/对比时请注意差异）")

from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.loss import ComputeLoss

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG


# ---------------------------
# small utils
# ---------------------------
def _extract_last_match(pattern: str, text: str) -> str | None:
    m = None
    for mm in re.finditer(pattern, text, flags=re.MULTILINE):
        m = mm
    return m.group(0) if m else None


def run_yolov5_val(
    repo_dir: Path,
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    half: bool = False,
    augment: bool = False,
) -> tuple[float | None, float | None, str | None, str | None]:
    """调用 YOLOv5 官方 val.py，输出只保留必要行，并返回 mAP。"""
    cmd = [
        sys.executable,
        str(repo_dir / "val.py"),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--img", str(imgsz),
        "--batch", str(batch),
        "--task", "val",
        "--device", str(device),
        "--conf-thres", str(conf_thres),
        "--iou-thres", str(iou_thres),
        "--max-det", str(max_det),
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
    # 修复: PyTorch 2.6+ 默认 weights_only=True 会阻止加载 YOLOv5 这种包含模型结构定义的检查点
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


def save_yolov5_ckpt_dict(model: torch.nn.Module, out_path: Path, ema: Any = None) -> None:
    """保存成 YOLOv5 常见 ckpt dict，val.py 加载更稳。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model,
        "ema": getattr(ema, "ema", None) if ema is not None else None,
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
    data_dict = check_dataset(str(data_yaml))
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
        shuffle=True,
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
    将 ALPHA_CFG 的“样本级超参”映射为 YOLO 侧要抓取的 alpha batches 数量。
    """
    n_alpha_val = int(getattr(ALPHA_CFG, "n_alpha_val", 0) or 0)
    eval_batch_limit = int(getattr(ALPHA_CFG, "eval_batch_limit", 0) or 0)
    alpha_batch_size = int(getattr(ALPHA_CFG, "alpha_batch_size", 0) or 0) or int(train_batch)

    if n_alpha_val <= 0:
        alpha_batches_n = 1
    else:
        n_imgs = min(n_alpha_val, int(trainset_len) if trainset_len > 0 else n_alpha_val)
        alpha_batches_n = int(math.ceil(n_imgs / float(alpha_batch_size)))
        alpha_batches_n = max(1, alpha_batches_n)

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
    """从 train_loader 抓取 n_batches 个 batch，保存在 CPU（imgs 做 /255）。"""
    batches = []
    it = iter(train_loader)
    for _ in range(max(1, int(n_batches))):
        try:
            imgs, targets, *_ = next(it)
        except StopIteration:
            break
        imgs = imgs.float() / 255.0
        batches.append((imgs, targets))
    return batches


# ---------------------------
# thop flops
# ---------------------------
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
    thop：返回 flops_m(MFLOPs), params_m(M Params)
    FLOPs = MACs * 2
    """
    model_to_copy = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_cpu = copy.deepcopy(model_to_copy).to("cpu").eval()
    wrapped = _ThopYOLOv5Wrapper(model_cpu).eval()
    dummy = torch.randn(*input_size).to("cpu")

    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(dummy,), verbose=False)

    flops_m = float(macs) * 2.0 / 1e6
    params_m = float(params) / 1e6
    return flops_m, params_m


# ---------------------------
# core train-one-epoch (fixed)
# ---------------------------
def train_one_epoch_yolov5(
    model: torch.nn.Module,
    train_loader,
    loss_fn: ComputeLoss,
    optimizer,
    device: torch.device,
    alpha_batches=None,
    *,
    epoch: int,
    epochs: int,
    hyp: dict[str, Any],
    ni0: int,
    last_opt_step: int,
    batch_size: int,
    nbs: int,
    amp_enabled: bool,
    scaler: GradScaler | None,
    ema: Any,
    log_interval: int = 50,
) -> tuple[float, int, int]:
    """
    对齐 YOLOv5 官方逻辑：
    - warmup: lr/momentum + accumulate 插值
    - step: 用 last_opt_step 判定 (ni - last_opt_step >= accumulate)
    - EMA: optimizer.step 后紧跟 ema.update
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    nb = len(train_loader)  # batches per epoch
    # warmup iters: max(round(warmup_epochs*nb), 100)
    warmup_epochs = float(hyp.get("warmup_epochs", 0.0))
    nw = max(int(round(warmup_epochs * nb)), 100) if warmup_epochs > 0 else 0

    # lr schedule factor (cosine like official one_cycle;你这里用 cosine 形式即可)
    lrf = float(hyp.get("lrf", 0.1))
    def lf(e: int):
        E = max(1, int(epochs))
        return ((1.0 + math.cos(math.pi * e / E)) / 2.0) * (1.0 - lrf) + lrf

    # ensure param_groups has initial_lr
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg.get("lr", 0.0))

    # set epoch base lr (non-warmup baseline), warmup 会覆盖前 nw steps
    lrfactor = lf(epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = float(pg["initial_lr"]) * float(lrfactor)

    warmup_momentum = float(hyp.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(hyp.get("warmup_bias_lr", 0.1))
    momentum = float(hyp.get("momentum", 0.937))

    t0 = time.time()
    total_loss = 0.0
    total_lr = 0.0
    n = 0

    # 进度显示
    print(f"[train] epoch {epoch+1}/{epochs} | nb={nb} | warmup_iters={nw} | amp={amp_enabled}")

    for i, (imgs, targets, *_rest) in enumerate(train_loader):
        ni = ni0 + i  # integrated iters since start

        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device, non_blocking=True)

        # warmup: lr/momentum + accumulate 插值（对齐官方）
        if nw > 0 and ni <= nw:
            xi = [0, nw]
            accumulate_now = int(max(1, round(float(np.interp(ni, xi, [1.0, float(nbs) / float(batch_size)])))))
            for j, pg in enumerate(optimizer.param_groups):
                # 官方：bias 组 j==0；但如果你的 optimizer 只有 1 组（如某些自定义优化器），就不要用 bias warmup
                is_bias_group = (j == 0) and (len(optimizer.param_groups) > 1)
                start_lr = warmup_bias_lr if is_bias_group else 0.0
                end_lr = float(pg["initial_lr"]) * float(lf(epoch))
                pg["lr"] = float(np.interp(ni, xi, [start_lr, end_lr]))
                if "momentum" in pg:
                    pg["momentum"] = float(np.interp(ni, xi, [warmup_momentum, momentum]))
        else:
            accumulate_now = max(round(float(nbs) / float(batch_size)), 1)

        # forward + loss
        with autocast(enabled=bool(amp_enabled and device.type == "cuda")):
            pred = model(imgs)
            loss, _ = loss_fn(pred, targets)

        # backward
        if amp_enabled and scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # OTO(HESSO) alpha 更新：保持你原逻辑（发生在即将进入新 period 之前）
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

        # optimize: official uses (ni - last_opt_step >= accumulate)
        do_step = (ni - last_opt_step) >= int(accumulate_now) or (i == nb - 1)
        if do_step:
            if amp_enabled and scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None and hasattr(ema, "update"):
                ema.update(model)  # after optimizer.step()

            last_opt_step = ni

        # logging
        lr_now = float(optimizer.param_groups[0].get("lr", 0.0))
        total_lr += lr_now
        total_loss += float(loss.detach().item())
        n += 1

        if (i + 1) % max(1, int(log_interval)) == 0 or (i == nb - 1):
            print(
                f"  [iter {i+1:>4d}/{nb}] "
                f"loss={total_loss/n:.4f} lr={lr_now:.6g} "
                f"acc={int(accumulate_now)} ni={ni} last_step={last_opt_step}"
            )

    dt = max(time.time() - t0, 1e-8)
    avg_loss = total_loss / max(n, 1)
    avg_lr = total_lr / max(n, 1)
    it_per_sec = n / dt

    eta_seconds = (epochs - epoch - 1) * dt
    eta_h = int(eta_seconds // 3600)
    eta_m = int((eta_seconds % 3600) // 60)

    print(
        f"[epoch done] {epoch+1}/{epochs} "
        f"avg_loss={avg_loss:.4f} avg_lr={avg_lr:.6f} "
        f"throughput={it_per_sec:.2f} it/s time={dt:.2f}s ETA={eta_h}h:{eta_m}m"
    )

    ni_end = ni0 + nb
    return avg_loss, ni_end, last_opt_step


# ---------------------------
# pipeline: prune
# ---------------------------
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

    # ---------------- user config ----------------
    imgsz = 640
    batch = 16
    workers = 16
    learning_rate = 1e-3
    group_sparsity = 0.5
    warmup_epochs = 0
    pruning_epochs = 1
    total_epochs = warmup_epochs + pruning_epochs

    # 对齐：把 hyp 里的 warmup_epochs/lr0 也同步（否则剪枝起点与 warmup 不一致）
    hyp["warmup_epochs"] = float(warmup_epochs)
    hyp["lr0"] = float(learning_rate)
    # 对 HESSO/单 param-group 的情况，bias warmup 值太大没意义：直接设 0 更稳
    hyp["warmup_bias_lr"] = 0.0

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_5"
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
    print(f"[baseline] mAP@0.5={map50_before*100:.2f}% mAP@0.5:0.95={map5095_before*100:.2f}%")

    print("\n=== [0.1] baseline flops/params (thop, cpu) ===")
    baseline_model_for_cost = load_yolov5_model_from_pt(weights, device=torch.device("cpu")).eval()
    flops_before_m, params_before_m = compute_flops_with_thop(
        baseline_model_for_cost, input_size=(1, 3, imgsz, imgsz)
    )
    print(f"[baseline] FLOPs={flops_before_m:.2f}M Params={params_before_m:.2f}M")

    print("\n=== [1] load model & dataloaders (prune) ===")
    model = load_yolov5_model_from_pt(weights, device=device)

    # 修复: 显式开启所有参数的梯度，防止 OTO 优化器报错 "empty parameter list"
    for k, v in model.named_parameters():
        v.requires_grad = True

    ema = ModelEMA(model) if ModelEMA is not None else None

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

    alpha_batch_size, alpha_batches_n, alpha_info = _alpha_cfg_for_yolo(
        train_batch=batch, trainset_len=len(train_dataset)
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
    with torch.no_grad():
        _ = model(dummy_input)

    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=out_dir)
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

    # prune 阶段：为了不影响 pruning step 计数，显式设置 nbs=batch => accumulate=1
    nbs_prune = batch

    ni = 0
    last_opt_step = -1
    for epoch in range(total_epochs):
        print(f"\n[prune] epoch {epoch+1}/{total_epochs}")

        _, ni, last_opt_step = train_one_epoch_yolov5(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer_prune,
            device=device,
            alpha_batches=alpha_batches,
            epoch=epoch,
            epochs=total_epochs,
            hyp=hyp,
            ni0=ni,
            last_opt_step=last_opt_step,
            batch_size=batch,
            nbs=nbs_prune,
            amp_enabled=False,
            scaler=None,
            ema=ema,
            log_interval=50,
        )

        if hasattr(optimizer_prune, "compute_metrics"):
            m = optimizer_prune.compute_metrics()
            print(f"[prune-metrics] group_sparsity={getattr(m, 'group_sparsity', 0.0):.4f}")

            if hasattr(optimizer_prune, "_alpha_keys") and hasattr(optimizer_prune, "alpha_scheduler"):
                try:
                    alpha_vals = optimizer_prune.alpha_scheduler.get_alpha().tolist()
                    print("[alpha]", dict(zip(optimizer_prune._alpha_keys, alpha_vals)))
                except Exception:
                    pass

        # EMA 属性对齐（可选，但更稳）
        if ema is not None and hasattr(ema, "update_attr"):
            try:
                ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            except Exception:
                pass

        # 周期评估
        temp_ckpt = out_dir / "temp_prune_epoch.pt"
        save_yolov5_ckpt_dict(model, temp_ckpt, ema=ema)

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
        m50_pct = (map50 or 0.0) * 100
        m5095_pct = (map5095 or 0.0) * 100
        print(f"--- [Prune Eval] Epoch {epoch+1}/{total_epochs}: mAP@0.5={m50_pct:.2f}%, mAP@0.5:0.95={m5095_pct:.2f}%")

    print("\n=== [4] construct subnet ===")
    oto.construct_subnet(out_dir=out_dir)
    compressed_path = Path(oto.compressed_model_path)
    print(f"[info] oto compressed model: {compressed_path}")

    # 更鲁棒：复用 loader（防止是 dict/Module 两种格式）
    compressed_model = load_yolov5_model_from_pt(compressed_path, device=device)
    compressed_model.hyp = hyp
    compressed_model.nc = 80
    compressed_model.gr = 1.0

    print("\n=== [4.1] pruned flops/params (thop, cpu) ===")
    compressed_model_cpu = copy.deepcopy(compressed_model).to("cpu").eval()
    flops_after_m, params_after_m = compute_flops_with_thop(
        compressed_model_cpu, input_size=(1, 3, imgsz, imgsz)
    )
    print(f"[pruned] FLOPs={flops_after_m:.2f}M Params={params_after_m:.2f}M")

    eps = 1e-8
    pruned_flops_m = max(flops_before_m - flops_after_m, 0.0)
    pruned_params_m = max(params_before_m - params_after_m, 0.0)
    pruned_flops_percent = pruned_flops_m / (flops_before_m + eps) * 100.0
    pruned_params_percent = pruned_params_m / (params_before_m + eps) * 100.0
    flops_compression_ratio = (flops_before_m / (flops_after_m + eps)) if flops_after_m > 0 else float("inf")
    params_compression_ratio = (params_before_m / (params_after_m + eps)) if params_after_m > 0 else float("inf")

    print("\n=== [4.2] compression summary ===")
    print(f"减少FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少Params: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")
    print(f"FLOPs压缩比: {flops_compression_ratio:.2f}x ({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    print(f"Params压缩比: {params_compression_ratio:.2f}x ({params_before_m:.2f}M -> {params_after_m:.2f}M)")

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    save_yolov5_ckpt_dict(compressed_model, pruned_ckpt_path, ema=None)

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
    if map50_before is not None and map50_after is not None:
        print(f"[prune] mAP@0.5: {map50_before*100:.2f}% -> {map50_after*100:.2f}% ({(map50_after-map50_before)*100:+.2f}%)")
    if map5095_before is not None and map5095_after is not None:
        print(f"[prune] mAP@0.5:0.95: {map5095_before*100:.2f}% -> {map5095_after*100:.2f}% ({(map5095_after-map5095_before)*100:+.2f}%)")


# ---------------------------
# pipeline: finetune
# ---------------------------
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
    workers = 16
    finetune_epochs = 300
    close_mosaic_epochs = 10  # 最后 N 个 epoch 关闭 mosaic（可按需调）

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")
    hyp = load_hyp(repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml")

    # finetune 更保守
    hyp["lr0"] = 0.005
    hyp["lrf"] = 0.1
    hyp["warmup_bias_lr"] = 0.0
    # hyp["warmup_momentum"] = 0.8
    hyp["weight_decay"] = 1e-4

    lr0 = float(hyp.get("lr0", 0.01))
    momentum = float(hyp.get("momentum", 0.937))
    weight_decay = float(hyp.get("weight_decay", 5e-4))

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
    print(f"[baseline] mAP@0.5={map50_before*100:.2f}% mAP@0.5:0.95={map5095_before*100:.2f}%")

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    if not pruned_ckpt_path.exists():
        candidates = sorted(out_dir.glob("*pruned*ckpt*.pt"))
        if candidates:
            pruned_ckpt_path = candidates[-1]
    if not pruned_ckpt_path.exists():
        raise FileNotFoundError(f"未找到剪枝后的模型 ckpt，请先运行 prune_model(): {out_dir}")

    print(f"\n=== [5] finetune phase (load: {pruned_ckpt_path}) ===")
    model = load_yolov5_model_from_pt(pruned_ckpt_path, device=device)
    ema = ModelEMA(model) if ModelEMA is not None else None

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

    # official scaling: hyp['weight_decay'] *= batch_size * accumulate / nbs
    accumulate_base = max(round(nbs / batch), 1)
    scaled_wd = float(weight_decay) * float(batch) * float(accumulate_base) / float(nbs)

    loss_fn = ComputeLoss(model)
    optimizer = smart_optimizer(model, name="SGD", lr=lr0, momentum=momentum, decay=scaled_wd)
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_map50 = 0.0
    best_map5095 = 0.0
    best_epoch = -1
    best_ckpt_path = out_dir / "yolov5l_pruned_finetuned_best.pt"

    ni = 0
    last_opt_step = -1

    print(f"[finetune-config] bs={batch} nbs={nbs} accumulate_base={accumulate_base} wd_scaled={scaled_wd:g} close_mosaic={close_mosaic_epochs}")

    for epoch in range(max(1, int(finetune_epochs))):
        # close mosaic in last N epochs (simple, works for v5 dataset)
        if close_mosaic_epochs > 0 and epoch == finetune_epochs - close_mosaic_epochs:
            print(f"\n[finetune] closing mosaic in last {close_mosaic_epochs} epochs ...")
            try:
                if hasattr(train_dataset, "mosaic"):
                    train_dataset.mosaic = False
                if hasattr(train_dataset, "hyp") and isinstance(train_dataset.hyp, dict):
                    train_dataset.hyp["mosaic"] = 0.0
                    train_dataset.hyp["mixup"] = 0.0
                print("[finetune] mosaic disabled.")
            except Exception as e:
                print(f"[finetune][warn] failed to disable mosaic: {e}")

        print(f"\n[finetune] epoch {epoch+1}/{finetune_epochs}")

        _, ni, last_opt_step = train_one_epoch_yolov5(
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
            last_opt_step=last_opt_step,
            batch_size=batch,
            nbs=nbs,
            amp_enabled=amp_enabled,
            scaler=scaler,
            ema=ema,
            log_interval=50,
        )

        if ema is not None and hasattr(ema, "update_attr"):
            try:
                ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            except Exception:
                pass

        temp_ckpt = out_dir / "temp_finetune_epoch.pt"
        save_yolov5_ckpt_dict(model, temp_ckpt, ema=ema)

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
        current_map50 = map50 if map50 is not None else 0.0
        current_map5095 = map5095 if map5095 is not None else 0.0

        is_best = (current_map50 > best_map50) or (
            math.isclose(current_map50, best_map50) and current_map5095 > best_map5095
        )
        if is_best:
            best_map50 = current_map50
            best_map5095 = current_map5095
            best_epoch = epoch
            save_yolov5_ckpt_dict(model, best_ckpt_path, ema=ema)

        print(
            f"[Finetune Eval] Epoch {epoch+1}: "
            f"mAP@0.5={current_map50*100:.2f}%, mAP@0.5:0.95={current_map5095*100:.2f}% "
            f"(Best: {best_map50*100:.2f}%, {best_map5095*100:.2f}% @ Ep{best_epoch+1})"
        )

    final_eval_ckpt = best_ckpt_path if best_ckpt_path.exists() else temp_ckpt
    print("\n=== [finetune] eval best (after finetune) ===")
    map50_after, map5095_after, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=final_eval_ckpt,
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

    if map50_before is not None and map5095_before is not None:
        print("[finetune] vs unpruned baseline:")
        print(f"  mAP@0.5: {map50_before*100:.2f}% -> {map50_after*100:.2f}% ({(map50_after-map50_before)*100:+.2f}%)")
        print(f"  mAP@0.5:0.95: {map5095_before*100:.2f}% -> {map5095_after*100:.2f}% ({(map5095_after-map5095_before)*100:+.2f}%)")


def main():
    # prune_model()
    finetune_model()


if __name__ == "__main__":
    main()
