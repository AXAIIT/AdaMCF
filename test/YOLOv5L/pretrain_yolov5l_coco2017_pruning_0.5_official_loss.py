from __future__ import annotations

import os
import re
import sys
import time
import math
import random
import subprocess
import copy
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from thop import profile
import warnings


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["YOLO_NO_AUTO_INSTALL"] = "True"
os.environ["ULTRALYTICS_SETTINGS"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "0"


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Stage2 开关（默认关闭）
# =========================
ENABLE_STAGE2 = True          # True: 进行 Stage2（关 mosaic/mixup 等）
CLOSE_MOSAIC_EPOCHS = 20       # Stage2 训练的 epoch 数（仅当 ENABLE_STAGE2=True 生效）

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
    from utils.torch_utils import ModelEMA
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


def _build_subprocess_env(repo_dir: Path) -> dict[str, str]:
    """确保 subprocess 能 torch.load() 我们保存的整模型对象（可能依赖工程内模块）。"""
    env = os.environ.copy()
    extra = [str(repo_dir), str(PROJECT_ROOT)]
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([old] if old else []))

    env["PYTHONUNBUFFERED"] = "1"

    env["YOLOv5_AUTOINSTALL"] = "0"
    env["YOLO_NO_AUTO_INSTALL"] = "True"
    env["ULTRALYTICS_SETTINGS"] = "0"
    env["ULTRALYTICS_AUTOINSTALL"] = "0"

    env["PYTHONWARNINGS"] = "ignore"
    return env


def _cleanup_cuda(tag: str = "") -> None:
    """在启动子进程(train.py/val.py)前，尽量释放父进程占用的 CUDA 资源。"""
    if not torch.cuda.is_available():
        return
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if tag:
            free, total = torch.cuda.mem_get_info()
            print(f"[cuda-cleanup] {tag} free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB", flush=True)
    except Exception:
        pass


def _pick_torch_and_cli_device() -> tuple[torch.device, str]:
    """
    统一 device 选择：
    - torch 侧：cuda:0
    - YOLOv5 CLI 侧：传 "0"（表示当前可见 GPU 的第0张）
    可用环境变量覆盖：
      * YOLOV5_DEVICE: 传给 train.py/val.py 的 --device
    """
    if torch.cuda.is_available():
        cli_dev = os.getenv("YOLOV5_DEVICE", "0").strip()
        return torch.device("cuda:0"), (cli_dev if cli_dev else "0")
    return torch.device("cpu"), "cpu"


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
    verbose: bool = True,
) -> tuple[float | None, float | None, str | None, str | None]:
    d = str(device).strip().lower()
    if d == "cuda":
        device = "3" if torch.cuda.is_available() else "cpu"
    elif d.startswith("cuda:"):
        device = d.split(":", 1)[1]

    cmd = [
        sys.executable,
        str(repo_dir / "val.py"),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--imgsz", str(imgsz),
        "--batch-size", str(batch),
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

    env = _build_subprocess_env(repo_dir)

    p = subprocess.run(cmd, cwd=str(repo_dir), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stdout.write(p.stdout or "")
        sys.stderr.write(p.stderr or "")
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

    if verbose:
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


def save_hyp(hyp: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)


def load_yolov5_model_from_pt(weights: Path, device: torch.device) -> torch.nn.Module:
    # 先加载到 CPU，避免 checkpoint 内部记录了不存在的 CUDA 索引（如 cuda:3）
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
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
    return model.float().to(device)


def save_yolov5_ckpt_dict(model: torch.nn.Module, out_path: Path, ema: Any = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ema is None:
        ema_obj = model
    elif hasattr(ema, "ema"):
        ema_obj = ema.ema
    elif hasattr(ema, "state_dict"):
        ema_obj = ema
    else:
        ema_obj = model

    ckpt = {
        "model": model,
        "ema": ema_obj,
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
    n_alpha_val = int(getattr(ALPHA_CFG, "n_alpha_val", 0) or 0)
    eval_batch_limit = int(getattr(ALPHA_CFG, "eval_batch_limit", 0) or 0)
    start_after_period = int(getattr(ALPHA_CFG, "start_after_period", 0) or 0)
    eval_batch_grow_every = int(getattr(ALPHA_CFG, "eval_batch_grow_every", 1) or 1)

    alpha_batch_size = int(getattr(ALPHA_CFG, "alpha_batch_size", 0) or 0) or int(train_batch)

    if n_alpha_val <= 0:
        num_val_microbatches = 1
    else:
        n_imgs = min(n_alpha_val, int(trainset_len) if trainset_len > 0 else n_alpha_val)
        num_val_microbatches = int(math.ceil(n_imgs / float(alpha_batch_size)))
        num_val_microbatches = max(1, num_val_microbatches)

    if eval_batch_limit > 0:
        num_val_microbatches = min(num_val_microbatches, eval_batch_limit)

    info = {
        "n_alpha_val": n_alpha_val,
        "eval_batch_limit": eval_batch_limit,
        "alpha_batch_size": alpha_batch_size,
        "num_val_microbatches": num_val_microbatches,
        "start_after_period": start_after_period,
        "eval_batch_grow_every": eval_batch_grow_every,
    }
    return alpha_batch_size, num_val_microbatches, info


def _get_optimizer_period(optimizer) -> int:
    """尽量从优化器/后端读取当前 pruning period。读取不到则回退 0。"""
    for name in ("cur_p", "cur_period", "current_period", "period_idx"):
        if hasattr(optimizer, name):
            try:
                return int(getattr(optimizer, name))
            except Exception:
                pass
    if hasattr(optimizer, "alpha_scheduler") and hasattr(optimizer.alpha_scheduler, "backend"):
        backend = optimizer.alpha_scheduler.backend
        for name in ("cur_p", "cur_period", "current_period", "period_idx"):
            if hasattr(backend, name):
                try:
                    return int(getattr(backend, name))
                except Exception:
                    pass
    return 0


def _num_val_microbatches_for_period(
    *,
    cur_p: int,
    total_cached: int,
    start_after_period: int,
    eval_batch_grow_every: int,
    eval_batch_limit: int,
) -> int:
    """
    对齐 alpha_scheduler.py 的增长逻辑：
      grow_k = max(0, cur_p - start_after_period) // eval_batch_grow_every
      want   = min(eval_batch_limit, 1 + grow_k)
      use_n  = min(total_cached, max(1, want))
    """
    total_cached = max(1, int(total_cached))
    grow_every = max(1, int(eval_batch_grow_every))
    start_p = int(start_after_period)

    if int(cur_p) < start_p:
        want = 1
    else:
        grow_k = max(0, int(cur_p) - start_p) // grow_every
        want = 1 + grow_k

    if int(eval_batch_limit) > 0:
        want = min(int(eval_batch_limit), want)

    return min(total_cached, max(1, int(want)))


@torch.no_grad()
def _grab_alpha_batches(train_loader, n_batches: int):
    batches = []
    n_batches = max(1, int(n_batches))

    if len(train_loader) == 0:
        return batches

    it = iter(train_loader)
    for _ in range(n_batches):
        try:
            imgs, targets, *_ = next(it)
        except StopIteration:
            it = iter(train_loader)
            imgs, targets, *_ = next(it)

        imgs = imgs.float() / 255.0
        batches.append((imgs, targets))

    return batches


class _ThopYOLOv5Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            return y[0]
        return y


def compute_flops_with_thop(model: torch.nn.Module, input_size=(1, 3, 640, 640)) -> tuple[float, float]:
    model_to_copy = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_cpu = copy.deepcopy(model_to_copy).to("cpu").eval()
    wrapped = _ThopYOLOv5Wrapper(model_cpu).eval()
    dummy = torch.randn(*input_size).to("cpu")
    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(dummy,), verbose=False)
    flops_m = float(macs) * 2.0 / 1e6
    params_m = float(params) / 1e6
    return flops_m, params_m


def train_one_epoch_yolov5(
    model: torch.nn.Module,
    train_loader,
    loss_fn: ComputeLoss,
    optimizer,
    device: torch.device,
    alpha_batches=None,
    alpha_runtime: dict[str, int] | None = None,
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
    model.train()
    optimizer.zero_grad(set_to_none=True)

    nb = len(train_loader)
    warmup_epochs = float(hyp.get("warmup_epochs", 0.0))
    nw = max(int(round(warmup_epochs * nb)), 100) if warmup_epochs > 0 else 0

    lrf = float(hyp.get("lrf", 0.1))

    def lf(e: int):
        E = max(1, int(epochs))
        return ((1.0 + math.cos(math.pi * e / E)) / 2.0) * (1.0 - lrf) + lrf

    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg.get("lr", 0.0))

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

    for i, (imgs, targets, *_rest) in enumerate(train_loader):
        ni = ni0 + i

        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device, non_blocking=True)

        if nw > 0 and ni <= nw:
            xi = [0, nw]
            accumulate_now = int(max(1, round(float(np.interp(ni, xi, [1.0, float(nbs) / float(batch_size)])))))
            for j, pg in enumerate(optimizer.param_groups):
                is_bias_group = (j == 0) and (len(optimizer.param_groups) > 1)
                start_lr = warmup_bias_lr if is_bias_group else 0.0
                end_lr = float(pg["initial_lr"]) * float(lf(epoch))
                pg["lr"] = float(np.interp(ni, xi, [start_lr, end_lr]))
                if "momentum" in pg:
                    pg["momentum"] = float(np.interp(ni, xi, [warmup_momentum, momentum]))
        else:
            accumulate_now = max(round(float(nbs) / float(batch_size)), 1)

        with torch.amp.autocast("cuda", enabled=bool(amp_enabled and device.type == "cuda")):
            pred = model(imgs)
            loss, _ = loss_fn(pred, targets)

        if amp_enabled and scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if hasattr(optimizer, "will_enter_new_period_on_next_step") and optimizer.will_enter_new_period_on_next_step():
            if alpha_batches:
                rt = alpha_runtime or {}
                cur_p = _get_optimizer_period(optimizer)
                use_n = _num_val_microbatches_for_period(
                    cur_p=cur_p,
                    total_cached=len(alpha_batches),
                    start_after_period=int(rt.get("start_after_period", 0)),
                    eval_batch_grow_every=int(rt.get("eval_batch_grow_every", 1)),
                    eval_batch_limit=int(rt.get("eval_batch_limit", len(alpha_batches))),
                )

                cursor = int(rt.get("cursor", 0))
                for _ in range(use_n):
                    idx = cursor % len(alpha_batches)
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
                    cursor += 1

                rt["cursor"] = cursor
                rt["last_use_n"] = use_n
                rt["last_cur_p"] = cur_p

        do_step = (ni - last_opt_step) >= int(accumulate_now) or (i == nb - 1)
        if do_step:
            if amp_enabled and scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None and hasattr(ema, "update"):
                ema.update(model)
            last_opt_step = ni

        lr_now = float(optimizer.param_groups[0].get("lr", 0.0))
        total_lr += lr_now
        total_loss += float(loss.detach().item())
        n += 1

    dt = max(time.time() - t0, 1e-8)
    avg_loss = total_loss / max(n, 1)
    avg_lr = total_lr / max(n, 1)
    it_per_sec = n / dt

    eta_seconds = (epochs - epoch - 1) * dt
    eta_h = int(eta_seconds // 3600)
    eta_m = int((eta_seconds % 3600) // 60)

    print(
        f"Epoch {epoch+1}/{epochs} 训练\n"
        f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
        f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
        f"估计剩余: {eta_h}h:{eta_m}m"
    )

    ni_end = ni0 + nb
    return avg_loss, ni_end, last_opt_step


def prune_model(seed: int = 1) -> None:
    print("=" * 60, flush=True)
    print("开始执行剪枝流程...", flush=True)
    print("=" * 60, flush=True)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed
    )
    random.seed(seed)

    repo_dir = YOLO_REPO
    data_yaml = BASE_DIR / "coco2017_yolov5.yaml"
    weights = BASE_DIR / "model" / "yolov5l.pt"

    device, device_str = _pick_torch_and_cli_device()

    hyp = load_hyp(repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml")

    imgsz = 640
    batch = 16
    workers = 16
    learning_rate = 1e-3
    group_sparsity = 0.5
    warmup_epochs = 0
    pruning_epochs = 120
    total_epochs = warmup_epochs + pruning_epochs

    hyp["warmup_epochs"] = float(warmup_epochs)
    hyp["lr0"] = float(learning_rate)
    hyp["warmup_bias_lr"] = 0.0

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== [0] baseline eval ===")
    print("正在评估原始模型性能...")
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
    print(f"剪枝前map50={(map50_before or 0)*100:.2f}% 剪枝前map5095={(map5095_before or 0)*100:.2f}%")

    print("\n=== [0.1] baseline flops/params (thop, cpu) ===")
    print("正在计算原始模型FLOPs和参数量...")
    baseline_model_for_cost = load_yolov5_model_from_pt(weights, device=torch.device("cpu")).eval()
    flops_before_m, params_before_m = compute_flops_with_thop(
        baseline_model_for_cost,
        input_size=(1, 3, imgsz, imgsz),
    )
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    print("\n=== [1] load model & dataloaders (prune) ===")
    print("正在加载模型和数据加载器...")
    model = load_yolov5_model_from_pt(weights, device=device)

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

    alpha_batch_size, num_val_microbatches, alpha_info = _alpha_cfg_for_yolo(
        train_batch=batch,
        trainset_len=len(train_dataset),
    )
    print(
        "[alpha-cfg] "
        f"n_alpha_val={alpha_info['n_alpha_val']} "
        f"alpha_batch_size={alpha_info['alpha_batch_size']} "
        f"eval_batch_limit={alpha_info['eval_batch_limit']} "
        f"start_after_period={alpha_info['start_after_period']} "
        f"eval_batch_grow_every={alpha_info['eval_batch_grow_every']} "
        f"-> num_val_microbatches={alpha_info['num_val_microbatches']}"
    )

    print("正在收集Alpha批次数据...")
    alpha_batches = _grab_alpha_batches(train_loader, n_batches=num_val_microbatches)

    alpha_runtime = {
        "cursor": 0,
        "start_after_period": int(alpha_info["start_after_period"]),
        "eval_batch_grow_every": int(alpha_info["eval_batch_grow_every"]),
        "eval_batch_limit": int(alpha_info["eval_batch_limit"] or len(alpha_batches)),
    }

    print("\n=== [2] init OTO ===")
    print("正在初始化OTO...")
    
    for p in model.parameters():
        p.requires_grad = True
    model.eval() 

    dummy_input = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = model(dummy_input)

    oto = OTO(model=model, dummy_input=dummy_input)
    oto.visualize(view=False, out_dir=out_dir)

    loss_fn = ComputeLoss(model)

    print("\n=== [3] pruning phase (OTO/HESSO) ===")
    print("开始剪枝阶段训练...")
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
            alpha_runtime=alpha_runtime,
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
            log_interval=steps_per_epoch,
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

        if ema is not None and hasattr(ema, "update_attr"):
            try:
                ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            except Exception:
                pass

        temp_ckpt = out_dir / "temp_prune_epoch.pt"
        save_yolov5_ckpt_dict(model, temp_ckpt, ema=ema)

        map50, map5095, all_line, _ = run_yolov5_val(
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
        
        if all_line:
            print(all_line)

        m50_str = f"{map50*100:.2f}%" if map50 is not None else "?"
        m95_str = f"{map5095*100:.2f}%" if map5095 is not None else "?"

        print(
            f"--- [Prune] Epoch {epoch + 1}/{total_epochs} Test: "
            f"mAP@0.5 {m50_str}, mAP@0.5:0.95 {m95_str}",
            file=sys.stdout,
            flush=True
        )

    print("\n=== [4] construct subnet ===", flush=True)
    print("正在构建子网络...", flush=True)
    oto.construct_subnet(out_dir=out_dir)
    compressed_path = Path(oto.compressed_model_path)
    print(f"[info] oto compressed model: {compressed_path}")

    compressed_obj = torch.load(compressed_path, map_location=device, weights_only=False)
    if isinstance(compressed_obj, dict) and "model" in compressed_obj and hasattr(compressed_obj["model"], "state_dict"):
        compressed_model = compressed_obj["model"]
    else:
        compressed_model = compressed_obj
    if not hasattr(compressed_model, "state_dict"):
        raise TypeError(f"Unexpected compressed object type: {type(compressed_model)}")

    compressed_model = compressed_model.float().to(device)
    compressed_model.hyp = hyp
    compressed_model.nc = 80
    compressed_model.gr = 1.0

    pruned_cfg_path = out_dir / "yolov5_pruned_cfg.yaml"
    _save_pruned_cfg_if_possible(compressed_model, pruned_cfg_path)

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    save_yolov5_ckpt_dict(compressed_model, pruned_ckpt_path)
    

    print("\n=== [4.1] pruned flops/params (thop, cpu) ===")
    print("正在计算剪枝后模型FLOPs和参数量...")
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
    print("正在评估剪枝后模型性能...")
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

    print("\n剪枝流程完成！")
    _cleanup_cuda("after prune_model")


def auto_calibrate_yolov5_yaml(model: torch.nn.Module, original_yaml: dict) -> dict:
    """
    自动校准 YOLOv5 的 YAML 配置 (静态名单版)。
    遍历物理模型层，读取真实的 out_channels，并将更新回写到 yaml 配置字典中。
    
    Q: 为什么只修改 args[0] (out_channels) 而不修改 in_channels?
    A: 因为 YOLOv5 的 YAML 格式标准只定义输出通道 ([out, k, s, p...])。
       'in_channels' 是在模型构建时(parse_model)根据'from'字段指向的上一层自动推断的。
       只要我们把每一层的'out'修正了，下一层的'in'自然就自动对齐了。
    """
    print("\n[Auto-Calibrate] 正在执行YAML配置自动校准 (Static List)...", flush=True)
    
    new_yaml = copy.deepcopy(original_yaml)
    
    if not hasattr(model, 'model'):
        print("[Warn] 模型没有 .model 属性，跳过校准。")
        return original_yaml
    physical_layers = list(model.model)
    
    yaml_backbone = new_yaml.get('backbone', [])
    yaml_head = new_yaml.get('head', [])
    yaml_layers = yaml_backbone + yaml_head
    
    if len(yaml_layers) != len(physical_layers):
        print(f"[Warn] 层数不匹配: YAML={len(yaml_layers)}, 物理模型={len(physical_layers)}。无法安全校准，跳过。")
        return original_yaml

    modules_check_list = {
        'Conv', 'GhostConv', 'Bottleneck', 'GhostBottleneck', 
        'SPP', 'SPPF', 'DWConv', 'MixConv2d', 'Focus', 
        'CrossConv', 'C3', 'C3TR', 'C3SPP', 'C3Ghost', 
        'C3x', 'C2', 'C2f', 'RepConv' # 包含常见的变体
    }

    changed_count = 0
    
    # 5. 并行遍历
    for i, (layer_cfg, layer_module) in enumerate(zip(yaml_layers, physical_layers)):
        if len(layer_cfg) < 4: continue
        
        module_name = layer_cfg[2]
        args = layer_cfg[3]
        
        if module_name in modules_check_list:
            real_channels = None
            
            if hasattr(layer_module, 'c2'):
                real_channels = layer_module.c2
            
            elif hasattr(layer_module, 'out_channels'):
                real_channels = layer_module.out_channels
                
            elif hasattr(layer_module, 'cv3') and hasattr(layer_module.cv3, 'conv') and hasattr(layer_module.cv3.conv, 'out_channels'): # C3
                real_channels = layer_module.cv3.conv.out_channels
            elif hasattr(layer_module, 'conv') and hasattr(layer_module.conv, 'out_channels'): # GhostConv / Bottleneck
                real_channels = layer_module.conv.out_channels
            elif hasattr(layer_module, 'cv2') and hasattr(layer_module.cv2, 'conv') and hasattr(layer_module.cv2.conv, 'out_channels'): # BottleneckCSP / C2f
                 real_channels = layer_module.cv2.conv.out_channels

            if real_channels is not None and len(args) > 0 and isinstance(args[0], int):
                cfg_channels = args[0]
                
                if cfg_channels != real_channels:
                    args[0] = int(real_channels)
                    print(f"  Layer {i:<3} {module_name:<15}: YAML_cfg={cfg_channels:<5} -> Real_weight={real_channels:<5} [已修正]")
                    changed_count += 1

    print(f"[Auto-Calibrate] 完成。共修正 {changed_count} 处配置不一致。\n", flush=True)
    return new_yaml


def _save_pruned_cfg_if_possible(model: torch.nn.Module, out_path: Path) -> Path | None:
    """
    尝试把 pruned 模型的 yaml/cfg 保存成文件，供官方 train.py --cfg 使用。
    返回 cfg 路径；如果拿不到则返回 None。
    """
    y = getattr(model, "yaml", None)
    if y is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(y, dict):
        calibrated_y = auto_calibrate_yolov5_yaml(model, y)
        
        model.yaml = calibrated_y  
        
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(calibrated_y, f, sort_keys=False)
        return out_path

    if isinstance(y, str):
        p = Path(y)
        if p.exists():
            return p
        
        if "nc:" in y or "backbone:" in y or "head:" in y:
            try:
                y_dict = yaml.safe_load(y)
                if isinstance(y_dict, dict):
                    calibrated_y = auto_calibrate_yolov5_yaml(model, y_dict)
                    with out_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(calibrated_y, f, sort_keys=False)
                    return out_path
            except Exception:
                pass 
                
            with out_path.open("w", encoding="utf-8") as f:
                f.write(y)
            return out_path

    return None


# ---------------------------
# finetune via official train.py
# ---------------------------
def _run_subprocess_with_filtered_console(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    full_log_path: Path,
    console_mode: str = "epoch",  # "epoch" | "quiet" | "full"
) -> None:
    """
    - full_log_path: 保存 train.py 的完整 stdout/stderr（含 tqdm）
    - console_mode:
        * "full"  : 原样打印到控制台（可能很冗余）
        * "quiet" : 控制台不打印 train.py 输出
        * "epoch" : 控制台只打印每个 epoch 的汇总信息（经过解析计算）
    """
    full_log_path.parent.mkdir(parents=True, exist_ok=True)

    if console_mode == "full":
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True)
        if p.returncode != 0:
            raise SystemExit(p.returncode)
        return

    # quiet/epoch：把子进程输出全部写文件；控制台仅输出过滤/汇总后的信息
    with full_log_path.open("w", encoding="utf-8") as log_f:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None
        
        current_epoch_idx = 0
        epoch_box_loss = []
        epoch_obj_loss = []
        epoch_cls_loss = []
        epoch_gpu_mem = []
        epoch_items_seen = 0
        last_print_time = time.time()

        # 新增：ETA 相关状态
        total_epochs_planned = None
        epoch_times: list[float] = []

        best_map50 = -1.0
        best_map5095 = -1.0
        best_map50_epoch = -1
        best_map5095_epoch = -1

        re_train_line = re.compile(r"^\s*\d+/\d+\s+([0-9.]+[GM])\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+\d+")
        re_start_epochs = re.compile(r"Starting training for\s+(\d+)\s+epochs", re.IGNORECASE)

        def _fmt_eta(seconds: float) -> str:
            s = max(0, int(seconds))
            h = s // 3600
            m = (s % 3600) // 60
            ss = s % 60
            return f"{h:02d}:{m:02d}:{ss:02d}"

        for raw in iter(p.stdout.readline, ""):
            if raw == "":
                break

            log_f.write(raw)

            if console_mode == "quiet":
                continue

            line = raw.strip()
            if not line:
                continue

            match = re_train_line.match(line)
            if match:
                try:
                    gpu_mem_str = match.group(1) # e.g. 2.1G
                    box_l = float(match.group(2))
                    obj_l = float(match.group(3))
                    cls_l = float(match.group(4))
                    
                    if "G" in gpu_mem_str:
                         epoch_gpu_mem.append(float(gpu_mem_str.replace("G", "")))
                    elif "M" in gpu_mem_str:
                         epoch_gpu_mem.append(float(gpu_mem_str.replace("M", "")) / 1024.0)
                    
                    epoch_box_loss.append(box_l)
                    epoch_obj_loss.append(obj_l)
                    epoch_cls_loss.append(cls_l)
                    epoch_items_seen += 1
                except Exception:
                    pass
                continue

            if line.startswith("all "):
                current_epoch_idx += 1
                
                avg_box = sum(epoch_box_loss) / max(1, len(epoch_box_loss))
                avg_obj = sum(epoch_obj_loss) / max(1, len(epoch_obj_loss))
                avg_cls = sum(epoch_cls_loss) / max(1, len(epoch_cls_loss))
                avg_loss = avg_box + avg_obj + avg_cls
                avg_mem = sum(epoch_gpu_mem) / max(1, len(epoch_gpu_mem))
                
                dt = time.time() - last_print_time
                last_print_time = time.time()
                epoch_times.append(dt)

                eta_txt = "?"
                if total_epochs_planned and current_epoch_idx <= total_epochs_planned:
                    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))
                    remain_epochs = max(0, total_epochs_planned - current_epoch_idx)
                    eta_txt = _fmt_eta(remain_epochs * avg_epoch_time)

                print(
                    f"Epoch {current_epoch_idx} 评估\n"
                    f"平均Loss: {avg_loss:.4f} (Box:{avg_box:.4f} Obj:{avg_obj:.4f} Cls:{avg_cls:.4f}), "
                    f"显存: {avg_mem:.2f}G, 用时: {dt:.2f}s, 预计剩余: {eta_txt}"
                , flush=True)

                parts = re.split(r"\s+", line)
                map50_str = "?"
                map5095_str = "?"
                map50_val = None
                map5095_val = None
                if len(parts) >= 7:
                    try:
                        map50_val = float(parts[-2])
                        map5095_val = float(parts[-1])
                        map50_str = f"{map50_val * 100:.2f}%"
                        map5095_str = f"{map5095_val * 100:.2f}%"
                    except Exception:
                        pass

                if map50_val is not None and map50_val > best_map50:
                    best_map50 = map50_val
                    best_map50_epoch = current_epoch_idx
                if map5095_val is not None and map5095_val > best_map5095:
                    best_map5095 = map5095_val
                    best_map5095_epoch = current_epoch_idx

                print(f"验证结果: mAP@0.5 {map50_str}, mAP@0.5:0.95 {map5095_str}", flush=True)

                # 新增：每个epoch都打印“截至当前最优”
                if best_map50_epoch > 0 and best_map5095_epoch > 0:
                    print(
                        f"当前最优: mAP@0.5={best_map50*100:.2f}%@Epoch{best_map50_epoch}, "
                        f"mAP@0.5:0.95={best_map5095*100:.2f}%@Epoch{best_map5095_epoch}",
                        flush=True
                    )
                
            elif "Starting training for" in line:
                m = re_start_epochs.search(line)
                if m:
                    try:
                        total_epochs_planned = int(m.group(1))
                    except Exception:
                        total_epochs_planned = None
                print(f"[yolov5-train] {line}", flush=True)
                last_print_time = time.time()
            
        rc = p.wait()

        if console_mode in {"epoch", "quiet"}:
            print("\n=== [finetune-official] Summary ===", flush=True)
            if best_map50_epoch > 0:
                print(f"最佳 mAP@0.5: {best_map50*100:.2f}% @ Epoch {best_map50_epoch}", flush=True)
            else:
                print("最佳 mAP@0.5: 未解析到有效结果", flush=True)

            if best_map5095_epoch > 0:
                print(f"最佳 mAP@0.5:0.95: {best_map5095*100:.2f}% @ Epoch {best_map5095_epoch}", flush=True)
            else:
                print("最佳 mAP@0.5:0.95: 未解析到有效结果", flush=True)

            if best_map50_epoch > 0 and best_map5095_epoch > 0:
                if best_map50_epoch == best_map5095_epoch:
                    print(f"两个指标最佳周期一致：Epoch {best_map50_epoch}", flush=True)
                else:
                    print(
                        f"两个指标最佳周期不同：mAP@0.5 在 Epoch {best_map50_epoch}，"
                        f"mAP@0.5:0.95 在 Epoch {best_map5095_epoch}",
                        flush=True,
                    )

        if rc != 0:
            raise SystemExit(rc)


def run_yolov5_train_official(
    repo_dir: Path,
    *,
    weights: Path,
    data_yaml: Path,
    hyp_yaml: Path,
    imgsz: int,
    batch_size: int,
    epochs: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
    cfg_yaml: Path | None = None,
    optimizer: str = "SGD",
    cos_lr: bool = False,
    rect: bool = False,
    exist_ok: bool = True,
    seed: int = 0,
    console_mode: str = "epoch",     # "epoch" | "quiet" | "full"
    full_log_path: Path | None = None,
) -> Path:
    project.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(repo_dir / "train.py"),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--hyp", str(hyp_yaml),
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--imgsz", str(int(imgsz)),
        "--device", str(device),
        "--workers", str(int(workers)),
        "--optimizer", str(optimizer),
        "--project", str(project),
        "--name", str(name),
        "--seed", str(int(seed)),
        "--noplots",
    ]
    if cfg_yaml is not None:
        cmd += ["--cfg", str(cfg_yaml)]
    if exist_ok:
        cmd.append("--exist-ok")
    if cos_lr:
        cmd.append("--cos-lr")
    if rect:
        cmd.append("--rect")

    env = _build_subprocess_env(repo_dir)

    out_dir = project / name
    if full_log_path is None:
        full_log_path = out_dir / "train_full.log"

    print("\n=== [finetune-official] running train.py ===", flush=True)
    print("[cmd]", " ".join(cmd), flush=True)
    print(f"[finetune-official] console_mode={console_mode}", flush=True)
    print(f"[finetune-official] full log -> {full_log_path}", flush=True)

    _run_subprocess_with_filtered_console(
        cmd,
        cwd=str(repo_dir),
        env=env,
        full_log_path=full_log_path,
        console_mode=console_mode,
    )

    print(f"[finetune-official] train output dir -> {out_dir}", flush=True)
    return out_dir


def finetune_model(seed: int = 1, enable_stage2: bool = False) -> None:
    print("=" * 60, flush=True)
    print("开始执行微调流程...", flush=True)
    print("=" * 60, flush=True)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed
    )
    random.seed(seed)

    _, device_str = _pick_torch_and_cli_device()

    repo_dir = YOLO_REPO
    data_yaml = BASE_DIR / "coco2017_yolov5.yaml"

    imgsz = 640
    batch = 16
    workers = 8

    finetune_epochs = 80

    if bool(enable_stage2):
        close_mosaic_epochs = min(int(CLOSE_MOSAIC_EPOCHS), max(0, finetune_epochs - 1))
    else:
        close_mosaic_epochs = 0

    e2 = int(close_mosaic_epochs)
    e1 = int(finetune_epochs - e2)

    out_dir = BASE_DIR / "cache_yolov5l_coco2017_pruned_0_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    pruned_ckpt_path = out_dir / "yolov5l_pruned_ckpt.pt"
    if not pruned_ckpt_path.exists():
        raise FileNotFoundError(f"未找到剪枝后的模型 ckpt，请先运行 prune_model(): {pruned_ckpt_path}")

    cfg_for_train = None 
    print(f"[finetune-official] cfg_for_train -> {cfg_for_train} (MANDATORY: None for pruned model)")

    print("\n=== [finetune-official] pre-check (val & flops) ===")
    print("正在评估微调前(剪枝后)模型性能...")
    map50_ft_start, map5095_ft_start, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=pruned_ckpt_path,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        verbose=True,
    )
    print(f"[finetune-pre] mAP@0.5={(map50_ft_start or 0)*100:.2f}% mAP@0.5:0.95={(map5095_ft_start or 0)*100:.2f}%")

    print("正在计算微调前(剪枝后)模型FLOPs和参数量...")
    model_ft_start = load_yolov5_model_from_pt(pruned_ckpt_path, device=torch.device("cpu")).eval()
    flops_ft_start_m, params_ft_start_m = compute_flops_with_thop(
        model_ft_start,
        input_size=(1, 3, imgsz, imgsz),
    )
    print(f"[finetune-pre] FLOPs: {flops_ft_start_m:.2f} M")
    print(f"[finetune-pre] 参数量: {params_ft_start_m:.2f} M")
    # ===============================

    print("\n=== [finetune-official] config ===")
    print(f"  enable_stage2={enable_stage2} close_mosaic_epochs={close_mosaic_epochs} total_epochs={finetune_epochs}")

    hyp0 = load_hyp(repo_dir / "data" / "hyps" / "hyp.scratch-low.yaml")
    hyp0["lr0"] = 0.005
    hyp0["lrf"] = 0.1
    hyp0["warmup_epochs"] = 5.0
    hyp0["warmup_bias_lr"] = 0.0
    hyp0["weight_decay"] = 1e-4

    project = out_dir / "official_train_runs"
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Stage1
    e1 = int(max(1, finetune_epochs - close_mosaic_epochs))
    e2 = int(max(0, close_mosaic_epochs))

    hyp1_path = out_dir / f"hyp_ft_stage1_{stamp}.yaml"
    save_hyp(copy.deepcopy(hyp0), hyp1_path)
    print(f"[finetune-official] stage1 hyp -> {hyp1_path} | epochs={e1}")

    print(f"[finetune-official] 开始Stage1训练... (epochs={e1})")
    name1 = f"finetune_stage1_{stamp}"
    out1 = run_yolov5_train_official(
        repo_dir,
        weights=pruned_ckpt_path,
        data_yaml=data_yaml,
        hyp_yaml=hyp1_path,
        imgsz=imgsz,
        batch_size=batch,
        epochs=e1,
        device=device_str,
        workers=workers,
        project=project,
        name=name1,
        cfg_yaml=cfg_for_train,
        optimizer="SGD",
        cos_lr=True,
        rect=False,
        exist_ok=True,
        seed=seed,
    )

    last1 = out1 / "weights" / "last.pt"
    best1 = out1 / "weights" / "best.pt"
    stage1_weight = best1 if best1.exists() else last1
    if not stage1_weight.exists():
        raise FileNotFoundError(f"[finetune-official] stage1 未找到 last/best: {out1}")

    final_eval_weight = stage1_weight

    # Stage2（最后10~20个epoch）
    if e2 > 0:
        hyp2 = copy.deepcopy(hyp0)
        hyp2["mosaic"] = 0.0
        hyp2["mixup"] = 0.0
        if "copy_paste" in hyp2:
            hyp2["copy_paste"] = 0.0
        hyp2["lr0"] = 0.0015  # Stage2再降一点

        hyp2_path = out_dir / f"hyp_ft_stage2_nomosaic_{stamp}.yaml"
        save_hyp(hyp2, hyp2_path)
        print(f"[finetune-official] stage2 hyp -> {hyp2_path} | epochs={e2}")

        name2 = f"finetune_stage2_nomosaic_{stamp}"
        out2 = run_yolov5_train_official(
            repo_dir,
            weights=stage1_weight,
            data_yaml=data_yaml,
            hyp_yaml=hyp2_path,
            imgsz=imgsz,
            batch_size=batch,
            epochs=e2,
            device=device_str,
            workers=workers,
            project=project,
            name=name2,
            cfg_yaml=cfg_for_train,
            optimizer="SGD",
            cos_lr=False,
            rect=False,
            exist_ok=True,
            seed=seed,
        )

        last2 = out2 / "weights" / "last.pt"
        best2 = out2 / "weights" / "best.pt"
        final_eval_weight = best2 if best2.exists() else last2
        if not final_eval_weight.exists():
            raise FileNotFoundError(f"[finetune-official] stage2 未找到 last/best: {out2}")
    else:
        print("[finetune-official] Stage2 disabled (default).")

    print(f"\n=== [finetune-official] eval final weight: {final_eval_weight} ===")
    print("正在评估最终微调模型性能...")
    map50, map5095, *_ = run_yolov5_val(
        repo_dir=repo_dir,
        weights=final_eval_weight,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
    )
    
    v_m50_start = map50_ft_start or 0.0
    v_m95_start = map5095_ft_start or 0.0
    v_m50_end = map50 or 0.0
    v_m95_end = map5095 or 0.0

    print("\n" + "=" * 20 + " [Finetune] Final Summary " + "=" * 20, flush=True)
    print(f"微调前(剪枝后) mAP@0.5:      {v_m50_start*100:.2f}%", flush=True)
    print(f"微调前(剪枝后) mAP@0.5:0.95: {v_m95_start*100:.2f}%", flush=True)
    print("-" * 66, flush=True)
    print(f"微调后最终 mAP@0.5:          {v_m50_end*100:.2f}%", flush=True)
    print(f"微调后最终 mAP@0.5:0.95:     {v_m95_end*100:.2f}%", flush=True)
    print("-" * 66, flush=True)
    print(f"mAP@0.5 绝对提升:            {(v_m50_end - v_m50_start)*100:+.2f}%", flush=True)
    print(f"mAP@0.5:0.95 绝对提升:       {(v_m95_end - v_m95_start)*100:+.2f}%", flush=True)
    print("=" * 66, flush=True)

    _cleanup_cuda("before launching official train.py")

    print("\n微调流程完成！")

def main():
    print("=" * 60, flush=True)
    print("程序启动 - 开始YOLOv5模型剪枝与微调流程", flush=True)
    print("=" * 60, flush=True)

    print("\n开始执行剪枝步骤...", flush=True)
    # prune_model()

    print("\n开始执行微调步骤...", flush=True)
    finetune_model(enable_stage2=ENABLE_STAGE2)

    print("\n" + "=" * 60, flush=True)
    print("整个YOLOv5模型剪枝与微调流程已全部完成！", flush=True)
    print("=" * 60, flush=True)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

if __name__ == "__main__":
    main()