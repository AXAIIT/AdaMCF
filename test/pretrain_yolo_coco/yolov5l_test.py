from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _extract_last_match(pattern: str, text: str) -> str | None:
    m = None
    for mm in re.finditer(pattern, text, flags=re.MULTILINE):
        m = mm
    return m.group(0) if m else None


def run_val(
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
    # 显式使用 YOLOv5 val.py 常用/默认评测参数，避免“到底是不是官方设置”的歧义
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
    # 关闭 YOLOv5 自动 pip 安装依赖，避免 requirements 噪声/失败
    env["YOLOv5_AUTOINSTALL"] = "0"

    p = subprocess.run(cmd, cwd=str(repo_dir), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        # 失败时再把完整输出抛出来便于定位
        sys.stdout.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)

    combined = (p.stdout or "") + "\n" + (p.stderr or "")

    # 解析最终汇总指标（"all ..."）
    map50 = None
    map5095 = None
    all_line = None
    lines = [ln.rstrip("\n") for ln in combined.splitlines() if ln.strip()]
    for ln in reversed(lines):
        s = ln.strip()
        if s.startswith("all "):
            all_line = s
            parts = re.split(r"\s+", s)
            # 末尾通常是 ... mAP50 mAP50-95
            if len(parts) >= 7:
                try:
                    map50 = float(parts[-2])
                    map5095 = float(parts[-1])
                except Exception:
                    pass
            break

    speed_line = _extract_last_match(r"^Speed:.*$", combined)
    results_line = _extract_last_match(r"^Results saved to .*?$", combined)

    # 精简输出：只打印必要信息
    print("[yolov5-val]")
    print(f"weights: {weights}")
    print(f"data   : {data_yaml}")
    print(f"imgsz  : {imgsz} | batch: {batch} | device: {device}")
    print(f"conf   : {conf_thres} | iou: {iou_thres} | max_det: {max_det} | half: {half} | augment: {augment}")
    if all_line:
        print(all_line)
    if speed_line:
        print(speed_line)
    if results_line:
        print(results_line)

    return map50, map5095, all_line, results_line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="data yaml 路径（默认使用同目录 coco2017_yolov5.yaml）")
    ap.add_argument("--device", type=str, default="0", help="例如 'cpu' / '0' / 'cuda:0'（默认 0）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--half", action="store_true", help="启用 FP16（一般不影响指标，主要影响速度/显存）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    repo_dir = root / "model" / "yolov5-master"
    weights = root / "model" / "yolov5l.pt"
    data_yaml = Path(args.data) if args.data else (root / "coco2017_yolov5.yaml")

    if not repo_dir.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found: {repo_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")

    map50, map5095, _, _ = run_val(
        repo_dir=repo_dir,
        weights=weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        half=args.half,
        augment=False,
    )

    print("\n[metrics]")
    print(f"mAP@0.5       : {map50}")
    print(f"mAP@0.5:0.95  : {map5095}")


if __name__ == "__main__":
    main()