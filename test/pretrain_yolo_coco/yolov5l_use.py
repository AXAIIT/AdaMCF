from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="例如 'cpu' / '0' / 'cuda:0'，默认自动选择")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo_dir = root / "model" / "yolov5-master"
    weights = root / "model" / "yolov5l.pt"

    img1 = repo_dir / "data" / "images" / "zidane.jpg"
    img2 = repo_dir / "data" / "images" / "bus.jpg"

    assert repo_dir.exists(), f"YOLOv5 repo not found: {repo_dir}"
    assert weights.exists(), f"weights not found: {weights}"
    assert img1.exists(), f"example image not found: {img1}"
    assert img2.exists(), f"example image not found: {img2}"

    # 通过本地 hubconf.py 加载（等价于调用 repo_dir 下的 hubconf.custom）
    model = torch.hub.load(
        repo_or_dir=str(repo_dir),
        model="custom",
        path=str(weights),
        source="local",
        device=args.device,
        autoshape=True,
        _verbose=True,
    )

    # 推理（支持 Path/str/list）
    imgs = [img1, img2]
    results = model(imgs, size=args.imgsz)

    # 打印与保存结果
    results.print()
    save_dir = root / "runs" / "yolov5l_demo"
    save_dir.mkdir(parents=True, exist_ok=True)
    results.save(save_dir=str(save_dir))
    print(f"saved to: {save_dir}")


if __name__ == "__main__":
    main()