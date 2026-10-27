from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _find_instances_json(coco_root: Path, split: str) -> Path:
    """
    你已调整好目录结构，这里只按标准 COCO2017 位置读取：
      coco_root/annotations/instances_{split}2017.json
    """
    p = coco_root / "annotations" / f"instances_{split}2017.json"
    if not p.exists():
        raise FileNotFoundError(f"instances json not found: {p}")
    return p


def validate_coco_layout(coco_root: Path) -> None:
    """仅校验目录结构，不创建软链接。"""
    need = [
        coco_root / "images" / "train2017",
        coco_root / "images" / "val2017",
        coco_root / "annotations" / "instances_train2017.json",
        coco_root / "annotations" / "instances_val2017.json",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"COCO layout invalid, missing:\n{msg}")


def coco_instances_to_yolo_labels(instances_json: Path, images_dir: Path, labels_dir: Path) -> None:
    """
    将 COCO instances_*.json 转为 YOLO txt labels。
    - class id 按 COCO categories 的 id 排序映射到 [0..79]
    - 跳过 iscrowd=1
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    with instances_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco.get("categories", []), key=lambda x: x["id"])
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(cats)}

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    image_id_to_anns = defaultdict(list)
    for a in anns:
        image_id_to_anns[a["image_id"]].append(a)

    for im in images:
        file_name = im["file_name"]
        w = float(im["width"])
        h = float(im["height"])

        img_path = images_dir / file_name
        if not img_path.exists():
            img_path2 = images_dir / Path(file_name).name
            if not img_path2.exists():
                raise FileNotFoundError(f"image not found for label gen: {img_path}")
            file_name = Path(file_name).name

        label_path = labels_dir / (Path(file_name).stem + ".txt")
        lines: list[str] = []

        for a in image_id_to_anns.get(im["id"], []):
            if int(a.get("iscrowd", 0)) == 1:
                continue
            cat_id = a["category_id"]
            if cat_id not in cat_id_to_cls:
                continue
            cls = cat_id_to_cls[cat_id]

            x, y, bw, bh = map(float, a["bbox"])
            if bw <= 0 or bh <= 0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            bw2 = max(0.0, x2 - x1)
            bh2 = max(0.0, y2 - y1)
            if bw2 <= 0 or bh2 <= 0:
                continue

            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            wn = bw2 / w
            hn = bh2 / h

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(yaml_path: Path, coco_root: Path) -> None:
    yaml_text = f"""# Auto-generated for YOLOv5 on COCO2017
path: {coco_root}
train: images/train2017
val: images/val2017

nc: 80
names: [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
  'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
  'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
  'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
  'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
  'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
  'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
  'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def _clean_yolov5_cache(coco_root: Path) -> None:
    for c in (
        coco_root / "val2017.cache",
        coco_root / "train2017.cache",
        coco_root / "images" / "val2017.cache",
        coco_root / "images" / "train2017.cache",
    ):
        try:
            if c.exists():
                c.unlink()
        except Exception:
            pass


def main() -> None:
    # 直接写死 COCO 根目录路径（不再从命令行读取）
    coco_root = Path("/workspace/data/coco2017")
    root = Path(__file__).resolve().parent

    if not coco_root.exists():
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    # 固定输出 yaml 到当前脚本目录
    out_yaml = root / "coco2017_yolov5.yaml"

    # 固定参数：只生成 val labels（如需 train，把 include_train 改 True）
    include_train = True
    force = True

    validate_coco_layout(coco_root)
    _clean_yolov5_cache(coco_root)

    images_val = coco_root / "images" / "val2017"

    inst_val = _find_instances_json(coco_root, "val")
    labels_val = coco_root / "labels" / "val2017"
    if force or (not labels_val.exists()) or (not any(labels_val.glob("*.txt"))):
        print(f"[info] generating val labels from: {inst_val}")
        coco_instances_to_yolo_labels(inst_val, images_dir=images_val, labels_dir=labels_val)
        print(f"[info] val labels written to: {labels_val}")
    else:
        print(f"[info] val labels already exist: {labels_val}")

    if include_train:
        images_train = coco_root / "images" / "train2017"
        inst_train = _find_instances_json(coco_root, "train")
        labels_train = coco_root / "labels" / "train2017"
        if force or (not labels_train.exists()) or (not any(labels_train.glob("*.txt"))):
            print(f"[info] generating train labels from: {inst_train}")
            coco_instances_to_yolo_labels(inst_train, images_dir=images_train, labels_dir=labels_train)
            print(f"[info] train labels written to: {labels_train}")
        else:
            print(f"[info] train labels already exist: {labels_train}")

    if force or (not out_yaml.exists()):
        write_data_yaml(out_yaml, coco_root)
        print(f"[info] data yaml written to: {out_yaml}")
    else:
        print(f"[info] data yaml exists: {out_yaml}")


if __name__ == "__main__":
    main()