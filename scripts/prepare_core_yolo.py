"""
Prepare DocLayNet CORE dataset for YOLOv8 training.

- Reads COCO annotations (train / val / test)
- Converts COCO bounding boxes to YOLO format
- Copies images
- Creates YOLO-compatible directory structure

Expected input structure:
data/doclaynet_core/
├─ PNG/
│  └─ *.png
└─ COCO/
   ├─ train.json
   ├─ val.json
   └─ test.json
"""

import json
import shutil
import collections
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CORE_DIR = Path("data/doclaynet_core")
COCO_DIR = CORE_DIR / "COCO"
IMG_SRC_DIR = CORE_DIR / "PNG"

YOLO_OUT_DIR = Path("data/yolo_core")

SPLITS = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
}

CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

# ============================================================
# Utility functions
# ============================================================

def load_coco(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    image_id_to_anns = collections.defaultdict(list)
    for ann in annotations:
        # COCO category_id is 1-based → YOLO expects 0-based
        if ann["category_id"] != -1:
            ann["category_id"] -= 1
        image_id_to_anns[ann["image_id"]].append(ann)

    return images, image_id_to_anns


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox: [x, y, width, height]
    YOLO bbox: [x_center, y_center, width, height] (normalized)
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def prepare_split(split_name, coco_file):
    print(f"\n=== Preparing split: {split_name} ===")

    images, image_id_to_anns = load_coco(COCO_DIR / coco_file)

    img_out_dir = YOLO_OUT_DIR / split_name / "images"
    lbl_out_dir = YOLO_OUT_DIR / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        img_name = img["file_name"]
        img_id = img["id"]
        img_w = img["width"]
        img_h = img["height"]

        src_img_path = IMG_SRC_DIR / img_name
        dst_img_path = img_out_dir / img_name

        if not src_img_path.exists():
            print(f"[WARN] Missing image: {src_img_path}")
            continue

        # Copy image
        shutil.copy2(src_img_path, dst_img_path)

        # Create label file
        label_lines = []
        for ann in image_id_to_anns.get(img_id, []):
            cls_id = ann["category_id"]
            bbox = ann["bbox"]

            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(bbox, img_w, img_h)
            label_lines.append(
                f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
            )

        label_path = lbl_out_dir / f"{Path(img_name).stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print(f"[OK] {split_name} prepared: {len(images)} images")


# ============================================================
# Main
# ============================================================

def main():
    print("Preparing DocLayNet CORE for YOLOv8")
    print("----------------------------------")

    for split, coco_file in SPLITS.items():
        prepare_split(split, coco_file)

    print("\n=== DONE ===")
    print(f"YOLO dataset available at: {YOLO_OUT_DIR.resolve()}")
    print("\nUse the following class order in data.yaml:")
    for i, c in enumerate(CLASSES):
        print(f"{i}: {c}")


if __name__ == "__main__":
    main()