import json
from pathlib import Path

# -----------------------------
# Class mapping (fix!)
# -----------------------------
DOC_CLASS_TO_ID = {
    "financial_reports": 0,
    "scientific_articles": 1,
    "laws_and_regulations": 2,
    "government_tenders": 3,
    "manuals": 4,
    "patents": 5,
}

def generate_labels(coco_json, image_dir, label_dir):
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    written = 0
    skipped = 0

    for img in coco["images"]:
        fname = img["file_name"]
        doc_class = img.get("doc_category")

        # Sicherheitschecks
        if doc_class not in DOC_CLASS_TO_ID:
            skipped += 1
            continue

        if not (image_dir / fname).exists():
            skipped += 1
            continue

        class_id = DOC_CLASS_TO_ID[doc_class]
        out = label_dir / f"{Path(fname).stem}.txt"
        out.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")
        written += 1

    print(f"{coco_json.name}: written={written}, skipped={skipped}")


# -----------------------------
# Run per split
# -----------------------------
BASE = Path("./data/yolo_dataset")
COCO = Path("./data/doclaynet_core")

generate_labels(
    COCO / "COCO/train.json",
    BASE / "train/images",
    BASE / "train/labels_docclass",
)

generate_labels(
    COCO / "COCO/val.json",
    BASE / "val/images",
    BASE / "val/labels_docclass",
)

generate_labels(
    COCO / "COCO/test.json",
    BASE / "test/images",
    BASE / "test/labels_docclass",
)