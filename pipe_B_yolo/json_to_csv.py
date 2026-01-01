import json
import csv
from pathlib import Path

ANNOTATION_DIR = Path("./data/doclaynet_core/COCO")
YOLO_DATASET_DIR = Path("./data//yolo_dataset")

SPLITS = ["train", "val", "test"]

def convert(split: str):
    json_path = ANNOTATION_DIR / f"{split}.json"
    output_dir = YOLO_DATASET_DIR / split
    output_csv = output_dir / "labels.csv"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # DocLayNet-Format
    if isinstance(data, dict) and "images" in data:
        entries = data["images"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unbekanntes JSON-Format in {json_path}")

    rows = []
    for e in entries:
        page_id = Path(e["file_name"]).stem
        doc_class = e["doc_category"]

        rows.append({
            "page_id": page_id,
            "doc_class": doc_class
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_id", "doc_class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split}: {len(rows)} Einträge → {output_csv}")

if __name__ == "__main__":
    for split in SPLITS:
        convert(split)
