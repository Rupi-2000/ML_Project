import json
import csv
import random
from pathlib import Path
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

EXTRA_JSON_DIR = Path("data/doclaynet_extra/JSON")
OUT_DIR = Path("data/text_dataset")

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

random.seed(RANDOM_SEED)

TARGET_CLASSES = {
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
}

# ============================================================
# Helpers
# ============================================================

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def iter_json_files():
    return EXTRA_JSON_DIR.glob("*.json")


# ============================================================
# Pass 1: determine document split
# ============================================================

def build_doc_split():
    pdf_ids = set()

    for json_file in iter_json_files():
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        pdf_name = meta.get("original_filename")

        if pdf_name:
            pdf_ids.add(pdf_name)

    pdf_ids = list(pdf_ids)
    random.shuffle(pdf_ids)

    n = len(pdf_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": set(pdf_ids[:n_train]),
        "val": set(pdf_ids[n_train:n_train + n_val]),
        "test": set(pdf_ids[n_train + n_val:]),
    }


# ============================================================
# Pass 2: stream → CSV
# ============================================================

def write_split_csv(split_name, doc_ids):
    out_path = OUT_DIR / f"{split_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "original_filename",
                "page_no",
                "label",
                "text"
            ]
        )
        writer.writeheader()

        count = 0

        for json_file in iter_json_files():
            with open(json_file, encoding="utf-8") as jf:
                data = json.load(jf)

            meta = data.get("metadata", {})
            original_filename = meta.get("original_filename")

            if original_filename not in doc_ids:
                continue

            with open(json_file, encoding="utf-8") as jf:
                data = json.load(jf)

            meta = data.get("metadata", {})
            original_filename = meta.get("original_filename")
            label = meta.get("doc_category")

            if label not in TARGET_CLASSES:
                continue

            page_no = meta.get("page_no")

            texts = [
                c.get("text", "")
                for c in data.get("cells", [])
                if c.get("text") and c.get("text").strip()
            ]

            if not texts:
                continue

            writer.writerow({
                "doc_id": json_file.stem,
                "original_filename": original_filename,
                "page_no": page_no,
                "label": label,
                "text": normalize(" ".join(texts)),
            })

            count += 1
            if count % 5000 == 0:
                print(f"[{split_name}] {count} pages written")

        print(f"[OK] {split_name}: {count} samples")


# ============================================================
# Main
# ============================================================

def main():
    print("Preparing DocLayNet EXTRA (optimized, streaming)")
    print("------------------------------------------------")

    split = build_doc_split()

    for name, docs in split.items():
        write_split_csv(name, docs)

    print("\n[FINISHED]")
    print("Output:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
