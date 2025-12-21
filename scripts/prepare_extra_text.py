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
# determine document split
# ============================================================


def collect_pdf_stats():
    pdf_pages = defaultdict(int)
    pdf_label = {}

    for json_file in iter_json_files():
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        pdf = meta.get("original_filename")
        label = meta.get("doc_category")

        if pdf and label in TARGET_CLASSES:
            pdf_pages[pdf] += 1
            pdf_label[pdf] = label

    return pdf_pages, pdf_label

def build_pdf_split_balance():
    pdf_pages, pdf_label = collect_pdf_stats()
    splits = {"train": set(), "val": set(), "test": set()}
    
    # Seiten-Zähler pro Split und Klasse initialisieren
    # Struktur: { "train": { "manuals": 120, ... }, "val": {...} }
    current_counts = {s: defaultdict(int) for s in splits}

    for label in TARGET_CLASSES:
        # PDFs dieser Klasse nach Größe absteigend sortieren
        # Das verhindert, dass dicke "Brocken" am Ende die Bilanz ruinieren
        pdfs_in_class = [p for p in pdf_label if pdf_label[p] == label]
        pdfs_in_class.sort(key=lambda p: pdf_pages[p], reverse=True)
        
        total_pages_class = sum(pdf_pages[p] for p in pdfs_in_class)
        
        # Zielwerte für diese Klasse
        targets = {
            "train": total_pages_class * TRAIN_RATIO,
            "val": total_pages_class * VAL_RATIO,
            "test": total_pages_class * (1.0 - TRAIN_RATIO - VAL_RATIO)
        }

        for pdf in pdfs_in_class:
            pages = pdf_pages[pdf]
            
            # Finde den Split, der im Vergleich zu seinem Target 
            # aktuell am meisten "unterversorgt" ist
            best_split = min(
                ["train", "val", "test"],
                key=lambda s: current_counts[s][label] / targets[s] if targets[s] > 0 else 1
            )
            
            splits[best_split].add(pdf)
            current_counts[best_split][label] += pages

    return splits


# ============================================================
# stream → CSV
# ============================================================

def write_split_csv(split_assignment):
    """
    split_assignment: Dict { "original_filename": "train" (oder "val", "test") }
    """
    # Alle Writer vorbereiten
    writers = {}
    handles = {}
    
    # Ordner erstellen und Files öffnen
    for split_name in ["train", "val", "test"]:
        out_path = OUT_DIR / f"{split_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        f = open(out_path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=["doc_id", "original_filename", "page_no", "label", "text"])
        w.writeheader()
        
        handles[split_name] = f
        writers[split_name] = w

    count = 0
    skipped = 0

    print("Starting single-pass writing...")

    # Nur EINMAL über alle JSONs iterieren
    for json_file in iter_json_files():
        with open(json_file, encoding="utf-8") as jf:
            data = json.load(jf)

        meta = data.get("metadata", {})
        original_filename = meta.get("original_filename")
        label = meta.get("doc_category")

        # Check: Ist das File relevant?
        # Wir prüfen nur, ob der Filename in unserem Assignment-Dict ist.
        # Das impliziert bereits, dass das Label korrekt war (aus Pass 1).
        if original_filename not in split_assignment:
            skipped += 1
            continue

        # Bestimmen, in welchen Split es gehört
        target_split = split_assignment[original_filename]
        
        texts = [
            c.get("text", "")
            for c in data.get("cells", [])
            if c.get("text") and c.get("text").strip()
        ]

        if not texts:
            continue

        # In den korrekten Writer schreiben
        writers[target_split].writerow({
            "doc_id": json_file.stem,
            "original_filename": original_filename,
            "page_no": meta.get("page_no"),
            "label": label,
            "text": normalize(" ".join(texts)),
        })

        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} pages...")

    # Aufräumen
    for f in handles.values():
        f.close()

    print(f"[FINISHED] Processed {count} pages. Skipped {skipped} (irrelevant class/empty).")


# ============================================================
# Main
# ============================================================

def main():
    print("Preparing DocLayNet EXTRA (optimized, streaming)")
    print("------------------------------------------------")

    # 1. Split berechnen (wie bisher)
    splits = build_pdf_split_balance()
    
    # 2. Umwandeln in ein Lookup-Dict für O(1) Zugriff:
    # { "filename1.pdf": "train", "filename2.pdf": "val", ... }
    file_to_split = {}
    for split_name, filenames in splits.items():
        for fname in filenames:
            file_to_split[fname] = split_name

    print(f"Split determined. Mapping contains {len(file_to_split)} PDFs.")

    # 3. Alles in einem Rutsch schreiben
    write_split_csv(file_to_split)

    print("Output:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
