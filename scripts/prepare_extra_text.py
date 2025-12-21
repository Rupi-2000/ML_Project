import json
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# ============================================================
# Configuration
# ============================================================

CORE_COCO_DIR = Path("data/doclaynet_core/COCO")
EXTRA_JSON_DIR = Path("data/doclaynet_extra/JSON")
OUT_DIR = Path("data/text_dataset")

TARGET_CLASSES = {
    "financial_reports", "scientific_articles", "laws_and_regulations",
    "government_tenders", "manuals", "patents",
}

# CSV Header definieren
CSV_FIELDS = ["doc_id", "original_filename", "page_no", "label", "text"]

# ============================================================
# Helpers
# ============================================================

def normalize(text: str) -> str:
    return " ".join(text.lower().split())

def get_split_mapping():
    """Liest train/val/test JSONs und baut eine Map: stem -> split"""
    mapping = {}
    for split in ["train", "val", "test"]:
        json_path = CORE_COCO_DIR / f"{split}.json"
        if not json_path.exists():
            print(f"Warn: {json_path} missing.")
            continue
            
        print(f"Reading {split}.json map...")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            
        for img in data["images"]:
            fname = img.get("file_name")
            if fname:
                stem = Path(fname).stem
                mapping[stem] = split
    return mapping

# ============================================================
# Worker Function (Muss top-level sein für Multiprocessing)
# ============================================================

def process_single_json(args):
    """
    Verarbeitet eine einzelne JSON Datei.
    args ist ein Tuple: (json_path, target_split)
    """
    json_path, target_split = args
    
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        
        meta = data.get("metadata", {})
        label = meta.get("doc_category")
        
        # Filter: Ist das Label relevant?
        if label not in TARGET_CLASSES:
            return None

        # Text extrahieren
        texts = [
            c.get("text", "")
            for c in data.get("cells", [])
            if c.get("text") and c.get("text").strip()
        ]
        
        if not texts:
            return None

        # Ergebnis zurückgeben (noch nicht schreiben!)
        return {
            "split": target_split,
            "row": {
                "doc_id": json_path.stem,
                "original_filename": meta.get("original_filename"),
                "page_no": meta.get("page_no"),
                "label": label,
                "text": normalize(" ".join(texts))
            }
        }
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

# ============================================================
# Main
# ============================================================

def main():
    print(f"Starting optimized processing on {multiprocessing.cpu_count()} cores...")

    # 1. Mapping laden
    file_to_split = get_split_mapping()
    print(f"Mapped {len(file_to_split)} documents.")

    # 2. Dateiliste vorbereiten (Vorfilterung!)
    # Wir erstellen eine Liste von Tuples (Pfad, Split), die wir verarbeiten wollen.
    tasks = []
    print("Scanning directory and preparing tasks...")
    
    # glob ist ein Generator, wir iterieren einmal durch
    for json_file in EXTRA_JSON_DIR.glob("*.json"):
        doc_id = json_file.stem
        if doc_id in file_to_split:
            # Nur Dateien zur Task-Liste hinzufügen, die wir auch brauchen
            tasks.append((json_file, file_to_split[doc_id]))
            
    print(f"Found {len(tasks)} relevant files to process.")

    # 3. CSV Writer öffnen
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    writers = {}
    
    for s in ["train", "val", "test"]:
        f = open(OUT_DIR / f"{s}.csv", "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        files[s] = f
        writers[s] = w

    # 4. Parallel verarbeiten
    count = 0
    
    # ProcessPoolExecutor startet Worker-Prozesse
    with ProcessPoolExecutor() as executor:
        # map führt process_single_json parallel für alle tasks aus.
        # chunksize=100 reduziert Overhead (nicht für jedes File ein neuer IPC Call)
        results = executor.map(process_single_json, tasks, chunksize=100)
        
        for result in results:
            if result:
                # Schreiben passiert im Hauptprozess (thread-safe bzgl CSV Files)
                split = result["split"]
                writers[split].writerow(result["row"])
                count += 1
                
                if count % 5000 == 0:
                    print(f"Processed {count} pages...")

    # 5. Cleanup
    for f in files.values():
        f.close()

    print(f"Done. Successfully processed {count} pages.")
    print(f"Output: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()