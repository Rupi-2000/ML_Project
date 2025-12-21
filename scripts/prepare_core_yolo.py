"""
Prepare DocLayNet CORE for YOLOv8 (High Performance Copy).
Safe for Windows, Linux, and macOS.
"""

import json
import shutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # pip install tqdm

# ============================================================
# Configuration
# ============================================================

CORE_DIR = Path("data/doclaynet_core")
COCO_DIR = CORE_DIR / "COCO"
IMG_SRC_DIR = CORE_DIR / "PNG"

YOLO_OUT_DIR = Path("data/yolo_dataset")

NUM_THREADS = 8

SPLITS = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
}

CLASSES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
]

# ============================================================
# Worker Function (läuft parallel)
# ============================================================

def process_single_image(args):
    """
    Kopiert Bild und erstellt Label.
    args ist ein Tuple, damit wir es einfach an den Executor geben können.
    """
    img, annotations, src_dir, img_out_dir, lbl_out_dir = args

    img_name = img["file_name"]
    img_w = img["width"]
    img_h = img["height"]
    
    # Pfade bauen
    src_path = src_dir / img_name
    dst_path = img_out_dir / img_name
    lbl_path = lbl_out_dir / f"{Path(img_name).stem}.txt"

    # 1. Bild kopieren
    # Wenn Bild im Source fehlt -> Skip
    if not src_path.exists():
        return # Silent skip or logging

    # Optimiertes Kopieren (shutil.copy ist schneller als copy2)
    shutil.copy(src_path, dst_path)

    # 2. Labels berechnen
    label_lines = []
    if annotations:
        for ann in annotations:
            cls_id = ann["category_id"]
            x, y, w, h = ann["bbox"]

            # Mathe: COCO (x,y,w,h) -> YOLO (centerX, centerY, w, h) normalized
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Begrenzung auf [0, 1] (falls Annotations leicht außerhalb liegen)
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # 3. Label schreiben
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))


# ============================================================
# Main Logic
# ============================================================

def load_coco_mapping(json_path):
    print(f"Lade Metadaten: {json_path} ...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Mapping für schnellen Zugriff
    image_id_to_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        
        # ID Korrektur (1-based -> 0-based)
        if ann["category_id"] != -1:
            ann["category_id"] -= 1
            image_id_to_anns[img_id].append(ann)

    return data["images"], image_id_to_anns

def prepare_split(split_name, coco_file):
    print(f"\n--- Starte Split: {split_name} ---")
    
    # Pfade vorbereiten
    img_out_dir = YOLO_OUT_DIR / split_name / "images"
    lbl_out_dir = YOLO_OUT_DIR / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Daten laden
    images, image_id_to_anns = load_coco_mapping(COCO_DIR / coco_file)

    # Task-Liste erstellen
    tasks = []
    for img in images:
        anns = image_id_to_anns.get(img["id"], [])
        # Wir packen alle nötigen Infos in ein Tuple
        tasks.append((img, anns, IMG_SRC_DIR, img_out_dir, lbl_out_dir))

    # Parallel abarbeiten
    print(f"Kopiere und konvertiere {len(tasks)} Bilder ({NUM_THREADS} Threads)...")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # map führt die Funktion parallel für alle tasks aus
        # tqdm zeigt dabei den Fortschritt an
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"))

def main():
    if YOLO_OUT_DIR.exists():
        print(f"WARNUNG: Output Ordner {YOLO_OUT_DIR} existiert bereits.")
    
    for split, coco_file in SPLITS.items():
        prepare_split(split, coco_file)
    
    print("\n=== FERTIG ===")
    print(f"Dataset bereit unter: {YOLO_OUT_DIR.resolve()}")
    
    # yaml Datei Hinweis
    yaml_content = f"""
path: {YOLO_OUT_DIR.resolve()} 
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {CLASSES}
    """
    print("\nKopiere dies in deine data.yaml:")
    print(yaml_content)

if __name__ == "__main__":
    main()