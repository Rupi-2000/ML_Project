import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
YOLO_WEIGHTS = "./runs/layout/doclaynet_yolov8n9/weights/best.pt"
BASE_DATASET_DIR = Path("./data/yolo_dataset")
SPLITS = ["train", "val", "test"]

N_CLASSES = 11
CONF_THRES = 0.5

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO(YOLO_WEIGHTS)

def extract_split(split: str):
    split_dir = BASE_DATASET_DIR / split
    image_dir = split_dir / "images"
    output_csv = split_dir / "layout_features.csv"

    image_paths = list(image_dir.glob("*.png"))
    rows = []

    print(f"\n[{split}] Processing {len(image_paths)} images")

    for img_path in tqdm(image_paths, desc=f"{split}", unit="img"):
        result = model.predict(
            source=img_path,
            conf=CONF_THRES,
            verbose=False
        )[0]

        counts = np.zeros(N_CLASSES, dtype=np.float32)
        areas = np.zeros(N_CLASSES, dtype=np.float32)
        y_positions = []

        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                _, y, w, h = box.xywhn[0].cpu().numpy()
                area = w * h

                counts[cls] += 1
                areas[cls] += area
                y_positions.append(y)

        total_area = areas.sum() + 1e-6
        area_ratios = areas / total_area

        if y_positions:
            y_arr = np.array(y_positions)
            mean_y = float(np.mean(y_arr))
            std_y = float(np.std(y_arr))
            header_ratio = float(np.mean(y_arr < 0.2))
            footer_ratio = float(np.mean(y_arr > 0.8))
        else:
            mean_y = std_y = header_ratio = footer_ratio = 0.0

        row = {
            "page_id": img_path.stem,
            **{f"count_{i}": float(counts[i]) for i in range(N_CLASSES)},
            **{f"area_{i}": float(area_ratios[i]) for i in range(N_CLASSES)},
            "mean_y": mean_y,
            "std_y": std_y,
            "header_ratio": header_ratio,
            "footer_ratio": footer_ratio,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[{split}] Saved {len(df)} samples → {output_csv}")

# -----------------------------
# Run for all splits
# -----------------------------
if __name__ == "__main__":
    for split in SPLITS:
        extract_split(split)
