import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
YOLO_WEIGHTS = "./runs/layout/doclaynet_yolov8n9/weights/best.pt"
IMAGE_DIR = Path("./data/yolo_dataset/train/images")
OUTPUT_CSV = "layout_features.csv"

N_CLASSES = 11

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO(YOLO_WEIGHTS)

rows = []

for img_path in IMAGE_DIR.glob("*.png"):
    result = model.predict(
        source=img_path,
        conf=0.5,
        verbose=False
    )[0]

    # init features
    counts = np.zeros(N_CLASSES)
    areas = np.zeros(N_CLASSES)
    y_positions = []

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            x, y, w, h = box.xywhn[0].cpu().numpy()
            area = w * h

            counts[cls] += 1
            areas[cls] += area
            y_positions.append(y)

    total_area = areas.sum() + 1e-6
    area_ratios = areas / total_area

    if y_positions:
        mean_y = np.mean(y_positions)
        std_y = np.std(y_positions)
        header_ratio = np.mean(np.array(y_positions) < 0.2)
        footer_ratio = np.mean(np.array(y_positions) > 0.8)
    else:
        mean_y = std_y = header_ratio = footer_ratio = 0.0

    row = {
        "page_id": img_path.stem,
        **{f"count_{i}": counts[i] for i in range(N_CLASSES)},
        **{f"area_{i}": area_ratios[i] for i in range(N_CLASSES)},
        "mean_y": mean_y,
        "std_y": std_y,
        "header_ratio": header_ratio,
        "footer_ratio": footer_ratio,
    }

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved features to {OUTPUT_CSV}")
