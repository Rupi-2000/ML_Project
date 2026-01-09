from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_YAML = "./data/yolo_dataset/data_docclass.yaml"
MODEL_NAME = "yolov8n.pt"
PROJECT = "runs_docclass"
EXPERIMENT = "yolov8n_docclass_fullpage"

IMG_SIZE = 640
EPOCHS = 20
BATCH = 16
SEED = 42

# -----------------------------
# Train
# -----------------------------
model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    project=PROJECT,
    name=EXPERIMENT,
    seed=SEED,
    device=0,              # set to "cpu" if no GPU
    workers=8,
    verbose=True,
)

# -----------------------------
# Validation (best model)
# -----------------------------
best_model_path = Path(PROJECT) / EXPERIMENT / "weights/best.pt"
best_model = YOLO(best_model_path)

val_results = best_model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    split="val",
)

print("Validation results:")
print(val_results)
