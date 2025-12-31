from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Konfiguration
# -----------------------------
DATA_YAML = r"C:/Users/rparz/Documents/GitHub/ML_Project/data/yolo_dataset/data.yaml"
MODEL_WEIGHTS = "yolov8m.pt"   # Pretrained weights
IMG_SIZE = 1024
EPOCHS = 50
BATCH = 8
DEVICE = "cpu"                  # 0 = erste GPU, "cpu" für CPU
PROJECT_DIR = "runs/layout"
RUN_NAME = "doclaynet_yolov8m"

# -----------------------------
# Sicherheitschecks
# -----------------------------
assert Path(DATA_YAML).exists(), f"data.yaml nicht gefunden: {DATA_YAML}"

# -----------------------------
# Modell laden
# -----------------------------
model = YOLO(MODEL_WEIGHTS)

# -----------------------------
# Training
# -----------------------------
train_results = model.train(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    pretrained=True,
    verbose=True
)

print("Training abgeschlossen.")
print(train_results)

# -----------------------------
# Validation (Val-Split)
# -----------------------------
val_results = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    device=DEVICE
)

print("\nValidation-Ergebnisse (val split):")
print(val_results)