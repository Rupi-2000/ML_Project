from ultralytics import YOLO
from pathlib import Path
import torch

# -----------------------------
# GPU Check
# -----------------------------
assert torch.cuda.is_available(), "CUDA nicht verfügbar – NVIDIA Treiber prüfen!"
print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Konfiguration
# -----------------------------
DATA_YAML = r"./data/yolo_dataset/data.yaml"
MODEL_WEIGHTS = "yolov8n.pt"   # Pretrained weights
IMG_SIZE = 640
EPOCHS = 20
BATCH = 16
DEVICE = 0                  # 0 = erste GPU, "cpu" für CPU
PROJECT_DIR = "runs/layout"
RUN_NAME = "doclaynet_yolov8n"

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