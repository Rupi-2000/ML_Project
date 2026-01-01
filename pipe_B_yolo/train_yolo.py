from ultralytics import YOLO
from pathlib import Path
import torch


def main():
    # -----------------------------
    # Device Handling
    # -----------------------------
    print("Torch:", torch.__version__)
    print("CUDA verfügbar:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNUNG: CUDA nicht verfügbar – Training läuft auf CPU")

    # -----------------------------
    # Konfiguration
    # -----------------------------
    DATA_YAML = r"./data/yolo_dataset/data.yaml"
    MODEL_WEIGHTS = "yolov8n.pt"
    IMG_SIZE = 640
    EPOCHS = 20
    BATCH = 16
    PROJECT_DIR = "runs/layout"
    RUN_NAME = "doclaynet_yolov8n"

    # -----------------------------
    # Checks
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
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        pretrained=True,
        workers=0
    )

    print("Training abgeschlossen.")
    print(train_results)

    # -----------------------------
    # Validation
    # -----------------------------
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=device
    )

    print("\nValidation-Ergebnisse (val split):")
    print(val_results)


if __name__ == "__main__":
    main()