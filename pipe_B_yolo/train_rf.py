import pandas as pd
import os
from pathlib import Path
from sklearn.calibration import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# -----------------------------
# Base path
# -----------------------------
BASE_DIR = Path("./data/yolo_dataset")
RESULTS_DIR = Path("./pipe_B_yolo/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helper: load split
# -----------------------------
def load_split(split: str):
    split_dir = BASE_DIR / split

    features = pd.read_csv(split_dir / "layout_features.csv")
    labels   = pd.read_csv(split_dir / "labels.csv")

    df = features.merge(labels, on="page_id")

    X = df.drop(columns=["page_id", "doc_class"])
    y = df["doc_class"]

    return X, y

# -----------------------------
# Load train / val
# -----------------------------
X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")

# Sanity check
assert set(X_train.columns) == set(X_val.columns), "Feature-Mismatch zwischen Train und Val!"

labels = sorted(y_val.unique())
sns.set_theme(style="white")

# ============================================================
# Model configs
# ============================================================
model_configs = [
    (
        "RF_200",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    ),
    (
        "LogReg",
        LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=1
        )
    ),
    (
        "LinearSVC",
        LinearSVC(
            C=1.0,
            class_weight="balanced"
        )
    ),
]

# ============================================================
# Model runner
# ============================================================
def run_model(name, clf):
    print(f"\n=== Running {name} ===")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=False)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    return {
        "model": name,
        "report": report,
        "confusion_matrix": cm
    }


# ============================================================
# Run models 
# ============================================================
results = [run_model(name, clf) for name, clf in model_configs]

# ============================================================
# Print results
# ============================================================
for res in results:
    model_name = res["model"]
    cm = res["confusion_matrix"]
    
    print(f"\n--- Classification Report: {model_name} ---")
    print(res["report"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (absolute) – {model_name}")
    plt.tight_layout()

    out_path = RESULTS_DIR / f"cm_absolute_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
