import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Base path
# -----------------------------
BASE_DIR = Path("./data/yolo_dataset")

# -----------------------------
# Helper: load split
# -----------------------------
def load_split(split: str):
    split_dir = BASE_DIR / split

    features = pd.read_csv(split_dir / "layout_features.csv")
    labels   = pd.read_csv(split_dir / "labels.csv")

    df = features.merge(labels, on="page_id")
    print(df.head(10))

    X = df.drop(columns=["page_id", "doc_class"])
    y = df["doc_class"]

    return X, y

# -----------------------------
# Load train / val
# -----------------------------
X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")

# -----------------------------
# Sanity check
# -----------------------------
assert set(X_train.columns) == set(X_val.columns), "Feature-Mismatch zwischen Train und Val!"

# -----------------------------
# Train RF
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# Evaluate on Val
# -----------------------------
y_pred = rf.predict(X_val)

print("=== Validation Results ===")
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
