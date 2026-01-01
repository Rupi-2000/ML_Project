import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Paths
# -----------------------------
YOLO_BASE_DIR = Path("./data/yolo_dataset")
TEXT_DATASET_DIR = Path("./data/text_dataset")

# -----------------------------
# Load split (features + labels)
# -----------------------------
def load_split(split: str):
    # Layout features
    features = pd.read_csv(
        YOLO_BASE_DIR / split / "layout_features.csv"
    )

    # Text dataset labels (Pipeline A Splits)
    text_df = pd.read_csv(
        TEXT_DATASET_DIR / f"{split}.csv"
    )

    # page_id aus original_filename
    text_df["page_id"] = text_df["original_filename"].apply(
        lambda x: Path(x).stem
    )

    labels = text_df[["page_id", "label"]].rename(
        columns={"label": "doc_class"}
    )

    # Merge
    df = features.merge(labels, on="page_id", how="inner")

    print(f"[{split}] Samples after merge: {len(df)}")

    X = df.drop(columns=["page_id", "doc_class"])
    y = df["doc_class"]

    return X, y

# -----------------------------
# Load train / val
# -----------------------------
X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")

# Feature consistency check
assert list(X_train.columns) == list(X_val.columns), \
    "Feature mismatch between train and val!"

# -----------------------------
# Models to compare
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}

# -----------------------------
# Evaluation loop
# -----------------------------
results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print(classification_report(y_val, y_pred))

    results.append({
        "model": name,
        "accuracy": acc
    })

# -----------------------------
# Summary table
# -----------------------------
summary_df = pd.DataFrame(results).sort_values(
    by="accuracy", ascending=False
)

print("\n=== Model Comparison Summary ===")
print(summary_df.to_string(index=False))
