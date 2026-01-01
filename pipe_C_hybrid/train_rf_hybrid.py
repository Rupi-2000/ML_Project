import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# -----------------------------
# Paths
# -----------------------------
YOLO_BASE_DIR = Path("./data/yolo_dataset")
TEXT_DATASET_DIR = Path("./data/text_dataset")

# -----------------------------
# Load split (features + text + labels)
# -----------------------------
def load_split(split: str):
    # Layout features
    layout_df = pd.read_csv(
        YOLO_BASE_DIR / split / "layout_features.csv"
    )

    # Text dataset
    text_df = pd.read_csv(
        TEXT_DATASET_DIR / f"{split}.csv"
    )

    # Merge
    df = layout_df.merge(text_df, on="page_id", how="inner")

    print(f"[{split}] merged pages: {len(df)}")

    # Layout features (numerisch)
    X_layout = df.drop(columns=["page_id", "label" , "page_no", "text","original_filename"])

    # Text
    texts = df["text"].fillna("")

    y = df["label"]

    return X_layout, texts, y

# -----------------------------
# Load train / val
# -----------------------------
X_layout_train, texts_train, y_train = load_split("train")
X_layout_val,   texts_val,   y_val   = load_split("val")

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=30_000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_text_train = tfidf.fit_transform(texts_train)
X_text_val   = tfidf.transform(texts_val)

# -----------------------------
# Combine layout + text
# -----------------------------
X_layout_train_sparse = csr_matrix(X_layout_train.values)
X_layout_val_sparse   = csr_matrix(X_layout_val.values)

X_train = hstack([X_layout_train_sparse, X_text_train])
X_val   = hstack([X_layout_val_sparse,   X_text_val])

# -----------------------------
# Models
# -----------------------------
models = {
    "LogReg (Text + Layout)": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "RF (Text + Layout)": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}

# -----------------------------
# Evaluation
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
# Summary
# -----------------------------
summary_df = pd.DataFrame(results).sort_values(
    by="accuracy", ascending=False
)

print("\n=== Hybrid Model Summary ===")
print(summary_df.to_string(index=False))
