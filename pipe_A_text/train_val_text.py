from pathlib import Path
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Paths & Setup
# ============================================================
path_df = "./data/text_dataset/"
RESULTS_DIR = Path("./pipe_A_text/results")
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "all_results_text_models.csv"

N_JOBS = max(1, os.cpu_count() - 1)

print(f"Using {N_JOBS} parallel jobs")

# ============================================================
# Load data
# ============================================================
train_df = pd.read_csv(f"{path_df}train.csv")
val_df   = pd.read_csv(f"{path_df}val.csv")

for df in (train_df, val_df):
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

# Leakage-Check
assert set(train_df["page_id"]).isdisjoint(set(val_df["page_id"])), \
    "Dokument-Leakage zwischen Train und Val!"

print("OK: dokumentbasierter Split ohne Leakage")

labels = sorted(train_df["label"].unique())

# ============================================================
# TF-IDF (EINMAL)
# ============================================================
print("Fitting TF-IDF...")

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=50_000
)

X_train = tfidf.fit_transform(train_df["text"])
X_val   = tfidf.transform(val_df["text"])

y_train = train_df["label"]
y_val   = val_df["label"]

print(f"TF-IDF shape train: {X_train.shape}")
print(f"TF-IDF shape val  : {X_val.shape}")

# ============================================================
# Model configs
# ============================================================
model_configs = [
    ("LogReg_C1", LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1
    )),
    ("LogReg_C3", LogisticRegression(
        C=3.0,
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1
    )),
    ("LinearSVC_C1", LinearSVC(
        C=1.0,
        class_weight="balanced"
    )),
    ("LinearSVC_C2", LinearSVC(
        C=2.0,
        class_weight="balanced"
    )),
    ("MultinomialNB", MultinomialNB(
        alpha=1.0
    )),
    ("RF_150", RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    )),
]

# ============================================================
# Run Models
# ============================================================
def run_model(name, clf):
    print(f"Running model: {name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    rows = []

    # ------------------
    # Metrics
    # ------------------
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        "precision_macro": precision_score(y_val, y_pred, average="macro"),
        "recall_macro": recall_score(y_val, y_pred, average="macro"),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "precision_weighted": precision_score(y_val, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_val, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_val, y_pred, average="weighted"),
    }

    for metric_name, value in metrics.items():
        rows.append({
            "model": name,
            "type": "metric",
            "row_label": metric_name,
            "col_label": "",
            "value": value
        })

    # ------------------
    # Confusion Matrix
    # ------------------
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "model": name,
                "type": "cm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm[i, j]
            })
            rows.append({
                "model": name,
                "type": "cm_norm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm_norm[i, j]
            })

    return rows


# ============================================================
# Run all models in parallel
# ============================================================
all_rows = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(run_model)(name, clf) for name, clf in model_configs
)

# Flatten
rows = [item for sublist in all_rows for item in sublist]

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved ALL results to: {OUTPUT_CSV}")


# ============================================================
# Plot absolute confusion matrices
# ============================================================
print("\nPlotting absolute confusion matrices...")

df = pd.read_csv(OUTPUT_CSV)
models = df["model"].unique()

sns.set_theme(style="white")

for model_name in models:
    cm_df = (
        df[
            (df["model"] == model_name) &
            (df["type"] == "cm")
        ]
        .pivot(
            index="row_label",
            columns="col_label",
            values="value"
        )
    )

    cm_df.index = cm_df.index.str.replace("true_", "")
    cm_df.columns = cm_df.columns.str.replace("pred_", "")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=".0f",
        cmap="Blues"
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (absolute) – {model_name}")
    plt.tight_layout()

    out_path = RESULTS_DIR / f"cm_absolute_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
