from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns


path_df = "./data/text_dataset/"

train_df = pd.read_csv(f"{path_df}train.csv")
val_df   = pd.read_csv(f"{path_df}val.csv")

for df in (train_df, val_df):
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

# Leakage-Check
assert set(train_df["page_id"]).isdisjoint(set(val_df["page_id"])), \
    "Dokument-Leakage zwischen Train und Val!"

print("OK: dokumentbasierter Split ohne Leakage")

model_configs = [
    ("LogReg_C1", LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    )),
    ("LogReg_C3", LogisticRegression(
        C=3.0,
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
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
    ("RF_200", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )),
]

results = []

labels = sorted(train_df["label"].unique())

RESULTS_DIR = Path("./pipe_A_text/results")
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "all_results_text_models.csv"
rows = []

for name, clf in model_configs:
    print(f"Running model: {name}")
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=30_000
        )),
        ("clf", clf)
    ])

    # Train & predict
    pipeline.fit(train_df["text"], train_df["label"])
    y_true = val_df["label"]
    y_pred = pipeline.predict(val_df["text"])

    # =====================
    # Metrics
    # =====================
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    for metric_name, value in metrics.items():
        rows.append({
            "model": name,
            "type": "metric",
            "row_label": metric_name,
            "col_label": "",
            "value": value
        })

    # =====================
    # Confusion Matrix (absolute)
    # =====================
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "model": name,
                "type": "cm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm[i, j]
            })

    # =====================
    # Confusion Matrix (normalized)
    # =====================
    cm_norm = confusion_matrix(
        y_true, y_pred, labels=labels, normalize="true"
    )

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "model": name,
                "type": "cm_norm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm_norm[i, j]
            })
            
results_df = pd.DataFrame(rows)

results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved ALL results to: {OUTPUT_CSV}")

print("\nPlotting absolute confusion matrices...")

df = pd.read_csv(OUTPUT_CSV)

models = df["model"].unique()

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

    # Labels aufräumen
    cm_df.index = cm_df.index.str.replace("true_", "")
    cm_df.columns = cm_df.columns.str.replace("pred_", "")

    # Plot
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

    # Speichern
    out_path = RESULTS_DIR / f"cm_absolute_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
