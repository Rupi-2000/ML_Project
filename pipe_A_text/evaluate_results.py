import pandas as pd
from pathlib import Path


INPUT_CSV = Path("./pipe_A_text/results/all_results_text_models.csv")
df = pd.read_csv(INPUT_CSV)

# Nur absolute Confusion Matrices
cm_df = df[df["type"] == "cm"].copy()

# Labelnamen extrahieren
cm_df["true"] = cm_df["row_label"].str.replace("true_", "")
cm_df["pred"] = cm_df["col_label"].str.replace("pred_", "")

rows = []

for model in cm_df["model"].unique():
    mdf = cm_df[cm_df["model"] == model]

    labels = sorted(mdf["true"].unique())

    # Confusion Matrix rekonstruieren
    cm = (
        mdf
        .pivot(index="true", columns="pred", values="value")
        .fillna(0)
        .astype(int)
    )

    total = cm.values.sum()

    for label in labels:
        TP = cm.loc[label, label]
        FP = cm[label].sum() - TP
        FN = cm.loc[label].sum() - TP
        TN = total - TP - FP - FN

        rows.append({
            "model": model,
            "class": label,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN
        })


metrics = []

for r in rows:
    TP, FP, FN, TN = r["TP"], r["FP"], r["FN"], r["TN"]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    metrics.append({
        **r,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

metrics_df = pd.DataFrame(metrics)

OUT_CSV = Path("./pipe_A_text/results/derived_metrics_from_cm.csv")
metrics_df.to_csv(OUT_CSV, index=False)

print(f"Saved derived metrics to: {OUT_CSV}")

# =========================
# Macro Recall & F1 je Modell
# =========================
model_scores = (
    metrics_df
    .groupby("model")[["recall", "f1"]]
    .mean()
    .reset_index()
    .rename(columns={
        "recall": "recall_macro",
        "f1": "f1_macro"
    })
)

print("\nMacro Scores je Modell:")
print(model_scores)

model_scores.to_csv(
    "./pipe_A_text/results/model_scores_macro.csv",
    index=False
)

best_model_row = model_scores.sort_values(
    "f1_macro", ascending=False
).iloc[0]

best_model = best_model_row["model"]

print(f"\nBestes Modell nach F1_macro: {best_model}")
print(best_model_row)

# =========================
# Klassenmetriken für bestes Modell
# =========================
best_model_class_metrics = (
    metrics_df[metrics_df["model"] == best_model]
    .loc[:, ["class", "recall", "f1", "TP", "FP", "FN", "TN"]]
    .sort_values("f1", ascending=False)
)

print(f"\nRecall & F1 je Klasse – Modell: {best_model}")
print(best_model_class_metrics)

best_model_class_metrics.to_csv(
    f"./pipe_A_text/results/{best_model}_class_metrics.csv",
    index=False
)
