import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# Paths & Setup
# ============================================================
RESULTS_DIR = Path("./pipe_A_text/results")
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "all_results_text_models.csv"

def save_cv_results(cv_results_df):
    """
    Saves CV summary results to a CSV file.
    """
    cv_results_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    print(f"\nSaved CV results to: {RESULTS_DIR / 'cv_results.csv'}")

def save_test_results_compatible(best_model_name, y_true, y_pred, labels):
    """
    Saves test results in a format compatible with `evaluate_results.py`.
    It mimics the structure of `all_results_text_models.csv` which contains
    metrics and confusion matrix entries.
    """
    rows = []

    # 1. Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    for metric_name, value in metrics.items():
        rows.append({
            "model": best_model_name,
            "type": "metric",
            "row_label": metric_name,
            "col_label": "",
            "value": value
        })

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "model": best_model_name,
                "type": "cm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm[i, j]
            })
            rows.append({
                "model": best_model_name,
                "type": "cm_norm",
                "row_label": f"true_{true_label}",
                "col_label": f"pred_{pred_label}",
                "value": cm_norm[i, j]
            })

    results_df = pd.DataFrame(rows)
    
    # Save to the main results CSV (overwriting or appending)
    # Since this script seems to run a full sweep and pick ONE best model,
    # we might want to just save this single result, or append if we ran multiple.
    # For now, we overwrite to match the expected single-file output for the viewer script.
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved compatible Test/Best-Model results to: {OUTPUT_CSV}")
