import pandas as pd
import os
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
# Load train / val / test
# -----------------------------
X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")
X_test, y_test   = load_split("test")

print(f"Train samples: {len(X_train)}")
print(f"Val samples:   {len(X_val)}")
print(f"Test samples:  {len(X_test)}")

# Sanity check
assert set(X_train.columns) == set(X_val.columns), "Feature-Mismatch zwischen Train und Val!"

labels = sorted(y_val.unique())
sns.set_theme(style="white")


# ============================================================
# Model configs (Grid Search)
# ============================================================
model_configs = []

# 1. Logistic Regression
for c in [0.1, 1.0, 5.0, 10.0]:
    model_configs.append((
        f"LogReg_C{c}",
        LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced"
        )
    ))

# 2. Linear SVC
for c in [0.1, 0.5, 1.0, 2.0]:
    model_configs.append((
        f"LinearSVC_C{c}",
        LinearSVC(
            C=c,
            class_weight="balanced",
            max_iter=5000
        )
    ))

# 3. Random Forest
rf_grids = [
    {"n": 100, "d": 20},
    {"n": 200, "d": 30},
    {"n": 300, "d": None},
]

for g in rf_grids:
    name = f"RF_n{g['n']}_d{g['d']}"
    model_configs.append((
        name,
        RandomForestClassifier(
            n_estimators=g['n'],
            max_depth=g['d'],
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
    ))

print(f"Generated {len(model_configs)} model configurations for grid search.")


# ============================================================
# Cross-Validation on Train Set
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'recall_macro': 'recall_macro'
}

cv_results_list = []

print("\n>>> Starting Cross-Validation on Train Set...")
for name, clf in model_configs:
    print(f"  CV for: {name}")
    scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    cv_results_list.append({
        "model": name,
        "cv_accuracy_mean": np.mean(scores['test_accuracy']),
        "cv_accuracy_std":  np.std(scores['test_accuracy']),
        "cv_f1_macro_mean": np.mean(scores['test_f1_macro']),
        "cv_f1_macro_std":  np.std(scores['test_f1_macro']),
        "cv_recall_macro_mean": np.mean(scores['test_recall_macro']),
        "cv_recall_macro_std":  np.std(scores['test_recall_macro']),
    })

cv_df = pd.DataFrame(cv_results_list).sort_values("cv_f1_macro_mean", ascending=False)
print("\n=== CV Results (Train) ===")
print(cv_df[["model", "cv_accuracy_mean", "cv_f1_macro_mean"]].to_string(index=False))

cv_df.to_csv(RESULTS_DIR / "cv_results_layout.csv", index=False)
print(f"\nSaved CV results to: {RESULTS_DIR / 'cv_results_layout.csv'}")


# ============================================================
# Validation: Evaluate all models on Validation Set
# ============================================================
print("\n>>> Evaluating all models on Validation Set...")

val_results_list = []

for name, clf in model_configs:
    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    
    val_results_list.append({
        "model": name,
        "val_accuracy": accuracy_score(y_val, y_pred_val),
        "val_balanced_accuracy": balanced_accuracy_score(y_val, y_pred_val),
        "val_f1_macro": f1_score(y_val, y_pred_val, average="macro"),
        "val_recall_macro": recall_score(y_val, y_pred_val, average="macro"),
    })

val_df = pd.DataFrame(val_results_list).sort_values("val_f1_macro", ascending=False)
print("\n=== Validation Results ===")
print(val_df[["model", "val_accuracy", "val_f1_macro"]].to_string(index=False))

val_df.to_csv(RESULTS_DIR / "val_results_layout.csv", index=False)
print(f"\nSaved Validation results to: {RESULTS_DIR / 'val_results_layout.csv'}")


# ============================================================
# Select Best Model (based on Validation F1 Macro)
# ============================================================
best_row = val_df.iloc[0]
best_model_name = best_row["model"]
print(f"\nBest Model (Validation F1 Macro): {best_model_name} (F1: {best_row['val_f1_macro']:.4f})")

# Find the config for the best model
best_clf = next(clf for name, clf in model_configs if name == best_model_name)


# ============================================================
# Train Best Model on Train and Evaluate on Test Set
# ============================================================
print(f"\n>>> Retraining {best_model_name} on Train and Evaluating on Test...")
best_clf.fit(X_train, y_train)
y_pred_test = best_clf.predict(X_test)

# Detailed Metrics for Test
metrics_test = {
    "accuracy": accuracy_score(y_test, y_pred_test),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
    "precision_macro": precision_score(y_test, y_pred_test, average="macro"),
    "recall_macro": recall_score(y_test, y_pred_test, average="macro"),
    "f1_macro": f1_score(y_test, y_pred_test, average="macro"),
    "precision_weighted": precision_score(y_test, y_pred_test, average="weighted"),
    "recall_weighted": recall_score(y_test, y_pred_test, average="weighted"),
    "f1_weighted": f1_score(y_test, y_pred_test, average="weighted"),
}

print("\n=== Test Set Metrics ===")
for k, v in metrics_test.items():
    print(f"{k:<20}: {v:.4f}")

# Save Test Results
test_results_df = pd.DataFrame([{"model": best_model_name, **metrics_test}])
test_results_df.to_csv(RESULTS_DIR / "test_results_layout.csv", index=False)
print(f"\nSaved Test results to: {RESULTS_DIR / 'test_results_layout.csv'}")

# Save Classification Report
report = classification_report(y_test, y_pred_test, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(RESULTS_DIR / f"test_class_report_{best_model_name}.csv")


# ============================================================
# Confusion Matrix (Test)
# ============================================================
cm = confusion_matrix(y_test, y_pred_test, labels=labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot CM
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix (Test) - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"cm_test_absolute_{best_model_name}.png")
plt.close()

# Plot Normalized CM
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix (Normalized Test) - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"cm_test_norm_{best_model_name}.png")
plt.close()

print(f"\nSaved plots and reports to {RESULTS_DIR}")
