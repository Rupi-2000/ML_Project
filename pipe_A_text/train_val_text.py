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
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from results_saver import save_test_results_compatible

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
test_df  = pd.read_csv(f"{path_df}test.csv")

# Combine Train + Val for Dev
dev_df = pd.concat([train_df, val_df], ignore_index=True)

for df in (dev_df, test_df):
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

print(f"Train samples: {len(train_df)}")
print(f"Val samples:   {len(val_df)}")
print(f"Dev samples:   {len(dev_df)} (Train + Val)")
print(f"Test samples:  {len(test_df)}")

labels = sorted(dev_df["label"].unique())

# ============================================================
# TF-IDF (EINMAL)
# ============================================================
print("Fitting TF-IDF on Dev set...")

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=50_000
)

X_dev  = tfidf.fit_transform(dev_df["text"])
X_test = tfidf.transform(test_df["text"])

y_dev  = dev_df["label"]
y_test = test_df["label"]

print(f"TF-IDF shape dev : {X_dev.shape}")
print(f"TF-IDF shape test: {X_test.shape}")


# ============================================================
# Model configs (Small Grid Search)
# ============================================================
model_configs = []

# 1. Logistic Regression
for c in [0.1, 1.0, 5.0, 10.0]:
    model_configs.append((
        f"LogReg_C{c}", 
        LogisticRegression(
            C=c, 
            solver="saga", 
            max_iter=4000, 
            class_weight="balanced",
        )
    ))

# 2. Linear SVC
for c in [0.1, 0.5, 1.0, 2.0]:
    model_configs.append((
        f"LinearSVC_C{c}", 
        LinearSVC(
            C=c, 
            class_weight="balanced"
        )
    ))

# 3. MultinomialNB
for alpha in [0.1, 0.5, 1.0]:
    model_configs.append((
        f"MultinomialNB_a{alpha}", 
        MultinomialNB(alpha=alpha)
    ))

# 4. Random Forest
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
# Run Models
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(name, clf):
    """Runs CV on Dev set and returns summary metrics."""
    print(f"Running CV for: {name}")
    
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'recall_macro': 'recall_macro'
    }
    
    scores = cross_validate(
        clf, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1
    )
    
    return {
        "model": name,
        "cv_accuracy_mean": np.mean(scores['test_accuracy']),
        "cv_accuracy_std":  np.std(scores['test_accuracy']),
        "cv_f1_macro_mean": np.mean(scores['test_f1_macro']),
        "cv_f1_macro_std":  np.std(scores['test_f1_macro']),
        "cv_recall_macro_mean": np.mean(scores['test_recall_macro']),
        "cv_recall_macro_std":  np.std(scores['test_recall_macro']),
    }

# 1. Run CV for all models
print("\n>>> Starting Cross-Validation...")

cv_results_list = []

for name, clf in model_configs:

    if isinstance(clf, LinearSVC):

        params = clf.get_params()
        if 'n_jobs' in params:
            del params['n_jobs']
            clf.set_params(**params)

    result = run_cv(name, clf)
    cv_results_list.append(result)

cv_df = pd.DataFrame(cv_results_list).sort_values("cv_f1_macro_mean", ascending=False)
print("\n=== CV Results ===")
print(cv_df[["model", "cv_accuracy_mean", "cv_f1_macro_mean"]].to_string(index=False))

cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)


# 2. Select Best Model
best_row = cv_df.iloc[0]
best_model_name = best_row["model"]
print(f"\nBest Model: {best_model_name} (F1 Macro: {best_row['cv_f1_macro_mean']:.4f})")

# Find the config for the best model
best_clf_config = next(clf for name, clf in model_configs if name == best_model_name)


# 3. Retrain on Full Dev & Evaluate on Test
print(f"\n>>> Retraining {best_model_name} on Data (Train+Val) and Evaluating on Test...")
best_clf_config.fit(X_dev, y_dev)
y_pred_test = best_clf_config.predict(X_test)


# 4. Detailed Metrics for Test
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

# Save detailed Classification Report
report = classification_report(y_test, y_pred_test, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(RESULTS_DIR / f"test_class_report_{best_model_name}.csv")

# 5. Confusion Matrix (Test)
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

# 6. Save results for evaluate_results.py
save_test_results_compatible(best_model_name, y_test, y_pred_test, labels)
