import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate


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
    max_features=50_000,
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

# ============================================================
# Model Configurations
# ============================================================

logreg_configs = [
    {"C": 0.1},
    {"C": 1.0},
    {"C": 5.0},
]

svm_configs = [
    {"C": 0.5},
    {"C": 1.0},
    {"C": 2.0},
]

rf_configs = [
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 300, "max_depth": None},
    {"n_estimators": 300, "max_depth": 30},
]

xgb_configs = [
    {"max_depth": 6, "learning_rate": 0.1},
    {"max_depth": 8, "learning_rate": 0.1},
    {"max_depth": 8, "learning_rate": 0.05},
]


# ============================================================
# Build Model List
# ============================================================

def cfg_to_name(prefix, cfg):
    parts = [f"{k}={v}" for k, v in sorted(cfg.items())]
    return prefix + "_" + "_".join(parts)

models = []

for cfg in logreg_configs:
    name = cfg_to_name("LogReg", cfg)
    models.append((
        name,
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            n_jobs=-1,
            **cfg
        )
    ))

for cfg in svm_configs:
    name = cfg_to_name("SVM", cfg)
    models.append((
        name,
        LinearSVC(
            class_weight="balanced",
            **cfg
        )
    ))

for cfg in rf_configs:
    name = cfg_to_name("RF", cfg)
    models.append((
        name,
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            **cfg
        )
    ))

for cfg in xgb_configs:
    name = cfg_to_name("XGB", cfg)
    models.append((
        name,
        XGBClassifier(
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **cfg
        )
    ))


# ============================================================
# Cross-Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scoring = {
    "acc": "accuracy",
    "f1": "f1_macro"
}

results = []

print("\n================ CV TRAINING ================\n")

for name, model in models:
    print(f"Running CV: {name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "model": name,
        "cv_acc_mean": scores["test_acc"].mean(),
        "cv_acc_std":  scores["test_acc"].std(),
        "cv_f1_macro": scores["test_f1"].mean()
    })


results_df = (
    pd.DataFrame(results)
    .sort_values("cv_f1_macro", ascending=False)
)

print("\n=== CV RESULTS (TRAIN) ===")
print(results_df.to_string(index=False))


# ============================================================
# Final Evaluation on Validation Set
# ============================================================

print("\n================ FINAL EVALUATION ================\n")

top_models = results_df.head(3)["model"].tolist()
model_dict = dict(models)

for name in top_models:
    model = model_dict[name]

    print(f"\nFinal model: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))


# ============================================================
# Save Summary
# ============================================================

results_df.to_csv("hybrid_cv_results.csv", index=False)
print("\nSaved results to hybrid_cv_results.csv")