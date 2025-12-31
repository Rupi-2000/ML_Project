from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

path_df = "./data/text_dataset/"

train_df = pd.read_csv(f"{path_df}train.csv")
val_df   = pd.read_csv(f"{path_df}val.csv")

for df in (train_df, val_df):
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

# Leakage-Check
assert set(train_df["doc_id"]).isdisjoint(set(val_df["doc_id"])), \
    "Dokument-Leakage zwischen Train und Val!"

print("OK: dokumentbasierter Split ohne Leakage")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=50_000
    )),
    ("clf", LinearSVC(C=1.0))
])

# Training
pipeline.fit(train_df["text"], train_df["label"])

# Validation
val_pred = pipeline.predict(val_df["text"])
print(classification_report(val_df["label"], val_pred, digits=3))
