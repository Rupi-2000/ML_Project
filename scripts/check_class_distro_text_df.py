import csv
from pathlib import Path
from collections import Counter

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data/text_dataset")
SPLITS = ["train", "val", "test"]


# ============================================================
# Helpers
# ============================================================

def load_labels(csv_path):
    labels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"])
    return labels


def print_distribution(split_name, labels):
    total = len(labels)
    counter = Counter(labels)

    print(f"\n=== {split_name.upper()} ===")
    print(f"Total samples: {total}")

    for label, count in counter.most_common():
        percent = 100 * count / total
        print(f"{label:25s} {count:6d}  ({percent:5.2f}%)")

    return counter, total


# ============================================================
# Main
# ============================================================

def main():
    distributions = {}

    for split in SPLITS:
        csv_path = DATASET_DIR / f"{split}.csv"
        labels = load_labels(csv_path)
        dist, total = print_distribution(split, labels)
        distributions[split] = (dist, total)

    # --------------------------------------------------------
    # Optional: sanity check (difference in percentages)
    # --------------------------------------------------------

    print("\n=== MAX RELATIVE DIFFERENCE (train vs val/test) ===")

    train_dist, train_total = distributions["train"]

    for split in ["val", "test"]:
        dist, total = distributions[split]
        print(f"\nTrain vs {split}:")

        for label in train_dist:
            p_train = train_dist[label] / train_total
            p_other = dist[label] / total if label in dist else 0
            diff = abs(p_train - p_other) * 100

            print(f"{label:25s} Δ = {diff:5.2f}%")

if __name__ == "__main__":
    main()
