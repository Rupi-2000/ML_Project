import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# CSV laden
# --------------------------------------------------
CSV_PATH = "./result_gesamt.csv"   # <- Pfad anpassen
df = pd.read_csv(CSV_PATH)

# Pipeline-Gruppe extrahieren (A/B/C/D)
df["pipeline"] = df["model"].str.extract(r"^([A-D])")

metrics = ["f1_macro", "recall", "accuracy"]

# ==================================================
# Plot 1: Modellvergleich (sortiert nach F1_macro)
# ==================================================
df_sorted = df.sort_values("f1_macro", ascending=False)

x = np.arange(len(df_sorted))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 6))

for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,
        df_sorted[metric],
        width,
        label=metric
    )

ax.set_xticks(x + width)
ax.set_xticklabels(df_sorted["model"], rotation=45, ha="right")
ax.set_ylim(0.4, 1.01)
ax.set_ylabel("Score")
ax.set_title("Modellvergleich: F1_macro, Recall, Accuracy")

# Grid
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

ax.legend()
fig.tight_layout()
plt.show()

# ==================================================
# Plot 2: Pipeline-Vergleich (Mittelwerte)
# ==================================================
group_means = (
    df
    .groupby("pipeline")[metrics]
    .mean()
    .sort_index()
)

x = np.arange(len(group_means))

fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,
        group_means[metric],
        width,
        label=metric
    )

ax.set_xticks(x + width)
ax.set_xticklabels(group_means.index)
ax.set_ylim(0.4, 1.01)
ax.set_ylabel("Score")
ax.set_title("Pipeline-Vergleich (Mittelwerte)")

# Grid
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

ax.legend()
fig.tight_layout()
plt.show()

# ==================================================
# Plot 3: F1_macro-Verteilung pro Pipeline (Boxplot)
# ==================================================
fig, ax = plt.subplots(figsize=(8, 5))

df.boxplot(
    column="f1_macro",
    by="pipeline",
    grid=False,
    ax=ax
)

ax.set_title("F1_macro-Verteilung pro Pipeline")
ax.set_ylabel("F1_macro")
ax.set_ylim(0.4, 1.01)

# Grid
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

plt.suptitle("")
fig.tight_layout()
plt.show()