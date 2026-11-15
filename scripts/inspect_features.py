
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("results/binary_windows.csv")
OUT = Path("results/feature_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

features = [
    "MAV", "RMS", "WL", "ZC",
    "SSC", "MNF", "MDF", "LF"
]

labels = {
    0: "Relaxed",
    1: "Clench",
    2: "Open"
}

# Plot feature distributions
fig, axes = plt.subplots(
    2, 4, figsize=(16, 8)
)

for ax, feature in zip(axes.flat, features):
    data = [
        df.loc[df.label == label, feature]
        for label in labels
    ]

    ax.boxplot(
        data,
        tick_labels=list(labels.values()),
        showfliers=False
    )

    ax.set_title(feature)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    OUT / "feature_distributions.png",
    dpi=200
)
plt.close()

# Calculate feature medians by original label
summary = df.groupby("label")[features].median()
summary.index = summary.index.map(labels)

print("\nMedian feature values:")
print(summary.round(4).to_string())

summary.to_csv(
    OUT / "feature_summary.csv"
)

print("\nSaved feature analysis to:", OUT)
