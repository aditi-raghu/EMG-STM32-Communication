
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("results/binary_windows.csv")
MODEL_PATH = Path("results/svm_clench_vs_nonclench.joblib")

FEATURES = [
    "MAV", "RMS", "WL", "ZC",
    "SSC", "MNF", "MDF", "LF"
]

df = pd.read_csv(DATA_PATH)

# Keep the original labels for separate evaluation.
df = df[df["label"].isin([0, 1, 2])].copy()
df["original_label"] = df["label"].astype(int)

# Binary classification:
# 0 = relaxed or open hand
# 1 = fist clench
df["binary_label"] = (
    df["original_label"] == 1
).astype(int)

print("Binary class distribution:")
print(df["binary_label"].value_counts().sort_index())

def create_model():
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )
    )

all_true = []
all_pred = []
all_original = []
fold_scores = []

subjects = sorted(df["subject"].unique())

print("\nLeave-One-Subject-Out evaluation")

for subject in subjects:
    train = df[df["subject"] != subject]
    test = df[df["subject"] == subject]

    model = create_model()
    model.fit(
        train[FEATURES],
        train["binary_label"]
    )

    predictions = model.predict(test[FEATURES])

    score = balanced_accuracy_score(
        test["binary_label"],
        predictions
    )

    fold_scores.append(score)

    all_true.extend(test["binary_label"])
    all_pred.extend(predictions)
    all_original.extend(test["original_label"])

    print(
        f"Subject {subject}: "
        f"balanced accuracy = {score:.3f}"
    )

print(
    "\nMean LOSO balanced accuracy:",
    round(np.mean(fold_scores), 3)
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        all_true, all_pred, labels=[0, 1]
    )
)

print("\nClassification report:")
print(
    classification_report(
        all_true,
        all_pred,
        labels=[0, 1],
        target_names=["Non-clench", "Clench"],
        zero_division=0,
    )
)

# Evaluate false clench predictions separately
all_pred = np.asarray(all_pred)
all_original = np.asarray(all_original)

for label, name in [
    (0, "Relaxed"),
    (2, "Open hand")
]:
    mask = all_original == label

    if mask.sum() > 0:
        false_clench_rate = np.mean(
            all_pred[mask] == 1
        )

        print(
            f"\n{name} false-clench rate: "
            f"{false_clench_rate:.3f} "
            f"({mask.sum()} windows)"
        )

# Train final model using all subjects.
final_model = create_model()
final_model.fit(
    df[FEATURES],
    df["binary_label"]
)

MODEL_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

joblib.dump(
    {
        "model": final_model,
        "features": FEATURES,
        "labels": {
            0: "Non-clench",
            1: "Clench"
        },
        "sampling_rate_hz": 400,
        "window_seconds": 0.25,
        "channel": "Channel 4",
    },
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")
