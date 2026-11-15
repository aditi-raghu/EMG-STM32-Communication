
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
MODEL_PATH = Path("results/emg_clench_final.joblib")

FEATURES = [
    "MAV", "RMS", "WL", "ZC",
    "SSC", "MNF", "MDF", "LF"
]

# Load extracted EMG features
df = pd.read_csv(DATA_PATH)

# Strict binary classification:
# 0 = Relaxed, 1 = Clench
# Label 2 is completely excluded.
df = df[df["label"].isin([0, 1])].copy()

print("Class distribution:")
print(df["label"].value_counts().sort_index())

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

# Leave-One-Subject-Out validation
subjects = sorted(df["subject"].unique())

all_true = []
all_pred = []
scores = []

for subject in subjects:
    train = df[df["subject"] != subject]
    test = df[df["subject"] == subject]

    model = create_model()
    model.fit(train[FEATURES], train["label"])

    pred = model.predict(test[FEATURES])

    score = balanced_accuracy_score(
        test["label"], pred
    )

    scores.append(score)
    all_true.extend(test["label"])
    all_pred.extend(pred)

    print(
        f"Subject {subject}: "
        f"balanced accuracy = {score:.3f}"
    )

print(
    "\nMean LOSO balanced accuracy:",
    round(np.mean(scores), 3)
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
        target_names=["Relaxed", "Clench"],
        zero_division=0,
    )
)

# Train final model on all available
# relaxed and clench recordings.
final_model = create_model()

final_model.fit(
    df[FEATURES],
    df["label"]
)

MODEL_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

joblib.dump(
    {
        "model": final_model,
        "features": FEATURES,
        "sampling_rate_hz": 400,
        "window_seconds": 0.25,
        "window_step_seconds": 0.125,
        "channel": "Channel 4",
        "labels": {
            0: "Relaxed",
            1: "Clench"
        }
    },
    MODEL_PATH
)

print(f"\nFinal model saved to: {MODEL_PATH}")
