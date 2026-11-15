
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

FEATURES = [
    "MAV", "RMS", "WL", "ZC",
    "SSC", "MNF", "MDF", "LF"
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--windows", default="results/binary_windows.csv"
)
parser.add_argument(
    "--save", default="results/svm_clench_vs_rest.joblib"
)
args = parser.parse_args()

df = pd.read_csv(args.windows)

# Train only on relaxed (0) and clench (1).
binary = df[df["label"].isin([0, 1])].copy()
open_hand = df[df["label"] == 2].copy()

subjects = sorted(binary["subject"].unique())

print("Binary training windows:", len(binary))
print("Subjects:", subjects)

all_true = []
all_pred = []
fold_scores = []

for subject in subjects:
    train = binary[binary["subject"] != subject]
    test = binary[binary["subject"] == subject]

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )
    )

    model.fit(train[FEATURES], train["label"])
    predictions = model.predict(test[FEATURES])

    score = balanced_accuracy_score(
        test["label"], predictions
    )

    fold_scores.append(score)
    all_true.extend(test["label"])
    all_pred.extend(predictions)

    print(
        f"Subject {subject}: "
        f"balanced accuracy = {score:.3f}"
    )

print("\nMean LOSO balanced accuracy:",
      round(np.mean(fold_scores), 3))

print("\nConfusion matrix:")
print(confusion_matrix(
    all_true, all_pred, labels=[0, 1]
))

print("\nClassification report:")
print(classification_report(
    all_true, all_pred,
    labels=[0, 1],
    target_names=["Relaxed", "Clench"],
    zero_division=0,
))

# Separate evaluation on open-hand recordings.
# Each open-hand subject is tested with a model
# trained on the other subjects.
open_predictions = []

for subject in subjects:
    train = binary[binary["subject"] != subject]
    test_open = open_hand[
        open_hand["subject"] == subject
    ]

    if test_open.empty:
        continue

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale",
        )
    )

    model.fit(train[FEATURES], train["label"])
    open_predictions.extend(
        model.predict(test_open[FEATURES])
    )

if open_predictions:
    false_clench_rate = np.mean(
        np.asarray(open_predictions) == 1
    )
    print(
        "\nOpen-hand false-clench rate:",
        round(false_clench_rate, 3)
    )

# Train final model on all available binary data.
final_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )
)

final_model.fit(binary[FEATURES], binary["label"])

output = Path(args.save)
output.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {"model": final_model, "features": FEATURES},
    output
)

print(f"\nFinal model saved to: {output}")
