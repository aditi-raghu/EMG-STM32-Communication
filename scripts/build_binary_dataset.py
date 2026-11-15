
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from emgprep import clean_emg, window_features

DATA_DIR = Path("data/labeled")
OUTPUT = Path("results/binary_windows.csv")

FS = 400
WINDOW_SECONDS = 0.25
STEP_SECONDS = 0.125
CHANNEL = "Channel 4"

window_size = round(FS * WINDOW_SECONDS)
step_size = round(FS * STEP_SECONDS)

rows = []

files = sorted(DATA_DIR.rglob("*.csv"))
print(f"Found {len(files)} CSV files")

if not files:
    raise FileNotFoundError("No CSV files in data/labeled")

for file in files:
    df = pd.read_csv(file)

    required = [
        CHANNEL, "label", "time_s",
        "subject_id", "session_id"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{file.name}: missing {missing}")

    raw = df[CHANNEL].to_numpy(dtype=float)
    labels = df["label"].to_numpy()

    # Filter the full recording before windowing.
    filtered = clean_emg(raw, FS)

    for start in range(
        0, len(raw) - window_size + 1, step_size
    ):
        end = start + window_size
        window_labels = labels[start:end]

        # Only retain windows containing a single label.
        # This excludes transitions between movements.
        if not np.all(window_labels == window_labels[0]):
            continue

        label = int(window_labels[0])

        if label not in (0, 1, 2):
            continue

        features = window_features(
            filtered[start:end],
            raw[start:end],
            FS
        )

        features.update({
            "label": label,
            "subject": str(df["subject_id"].iloc[start]),
            "session": str(df["session_id"].iloc[start]),
            "recording": file.name,
            "start_sample": start,
        })

        rows.append(features)

    print(f"Processed {file.name}")

dataset = pd.DataFrame(rows)

if dataset.empty:
    raise ValueError("No valid windows were generated")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT, index=False)

print("\nDataset created successfully!")
print(f"Total windows: {len(dataset)}")
print("\nLabel distribution:")
print(dataset["label"].value_counts().sort_index())
print(f"\nSaved to: {OUTPUT}")
