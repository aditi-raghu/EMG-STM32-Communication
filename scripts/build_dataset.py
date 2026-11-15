"""Build a labelled window dataset from data/No.*/<subject>.
Usage: python scripts/build_dataset.py --data data --out results/windows.csv
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emgprep import load_recording, clean, envelope, detect_events, window_features

CLASSES = {"No.1": "close", "No.2": "close", "No.3": "open", "No.4": "artifact"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results/windows.csv")
ap.add_argument("--win", type=float, default=0.25)
ap.add_argument("--step", type=float, default=0.125)
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.data, "No.*", "*"))):
    trial, name = os.path.basename(os.path.dirname(f)), os.path.basename(f)
    if "(fei)" in name or name.endswith(".md") or trial not in CLASSES:
        continue
    subject = re.sub(r"-.*", "", name)
    x, fs = load_recording(f)
    y = clean(x, fs)
    env = envelope(y, fs)
    ev = detect_events(env, fs, keep_frac=0.15 if trial == "No.4" else 0.3)
    active = np.zeros(len(y))
    for s, e in ev:
        active[s:e] = 1
    print(f"{trial}/{name:14s} fs={fs:5.1f} Hz  events={len(ev)}")
    W, S = int(a.win * fs), int(a.step * fs)
    for i in range(0, len(y) - W, S):
        frac = active[i:i + W].mean()
        if 0.2 < frac < 0.8:          # skip transition windows
            continue
        r = window_features(y[i:i + W], x[i:i + W], fs)
        r.update(subject=subject, trial=trial, t=i / fs,
                 label=CLASSES[trial] if frac >= 0.8 else "rest")
        rows.append(r)

df = pd.DataFrame(rows)
for c in ["MAV", "RMS", "WL"]:     # per-subject normalisation to the rest level
    ref = df[df.label == "rest"].groupby("subject")[c].median()
    df[c + "_n"] = np.log(df[c] / df.subject.map(ref))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(df.label.value_counts().to_string())
