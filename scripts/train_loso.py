"""Leave-one-subject-out evaluation.
Usage: python scripts/train_loso.py --windows results/windows.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["MAV_n", "RMS_n", "WL_n", "ZC", "SSC", "MNF", "MDF", "LF"]
LABELS = ["rest", "close", "open", "artifact"]

ap = argparse.ArgumentParser()
ap.add_argument("--windows", default="results/windows.csv")
df = pd.read_csv(ap.parse_args().windows)

models = {
    "LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "RF": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1),
}
for name, make in models.items():
    pred, true, acc = [], [], []
    for s in sorted(df.subject.unique()):
        tr, te = df[df.subject != s], df[df.subject == s]
        p = make().fit(tr[FEATURES], tr.label).predict(te[FEATURES])
        pred += list(p); true += list(te.label)
        acc.append(balanced_accuracy_score(te.label, p))
    print(f"\n{name}: mean LOSO balanced accuracy {np.mean(acc):.2f} "
          f"(range {min(acc):.2f}-{max(acc):.2f})")
    print(pd.DataFrame(confusion_matrix(true, pred, labels=LABELS), index=LABELS, columns=LABELS))
