from pathlib import Path

import joblib
import numpy as np
import pandas as pd


folder = Path(__file__).resolve().parent.parent

model_file = folder / "results" / "emg_clench_final.joblib"

header_file = folder / "stm32" / "Inc" / "emg_model_params.h"

header_file.parent.mkdir(parents=True, exist_ok=True)

if not model_file.is_file():
    raise FileNotFoundError(f"Place emg_clench_final.joblib in {folder}")

bundle = joblib.load(model_file)
model = bundle["model"]
scaler = model.named_steps["standardscaler"]
svm = model.named_steps["svc"]
support_vectors = svm.support_vectors_

# Select two inputs far from the decision boundary for the STM32 self-test.
inputs = scaler.inverse_transform(support_vectors)
features = pd.DataFrame(inputs, columns=bundle["features"])
scores = model.decision_function(features)
check = [int(np.argmin(scores)), int(np.argmax(scores))]


def c_values(values):
    if np.ndim(values) == 0:
        return f"{float(values):.9e}f"
    separator = ",\n    " if np.ndim(values) == 2 else ", "
    return "{ " + separator.join(c_values(value) for value in values) + " }"


fields = [
    ("emg_mean[8]", scaler.mean_),
    ("emg_scale[8]", scaler.scale_),
    ("emg_sv[EMG_SV_COUNT][8]", support_vectors),
    ("emg_dual[EMG_SV_COUNT]", svm.dual_coef_[0]),
    ("emg_bias", svm.intercept_[0]),
    ("emg_gamma", svm._gamma),
    ("emg_test_x[2][8]", inputs[check]),
    ("emg_test_score[2]", scores[check]),
]

lines = [
    "#pragma once",
    f"#define EMG_SV_COUNT {len(support_vectors)}",
    "",
]
for name, values in fields:
    lines.append(f"static const float {name} = {c_values(values)};")

header_file.write_text("\n".join(lines) + "\n", encoding="ascii")

print(f"Support vector count: {len(support_vectors)}")
print(f"Self-test reference scores: {scores[check]}")
print(f"Header file generated: {header_file}")