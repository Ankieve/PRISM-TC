"""Tests for the torch-free half of backend/calibration.py.  Run:
    python backend/tests/test_calibration.py

Deliberately does NOT import torch - fitting a real temperature needs torch
and the real model.pth (see backend/fit_calibration.py, run on a machine that
has both). This file only checks the math that predict.py actually runs on
every request: apply_temperature() and the ECE metric used to judge it.
"""
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import calibration  # noqa: E402

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------- apply_temperature
logits = [2.0, 1.0, 0.1, -1.0, -2.0]
check("T=1.0 is a no-op", calibration.apply_temperature(logits, 1.0) == logits)
check("T=None is a no-op", calibration.apply_temperature(logits, None) == logits)
check("T=0 falls back to no-op (never divides by zero)", calibration.apply_temperature(logits, 0) == logits)
check("T=-5 falls back to no-op", calibration.apply_temperature(logits, -5) == logits)
check("T=2.0 halves every logit", calibration.apply_temperature(logits, 2.0) == [1.0, 0.5, 0.05, -0.5, -1.0])

rng = np.random.default_rng(0)
argmax_preserved = True
for _ in range(500):
    lg = rng.normal(0, 3, size=5)
    for T in [0.3, 1.0, 2.5, 10.0]:
        scaled = calibration.apply_temperature(lg.tolist(), T)
        if np.argmax(scaled) != np.argmax(lg):
            argmax_preserved = False
check("temperature scaling never changes the predicted class (500 trials x 4 temperatures)", argmax_preserved)

# --------------------------------------------------------------- load_temperature
t, fitted = calibration.load_temperature()
check("load_temperature returns (1.0, False) with no temperature.json present",
      not calibration.TEMPERATURE_PATH.is_file() and (t, fitted) == (1.0, False) or calibration.TEMPERATURE_PATH.is_file(),
      f"got {(t, fitted)}, file exists={calibration.TEMPERATURE_PATH.is_file()}")

# ------------------------------------------------------------------- compute_ece
n = 4000
confidences = rng.uniform(0.2, 1.0, n)
correct = rng.uniform(0, 1, n) < confidences
labels = np.zeros(n, dtype=int)
probs = np.zeros((n, 5))
for i in range(n):
    if correct[i]:
        probs[i, 0] = confidences[i]
        probs[i, 1:] = (1 - confidences[i]) / 4
    else:
        probs[i, 1] = confidences[i]
        probs[i, 0] = (1 - confidences[i]) / 4
        probs[i, 2:] = (1 - confidences[i]) / 4
ece_calibrated = calibration.compute_ece(probs, labels)
check("a near-perfectly-calibrated synthetic set scores low ECE", ece_calibrated < 0.05, str(ece_calibrated))

n2 = 4000
labels2 = np.zeros(n2, dtype=int)
probs2 = np.zeros((n2, 5))
is_correct2 = rng.uniform(0, 1, n2) < 0.6
for i in range(n2):
    if is_correct2[i]:
        probs2[i, 0] = 0.95
        probs2[i, 1:] = 0.05 / 4
    else:
        probs2[i, 1] = 0.95
        probs2[i, 0] = 0.05 / 4
        probs2[i, 2:] = 0.05 / 4
ece_overconfident = calibration.compute_ece(probs2, labels2)
check("an overconfident synthetic set (95% conf, 60% accuracy) scores high ECE",
      ece_overconfident > 0.25, str(ece_overconfident))
check("overconfident set scores worse than calibrated set", ece_overconfident > ece_calibrated)

try:
    calibration.compute_ece(np.zeros((0, 5)), np.zeros(0))
    check("compute_ece rejects empty input", False)
except ValueError:
    check("compute_ece rejects empty input", True)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASSED")
