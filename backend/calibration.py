"""
Confidence calibration via temperature scaling.

Why this exists
----------------
CLAUDE.md documents that the shipped classifier's confidence is overconfident:
Expected Calibration Error (ECE) of about 27% on the held-out test set. A
reported "92% confidence" does NOT mean the model is right 92% of the time.

Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks") is the standard, minimally invasive fix: divide the model's raw
logits by a single learned scalar T > 1 before the softmax. This does not
change which class wins - argmax(logits / T) == argmax(logits) for any T > 0
- it only spreads out the probabilities so the reported confidence is closer
to the true chance of being correct.

This module has two torch-free parts (fully testable without torch/model.pth,
see backend/tests/test_calibration.py) and is used by two different places:
  - predict.py calls apply_temperature() on every real prediction.
  - backend/model_card.py (-> GET /api/model/transparency) calls
    load_temperature() to honestly report whether calibration has actually
    been fitted yet.

Fitting T itself needs torch + the real model.pth, neither of which exist in
the sandbox that wrote this code - see backend/fit_calibration.py, a separate
script meant to be run once by whoever has both, on their own machine.
"""

import json
from pathlib import Path

TEMPERATURE_PATH = Path(__file__).with_name("temperature.json")


def load_temperature():
    """Returns (temperature: float, fitted: bool).

    If backend/temperature.json does not exist or is unreadable/invalid,
    returns (1.0, False) - i.e. behaves exactly as if calibration had never
    been added (no change to predictions). This must never raise: a missing
    or corrupt calibration file should never be able to break a prediction.
    """
    try:
        data = json.loads(TEMPERATURE_PATH.read_text())
        t = float(data["temperature"])
        if t > 0:
            return t, True
    except Exception:  # noqa: BLE001 - calibration is best-effort, never fatal
        pass
    return 1.0, False


def apply_temperature(logits, temperature):
    """logits: a 1-D iterable of raw (pre-softmax) class scores.
    Returns a new list of the same length, each divided by `temperature`.

    temperature <= 0 or None is invalid input and is treated as 1.0 (a
    no-op), defensively - this function must never divide by zero or flip
    signs unexpectedly.
    """
    if temperature is None or temperature <= 0:
        temperature = 1.0
    return [float(x) / temperature for x in logits]


def compute_ece(probs, labels, n_bins=15):
    """Standard binned Expected Calibration Error - pure numpy, no torch.

    probs  : array-like of shape (N, C), softmax probabilities per class.
    labels : array-like of shape (N,), true class index per example.
    n_bins : number of equal-width confidence bins (15 is the value most
             commonly used in the calibration literature, e.g. Guo et al.).

    For each bin of predicted confidence (the winning class's probability),
    compares that bin's average confidence to its average accuracy, weights
    by how many examples fall in the bin, and sums the (weighted) gaps.
    Returns a float in [0, 1]; multiply by 100 to compare against the ~27%
    figure quoted elsewhere in this project.
    """
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    if probs.ndim != 2 or len(labels) != probs.shape[0] or len(labels) == 0:
        raise ValueError("probs must be (N, C) and labels must be length N, N > 0.")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
