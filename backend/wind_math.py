"""
Torch-free half of predict.py's v3 ensemble: converting a wind-speed
estimate (+ its uncertainty) into an IMD class and a set of derived class
probabilities. Split out from predict.py so this logic - the part with real
room for an off-by-one or a sign error - can be unit-tested (see
backend/tests/test_wind_math.py) without needing torch, timm, or the real
model checkpoints, the same reasoning that split calibration.py out of the
old predict.py.

CLASSES here MUST stay identical (same strings, same order) to
backend/logic.py's CLASS_ORDER - see predict.py's docstring for what breaks
if they ever drift apart. test_wind_math.py asserts this directly.
"""

import math

import numpy as np

CLASSES = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe/Super Cyclone",
]

# IMD class boundaries in knots (3-min sustained wind).
WIND_THRESHOLDS_KT = [34, 48, 64, 90]

# The ensemble's own measured mean absolute error on its storm-wise held-out
# test (12 unseen storms, MODAK.ipynb "ENSEMBLE v3b+s1" row). Used as a floor
# on the uncertainty behind the derived confidence, so a prediction is never
# presented as more precise than the model has actually been shown to be.
MEASURED_MAE_KT = 10.2


def wind_to_class(kt):
    """kt -> IMD class index 0-4, via WIND_THRESHOLDS_KT."""
    return int(np.digitize(kt, WIND_THRESHOLDS_KT))


def _normal_cdf(x, mean, sigma):
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))


def class_probs_from_wind(kt, sigma):
    """Turns a point wind estimate + its uncertainty into one probability per
    IMD class, by integrating Normal(kt, sigma) over each class's knot
    interval. This is deterministic math given kt/sigma, not a fitted
    calibration - see predict.py's docstring for why. Returns a list of 5
    floats summing to 1.0 (renormalised defensively against float error).

    predict.py always calls this with sigma = max(MEASURED_MAE_KT, spread),
    which can never be <= 0, so this guard is defense-in-depth for any other
    caller (e.g. a future direct use of this module) rather than a path
    production traffic can reach."""
    if not sigma or sigma <= 0:
        # A zero/negative sigma has no meaningful distribution to integrate
        # (and would divide by zero below) - fall back to a one-hot on the
        # point estimate rather than raising.
        probs = [0.0] * len(CLASSES)
        probs[wind_to_class(kt)] = 1.0
        return probs
    edges = [-math.inf] + list(WIND_THRESHOLDS_KT) + [math.inf]
    raw = [_normal_cdf(hi, kt, sigma) - _normal_cdf(lo, kt, sigma)
           for lo, hi in zip(edges[:-1], edges[1:])]
    total = sum(raw)
    if total <= 0:
        # Degenerate (numerical underflow far from kt) - never divide by
        # zero; fall back to a one-hot on the point estimate.
        probs = [0.0] * len(CLASSES)
        probs[wind_to_class(kt)] = 1.0
        return probs
    return [p / total for p in raw]
