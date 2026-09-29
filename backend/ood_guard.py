"""
Out-of-distribution (OOD) input guard - a heuristic safety net, NOT a learned model.

Why this exists
----------------
predict.py always returns a confident-looking class + probabilities, even for
an image that looks nothing like what the model was trained on (a phone photo,
a screenshot, a solid-color image, a chart). Softmax always sums to 100%
regardless of whether the input made any sense, so there is no free signal
telling the user "this input is weird" - unless something checks for it.

This module computes three cheap statistics from an image:
  - mean brightness (0-255 grayscale)
  - brightness std / contrast
  - "colorfulness" (mean absolute difference between the R, G and B channels)
and compares them against a reference range measured directly from this
project's own labelled TCIR examples in backend/samples/*.png - the only real
examples of "what the model was trained on" available in this repo.

This is a heuristic, calibrated from only 10 reference images. It will not
catch every out-of-distribution input, and it can occasionally flag a
legitimate-but-unusual image. It is a disclosed, best-effort guard, not a
trained OOD detector - do not oversell it as one.

Reference stats measured from backend/samples/*.png (10 images, 201x201,
true single-channel IR data replicated into RGB, so colorfulness ~= 0):
    mean brightness : 115.9 - 199.9
    brightness std   : 39.0  - 68.9
    colorfulness      : ~0.00 (true grayscale)
"""

from pathlib import Path

import numpy as np
from PIL import Image

SAMPLES_DIR = Path(__file__).with_name("samples")

# Fallback constants (see docstring) used if backend/samples/ is ever missing
# or unreadable at import time - the guard must never crash server startup.
_FALLBACK_MEAN_RANGE = (115.9, 199.9)
_FALLBACK_STD_RANGE = (39.0, 68.9)

# How far outside the observed reference range before we treat something as
# a soft "worth a warning" case vs. a stronger "likely not a TCIR IR chip" case.
#
# _STRONG_PAD for brightness was originally 55.0. Lowered to 35.0 after a real
# user upload (a document/bill-style image: brightness 244.1 vs a reference
# max of 199.9, a diff of 44.2) landed in the "warning" band and still got a
# full confident classification instead of being rejected. 35.0 catches that
# case (and anything further out) while still leaving a 25-35 "warning" zone
# for genuinely borderline images. See CLAUDE.md's OOD guard section.
_SOFT_PAD = 25.0
_STRONG_PAD = 35.0
_COLORFULNESS_SOFT = 6.0
_COLORFULNESS_STRONG = 15.0

# If two or more signals are each only "warning"-level on their own, that
# combination is treated as strong evidence together (a document that is a
# bit too bright AND a bit too flat/uncolorful, say) even though neither one
# alone crossed the strong threshold above.
_COMPOUND_WARNING_THRESHOLD = 2


def _measure(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    gray = arr.mean(axis=2)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    colorfulness = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3.0
    return float(gray.mean()), float(gray.std()), float(colorfulness)


def _reference_ranges():
    """Recompute the reference range from the real sample images if possible,
    so the guard stays accurate if samples/ is ever changed. Falls back to the
    hard-coded constants above (measured once, see docstring) if that fails
    for any reason - this must never raise."""
    try:
        means, stds = [], []
        paths = sorted(SAMPLES_DIR.glob("*.png"))
        if not paths:
            raise FileNotFoundError("no sample images found")
        for path in paths:
            mean_b, std_b, _ = _measure(path)
            means.append(mean_b)
            stds.append(std_b)
        return (min(means), max(means)), (min(stds), max(stds))
    except Exception:  # noqa: BLE001 - the guard is best-effort, never fatal
        return _FALLBACK_MEAN_RANGE, _FALLBACK_STD_RANGE


MEAN_RANGE, STD_RANGE = _reference_ranges()


def assess(image):
    """image: a PIL.Image (any mode).

    Returns a dict:
        {
          "level": "none" | "warning" | "likely_ood",
          "reasons": [str, ...],
          "stats": {"mean_brightness": float, "brightness_std": float, "colorfulness": float},
        }
    """
    mean_b, std_b, colorfulness = _measure_image(image)
    reasons = []
    level = "none"
    warning_signal_count = 0

    def bump(new_level):
        nonlocal level, warning_signal_count
        if new_level == "warning":
            warning_signal_count += 1
        order = {"none": 0, "warning": 1, "likely_ood": 2}
        if order[new_level] > order[level]:
            level = new_level

    if colorfulness > _COLORFULNESS_STRONG:
        reasons.append(
            f"Image has strong color content (colorfulness score {colorfulness:.1f}). "
            "The model was trained only on single-channel infrared images that look "
            "grayscale, so this is probably not a TCIR-style IR chip."
        )
        bump("likely_ood")
    elif colorfulness > _COLORFULNESS_SOFT:
        reasons.append(
            f"Image has some color content (colorfulness score {colorfulness:.1f}) - "
            "TCIR training chips are effectively grayscale."
        )
        bump("warning")

    lo, hi = MEAN_RANGE
    if mean_b < lo - _STRONG_PAD or mean_b > hi + _STRONG_PAD:
        reasons.append(
            f"Average brightness ({mean_b:.0f}/255) is far outside the range measured "
            f"from this project's own sample IR images ({lo:.0f}-{hi:.0f})."
        )
        bump("likely_ood")
    elif mean_b < lo - _SOFT_PAD or mean_b > hi + _SOFT_PAD:
        reasons.append(
            f"Average brightness ({mean_b:.0f}/255) is outside the range measured "
            f"from this project's own sample IR images ({lo:.0f}-{hi:.0f})."
        )
        bump("warning")

    slo, shi = STD_RANGE
    if std_b < max(slo - _STRONG_PAD, 2.0):
        reasons.append(
            f"Image has very little contrast (std {std_b:.0f}) - looks close to a "
            "solid or near-uniform color, unlike a real cloud-top IR image."
        )
        bump("likely_ood")
    elif std_b < slo - _SOFT_PAD:
        reasons.append(f"Image contrast (std {std_b:.0f}) is lower than the reference range.")
        bump("warning")

    # Compound signals: no single measurement crossed the strong threshold on
    # its own, but two or more independent measurements are each borderline
    # at once - that combination is itself evidence this isn't a TCIR chip
    # (e.g. a document that is a bit too bright AND a bit too low-contrast).
    if level == "warning" and warning_signal_count >= _COMPOUND_WARNING_THRESHOLD:
        reasons.append(
            f"{warning_signal_count} separate measurements (brightness/contrast/colorfulness) "
            "are each individually borderline - taken together this looks unlikely to be a "
            "TCIR-style IR chip, even though no single measurement was extreme enough on its own."
        )
        level = "likely_ood"

    return {
        "level": level,
        "reasons": reasons,
        "stats": {
            "mean_brightness": round(mean_b, 1),
            "brightness_std": round(std_b, 1),
            "colorfulness": round(colorfulness, 2),
        },
        "reference": {
            "mean_brightness_range": [round(lo, 1), round(hi, 1)],
            "brightness_std_range": [round(slo, 1), round(shi, 1)],
            "source": "backend/samples/*.png (10 labelled TCIR examples shipped with this project)",
        },
    }


def _measure_image(image):
    arr = np.asarray(image.convert("RGB"), dtype=float)
    gray = arr.mean(axis=2)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    colorfulness = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3.0
    return float(gray.mean()), float(gray.std()), float(colorfulness)
