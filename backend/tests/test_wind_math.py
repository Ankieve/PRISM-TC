"""Tests for the torch-free half of the v3 ensemble predict.py (backend/wind_math.py).  Run:
    python backend/tests/test_wind_math.py

Deliberately does NOT import predict.py or torch - loading the real ensemble
needs torch, timm and the real model_v3b.pth/model_v4_seed1.pth checkpoints
(see backend/tests/smoke_test.py --real, which requires all three). This file
covers the part that is actually the highest-risk to get wrong silently: the
class-label contract with server.py/logic.py, and the wind -> probability
math. See wind_math.py's own docstring for why this split exists (the same
reasoning as backend/calibration.py / test_calibration.py).
"""
import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import logic  # noqa: E402
import wind_math  # noqa: E402

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


# ------------------------------------------------ the contract with server.py
# This is the single most important check in this file: if these two lists
# ever diverge, server.py's Predictor.classify() throws on EVERY prediction
# request (logic.CLASS_ORDER.index(category) / {c: probs[c] for c in
# logic.CLASS_ORDER}), which is exactly the "destroy my software" failure
# mode this integration was checked for. See predict.py's own comment on this.
check("wind_math.CLASSES matches logic.CLASS_ORDER exactly",
      wind_math.CLASSES == logic.CLASS_ORDER,
      f"{wind_math.CLASSES} vs {logic.CLASS_ORDER}")

# ------------------------------------------------------------- wind_to_class
check("34 kt is the CS boundary (>=34 -> class 1)", wind_math.wind_to_class(34) == 1)
check("33.9 kt is still Depression (class 0)", wind_math.wind_to_class(33.9) == 0)
check("48 kt is the SCS boundary (>=48 -> class 2)", wind_math.wind_to_class(48) == 2)
check("90 kt is the top class (class 4)", wind_math.wind_to_class(90) == 4)
check("200 kt is still the top class (class 4)", wind_math.wind_to_class(200) == 4)
check("0 kt is Depression (class 0)", wind_math.wind_to_class(0) == 0)

# ------------------------------------------------------- class_probs_from_wind
for kt, sigma in [(20, 10.2), (34, 10.2), (75, 10.2), (100, 10.2), (48, 2.0), (0, 10.2), (200, 10.2)]:
    probs = wind_math.class_probs_from_wind(kt, sigma)
    check(f"probs sum to 1 (kt={kt}, sigma={sigma})",
          math.isclose(sum(probs), 1.0, abs_tol=1e-6), str(probs))
    check(f"probs has 5 entries, none negative (kt={kt}, sigma={sigma})",
          len(probs) == 5 and all(p >= 0 for p in probs), str(probs))

# A point estimate right at a class boundary should split ~50/50 between the
# two neighbouring classes, not silently favour one - this would be the
# subtle way to get an off-by-one wrong in the interval edges.
probs_at_48 = wind_math.class_probs_from_wind(48, 2.0)
check("wind estimate at the 48kt SCS boundary splits evenly either side",
      math.isclose(probs_at_48[1], probs_at_48[2], abs_tol=1e-6), str(probs_at_48))

# A very confident (tiny-sigma) estimate deep inside one class should put
# almost all probability mass on that class.
probs_confident = wind_math.class_probs_from_wind(75, 0.5)
check("tiny sigma deep in a class concentrates probability there",
      probs_confident[3] > 0.99, str(probs_confident))

# Degenerate sigma must never crash or divide by zero.
try:
    degenerate = wind_math.class_probs_from_wind(50, 0.0)
    check("sigma=0 does not crash and still sums to 1",
          math.isclose(sum(degenerate), 1.0, abs_tol=1e-6), str(degenerate))
except ZeroDivisionError:
    check("sigma=0 does not crash and still sums to 1", False, "raised ZeroDivisionError")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASSED")
