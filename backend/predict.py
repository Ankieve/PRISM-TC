"""
SIH26070 - Cyclone Intensity Prediction
v3 - wind-speed regression ENSEMBLE (model_v3b.pth + model_v4_seed1.pth),
replacing the v2 direct-classification model.pth.

Why this replaced v2
---------------------
v2 (5-class classifier, single model) was evaluated two different ways during
this project:
  - Its own original 372-image held-out test: 60.5% exact.
  - A storm-wise (grouped by storm_id) held-out test of 12 unseen storms,
    520 images, built later specifically to catch leakage: 54.8% exact, but
    97.2% "train" exact on that same split - a 42.4-point train-test gap.
    That gap is the signature of overfitting: earlier train/test splits let
    images from the same storm appear on both sides, so v2 had partly
    memorised storms rather than learned to generalise. See CLAUDE.md and
    MODAK.ipynb for the full diagnosis.

The v3b + v4_seed1 ensemble was trained on the SAME TCIR dataset (no new
MOSDAC data is in it yet - that pipeline is still being built, see
backend/predict_insat_tools.py and prism_tc_build_dataset.py) but with a
storm-wise split enforced from the start, a wind-speed regression target
instead of a direct class label, and 4-rotation test-time averaging.
Measured on the same 12-storm/520-image storm-wise test set:
    exact 53.7%, within-one-class 93.1%, macro-F1 0.487,
    MAE 10.2 kt, train-test gap 6.0 points (vs v2's 42.4).

Honest trade-off - read before presenting this as a strict improvement
------------------------------------------------------------------------
On a live validation against real MOSDAC INSAT-3DR imagery of Cyclone
Biparjoy (36 frames, IBTrACS ground truth), v2 actually scored HIGHER exact
accuracy than the ensemble (v2 83.3% vs ensemble 75.0%), though the ensemble
won on within-one-class (100% vs 94.4%) and adds an MAE figure v2 cannot
produce (4.9 kt). v2's Biparjoy number may partly reflect the same
overfitting the storm-wise test exposed - Biparjoy-like conditions may
simply be well represented in what v2 memorised - so it is not a reason to
prefer v2 for new, unseen storms. But it means the honest claim is "more
consistent across many unseen storms, not uniformly more accurate on every
one," not "strictly better." See MODAK.ipynb's final comparison cells.

IMPORTANT input pipeline change vs v2: v3b/v4_seed1 WERE trained with
standard ImageNet normalization (unlike v2, which was explicitly trained
WITHOUT it - see the old predict.py in git history). Do not remove the
transforms.Normalize call below; that would silently hurt accuracy the same
way adding it to v2 used to.

Usage in server.py (unchanged contract):
    from predict import predict
    category, confidence, all_probs = predict(uploaded_image)   # PIL.Image

Confidence here is NOT a softmax probability (the model has one regression
output, not 5 class logits). It is derived by treating the model's wind
estimate as the mean of a Normal distribution whose spread (sigma) is
max(the model's own measured test MAE of 10.2 kt, this image's 4-rotation
disagreement) and integrating that distribution over each class's IMD wind
boundaries. This is a documented derivation, not a measured calibration
curve - no Expected Calibration Error has been computed for it. Do not
describe it to judges as "the model's confidence" without this caveat; see
CLAUDE.md.

Needs model_v3b.pth and model_v4_seed1.pth in this same folder. Falls back
to raising (caught by server.py's Predictor, which then serves clearly
labelled DEMO MODE placeholders) if they are missing or torch/timm are not
installed - same fallback behaviour as before.
"""

import os
from pathlib import Path

import gc
import numpy as np
import torch

# Memory: single-threaded inference. torch's default uses all cores (8 here),
# each thread pool + oneDNN workspace adding tens of MB of peak RSS per
# request for zero accuracy benefit on single-image CPU inference.
# Override only for local benchmarking: TORCH_NUM_THREADS=N.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
import timm
from torchvision import transforms

from wind_math import (
    CLASSES,
    MEASURED_MAE_KT,
    class_probs_from_wind,
    wind_to_class,
)

# Must stay IDENTICAL (same strings, same order) to backend/logic.py's
# CLASS_ORDER - server.py does logic.CLASS_ORDER.index(category) and
# {c: probs[c] for c in logic.CLASS_ORDER} on whatever predict() returns, so
# any mismatch here throws on every single prediction request. (The
# teammate's original new predict.py used 'Depression/Deep Depression' for
# class 0, which does NOT match logic.CLASS_ORDER's 'Depression' - fixed in
# wind_math.py rather than carried over. See backend/tests/test_wind_math.py,
# which asserts CLASSES == logic.CLASS_ORDER directly.)

BACKEND_DIR = Path(__file__).parent
_MODEL_PATHS = [BACKEND_DIR / "model_v3b.pth", BACKEND_DIR / "model_v4_seed1.pth"]

# Number of test-time-augmentation rotations per prediction. The measured
# 53.7% exact / 93.1% within-one-class numbers (CLAUDE.md, model_card.py,
# MODAK.ipynb cell 21) were all measured with the default of 4. Lower this
# ONLY as a deployment-specific speed/memory workaround (e.g. a slow free-tier
# CPU host timing out mid-request) via the TTA_ROTATIONS env var - do not
# change the default here, or the live site's real accuracy will quietly
# drift from the numbers the dashboard/model card claim. A deployment running
# fewer than 4 rotations is measurably less accurate than the tested figures
# (TTA rotation-averaging is part of what was measured) and should say so
# wherever those numbers are shown, rather than presenting them as-is.
TTA_ROTATIONS = int(os.environ.get("TTA_ROTATIONS", "4"))


class _WindEnsemble(torch.nn.Module):
    """Averages the wind output of the ensemble members, one after the
    other, freeing each member's output before running the next - peak
    memory stays at one member + one output tensor, never both members
    plus a stacked buffer."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        total = None
        for m in self.members:
            out = m(x)
            total = out if total is None else total + out
            del out
        result = total / len(self.members)
        del total
        return result


def _load_member(path):
    sd = torch.load(str(path), map_location=torch.device("cpu"))
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    n_out = sd["classifier.weight"].shape[0]
    if n_out != 1:
        raise ValueError(
            f"{path.name}: expected a 1-output wind-regression checkpoint, "
            f"got {n_out} outputs - wrong file?")
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    m.load_state_dict(sd)
    m.eval()
    return m


# Loaded once at import time, same as v2 did - a missing file or missing
# torch/timm raises here, which server.py's Predictor.__init__ catches and
# falls back to DEMO MODE placeholders for.
_members = [_load_member(p) for p in _MODEL_PATHS]

# Exposed for backend/gradcam.py (`from predict import _model, ...`): one
# representative member, since Grad-CAM needs a single concrete conv
# architecture with a conv_head, not an ensemble wrapper. This explains what
# drove model_v3b's wind estimate, not a blended ensemble explanation - a
# real simplification, noted in gradcam.py.
_model = _members[0]

_ensemble = _WindEnsemble(_members)
_ensemble.eval()

# Resize + tensor + ImageNet normalization - matches how v3b/v4_seed1 were
# trained (see module docstring; this DIFFERS from v2's no-normalization
# pipeline). Exposed for gradcam.py too.
_predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def predict(image):
    """
    image: a PIL.Image (any mode - converted to RGB internally).
    Returns: (category: str, confidence: float as %, all_probs: dict of all
    5 classes -> %) - identical shape to the old v2 predict(), so
    server.py's Predictor needs no changes.
    """
    image = image.convert("RGB")
    tensor = _predict_transform(image).unsqueeze(0)
    with torch.no_grad():
        # Test-time-averaging across TTA_ROTATIONS rotations (default 4,
        # see the module-level comment above) - cyclones look the same
        # rotated, so disagreement across rotations is a real per-image
        # uncertainty signal (see MEASURED_MAE_KT above for why it's floored).
        winds = np.array([
            float(_ensemble(torch.rot90(tensor, k, (2, 3)))[0, 0]) * 100.0
            for k in range(TTA_ROTATIONS)
        ])
    del tensor
    kt = float(winds.mean())
    spread = float(winds.max() - winds.min())
    del winds
    gc.collect()
    sigma = max(MEASURED_MAE_KT, spread)

    idx = wind_to_class(kt)
    category = CLASSES[idx]
    probs = class_probs_from_wind(kt, sigma)
    all_probs = {c: round(p * 100.0, 2) for c, p in zip(CLASSES, probs)}
    confidence = all_probs[category]
    return category, confidence, all_probs
