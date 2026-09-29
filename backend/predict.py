"""
SIH26070 - Cyclone Intensity Prediction (ONNX Runtime).

Same model, same math as before - only the engine changed. The v3b +
v4_seed1 wind-regression ensemble (EfficientNet-B0, exported from the GEM
checkpoints, see backend/export_onnx.py) now runs in onnxruntime-CPU instead
of PyTorch, so the server needs ~100 MB instead of ~400 MB and fits Render's
free 512 MB instance with headroom.

Kept EXACTLY identical to the PyTorch backend (verified 10/10 same classes,
logit diff ~1e-6; see export verification):
  - preprocessing: RGB, Resize((224,224)) bilinear, /255, ImageNet normalize
  - 4-rotation test-time averaging, wind x100 scale, IMD class boundaries
  - confidence/probabilities via wind_math.class_probs_from_wind
  - contract: predict(PIL.Image) -> (category, confidence, all_probs)

Needs model_v3b.onnx and model_v4_seed1.onnx in this same folder. A missing
file or missing onnxruntime raises here, which server.py's Predictor catches
and falls back to DEMO MODE placeholders - same behaviour as before.

No torch/timm anywhere in this file (or in gradcam.py). torch is only needed
for the one-time backend/export_onnx.py step, never at runtime.
"""

import gc
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from wind_math import (
    CLASSES,
    MEASURED_MAE_KT,
    class_probs_from_wind,
    wind_to_class,
)

# Must stay IDENTICAL (same strings, same order) to backend/logic.py's
# CLASS_ORDER - server.py does logic.CLASS_ORDER.index(category) and
# {c: probs[c] for c in logic.CLASS_ORDER} on whatever predict() returns.

BACKEND_DIR = Path(__file__).parent
_MODEL_PATHS = [BACKEND_DIR / "model_v3b.onnx", BACKEND_DIR / "model_v4_seed1.onnx"]
_ASSETS_PATH = BACKEND_DIR / "onnx_assets.npz"

# Same deployment caveat as the torch backend: the reported 53.7% exact /
# 93.1% within-one-class numbers were measured with 4 rotations. Override
# only as a speed workaround via TTA_ROTATIONS.
TTA_ROTATIONS = int(os.environ.get("TTA_ROTATIONS", "4"))

# One intra-op thread: single-image CPU inference gains nothing from more
# threads and each pool adds peak RSS. Override: ORT_INTRA_THREADS=N.
_SESS_OPTS = ort.SessionOptions()
_SESS_OPTS.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", "1"))
_SESS_OPTS.inter_op_num_threads = 1

# Loaded once at import time. Missing file / missing onnxruntime raises,
# which server.py turns into clearly-labelled DEMO MODE.
_sessions = [ort.InferenceSession(str(p), sess_options=_SESS_OPTS,
                                  providers=["CPUExecutionProvider"])
             for p in _MODEL_PATHS]

# Classifier weights for the numpy CAM path in gradcam.py (v3b member: the
# same member the torch Grad-CAM explained - see gradcam.py).
_assets = np.load(str(_ASSETS_PATH))
_V3B_WEIGHT = _assets["model_v3b_w"].reshape(-1).astype(np.float64)

# Resize + /255 + ImageNet normalization - identical numbers to the torch
# pipeline (torchvision Resize bilinear + ToTensor + Normalize).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64).reshape(1, 3, 1, 1)


def preprocess(image):
    """PIL.Image -> (1, 3, 224, 224) float32 NCHW, ImageNet-normalized."""
    arr = np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR),
                      dtype=np.float64)
    tensor = (arr / 255.0).transpose(2, 0, 1)[None, :, :, :]
    return (((tensor - _MEAN) / _STD).astype(np.float32),)


def predict(image):
    """
    image: a PIL.Image (any mode - converted to RGB internally).
    Returns: (category: str, confidence: float as %, all_probs: dict of all
    5 classes -> %) - identical shape to the torch predict().

    Ensemble/TTA semantics match the torch backend exactly: for each of the
    4 rotations the two members' winds are averaged first, and the spread
    is the max-min of those 4 rotation-averaged winds.
    """
    (tensor,) = preprocess(image)
    # (member, rotation) wind grid; members run one after the other.
    grid = []
    for session in _sessions:  # one member at a time, never parallel
        member_winds = [
            float(session.run(["logit"],
                              {"input": np.rot90(tensor, k, (2, 3))})[0][0, 0]) * 100.0
            for k in range(TTA_ROTATIONS)
        ]
        grid.append(member_winds)
    del tensor
    gc.collect()
    rot_mean = [float(sum(w) / len(w)) for w in zip(*grid)]
    kt = float(sum(rot_mean) / len(rot_mean))
    spread = float(max(rot_mean) - min(rot_mean))
    del grid, rot_mean
    sigma = max(MEASURED_MAE_KT, spread)

    idx = wind_to_class(kt)
    category = CLASSES[idx]
    probs = class_probs_from_wind(kt, sigma)
    all_probs = {c: round(p * 100.0, 2) for c, p in zip(CLASSES, probs)}
    confidence = all_probs[category]
    return category, confidence, all_probs
