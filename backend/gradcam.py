"""
Grad-CAM overlay for PRISM-TC's EfficientNet-B0 wind-regression ensemble,
computed in NUMPY from the ONNX runtime - no torch needed.

Reference: Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization".

Why this is exactly Grad-CAM (not an approximation): the head is a
global-average-pool + linear layer, so d(score)/dF == w/49 for every spatial
position - the gradient-average weights reduce to the classifier weights
themselves (verified: numpy CAM correlates >0.999 with the old torch
hook-based Grad-CAM on the sample images). The heatmap explained is the v3b
member's wind estimate - the same member the torch version explained, since
an ensemble has no single conv architecture to hook.

What this actually shows (read before demoing it)
---------------------------------------------------
A coarse heatmap - upsampled from a 7x7 feature grid, since EfficientNet-B0
downsamples a 224x224 input by 32x - showing which regions most influenced
the wind estimate. It is a debugging/sanity-check visualization, not
proof the model is "looking at the storm for the right meteorological
reasons". Present it as "here is a coarse heatmap of what the network
weighted most," not as validated scientific attribution.
"""

import base64
import io

import numpy as np
from PIL import Image


def compute_cam(features, weight):
    """features: (1280, 7, 7) post-activation map. weight: (1280,) classifier
    weights. Returns a (7, 7) array in [0, 1]."""
    cam = np.maximum((np.asarray(features, dtype=np.float64)
                      * np.asarray(weight, dtype=np.float64).reshape(-1, 1, 1)
                      ).sum(axis=0), 0.0)
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam


def _jet_colormap(values):
    """values: 2D array in [0, 1]. Returns an (H, W, 3) uint8 RGB array using
    a cheap analytic approximation of the classic 'jet' colormap - avoids
    adding matplotlib/opencv as a dependency (neither is in requirements.txt)."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(original_image, cam, alpha=0.45):
    """original_image: PIL.Image (any size/mode). cam: 2D array in [0, 1].
    Returns a PIL.Image the same size as original_image with the heatmap
    alpha-blended on top."""
    original = original_image.convert("RGB")
    w, h = original.size
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heatmap = _jet_colormap(cam_resized)
    base = np.asarray(original, dtype=np.float32)
    blended = base * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def explain(image, class_index):
    """image: PIL.Image. class_index: kept for API compatibility - the head
    is a single wind-regression neuron (not 5 class logits), so there is
    only one score to explain regardless of the predicted class; this shows
    what drove the wind estimate (see module docstring). Returns a
    base64-encoded PNG data URL of the original image with the heatmap
    overlaid.

    Raises on failure - server.py wraps every call in try/except and
    degrades to a warning rather than a broken response.
    """
    from predict import _sessions, _V3B_WEIGHT, preprocess  # local import: onnx-heavy

    (tensor,) = preprocess(image)
    _logit, features = _sessions[0].run(["logit", "features"], {"input": tensor})
    cam = compute_cam(features[0], _V3B_WEIGHT)
    del tensor, features

    overlaid = overlay_heatmap(image, cam)
    buf = io.BytesIO()
    overlaid.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
