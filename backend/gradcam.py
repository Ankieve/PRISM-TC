"""
Grad-CAM overlay for PRISM-TC's EfficientNet-B0 classifier.

Reference: Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization".

What this actually shows (read before demoing it)
---------------------------------------------------
A coarse heatmap - upsampled from a 7x7 feature grid, since EfficientNet-B0
downsamples a 224x224 input by 32x - showing which regions most influenced
the score for one class. It is a debugging/sanity-check visualization, not
proof the model is "looking at the storm for the right meteorological
reasons": it can just as easily highlight image borders, compression
artifacts, or the chip's edge as the storm's actual eye or convective core.
Present it as "here is a coarse heatmap of what the network weighted most,"
not as validated scientific attribution.

This module cannot be run or verified in the sandbox that wrote it - no
torch, no model.pth, no internet to fetch either. It is imported lazily and
defensively by server.py (see the try/except around `import gradcam` there,
the same pattern already used for the optional satellite package) so a
problem here can never take down /api/predict or the rest of the app.
Run backend/gradcam_demo.py on a machine with the real model.pth and torch
installed (the same one you run `smoke_test.py --real` on) to generate a
real example and eyeball it before relying on it in front of judges.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _find_target_layer(model):
    """timm's EfficientNet-B0: conv_head is the last conv layer before
    global pooling - the conventional Grad-CAM target for this architecture.
    Falls back to the last MBConv block if a different timm version doesn't
    expose conv_head, so a timm version bump doesn't silently break this."""
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise AttributeError(
        "Could not find a target conv layer for Grad-CAM on this model "
        "(expected timm's EfficientNet-B0 with a 'conv_head' or 'blocks' attribute).")


class GradCAM:
    """One-shot Grad-CAM helper. Registers hooks on construction; always
    call remove() afterwards (or use as a context manager) so hooks don't
    accumulate on the shared model across repeated requests."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or _find_target_layer(model)
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        self.model.zero_grad(set_to_none=True)

    def generate(self, input_tensor, class_index):
        """input_tensor: (1, C, H, W), already preprocessed exactly like
        predict.py's _predict_transform (as of the v3 wind-regression
        ensemble, that means WITH ImageNet normalization - it no longer
        matches the old no-normalization v2 pipeline; see predict.py's
        docstring). class_index: which class's score to explain - pass in
        the model's own predicted class (computed by predict.py) rather than
        recomputing argmax here, so the heatmap always matches the class
        already shown to the user.

        predict.py's _model is a single wind-regression member (one output
        neuron, not 5 class logits) - see its own docstring on why an
        ensemble member rather than the full ensemble is used here. There is
        only one score to explain regardless of which class was predicted,
        so class_index is ignored whenever the model has fewer outputs than
        it - this heatmap then shows what drove the wind estimate, not a
        specific class's logit, which is the most honest thing available for
        a regression head."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        target = class_index if class_index < output.shape[1] else 0
        score = output[0, target]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,) global-average-pooled gradients

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.numpy()  # HxW, values in [0, 1]


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
    """original_image: PIL.Image (any size/mode). cam: 2D array in [0, 1]
    from GradCAM.generate(). Returns a PIL.Image the same size as
    original_image with the heatmap alpha-blended on top."""
    original = original_image.convert("RGB")
    w, h = original.size
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heatmap = _jet_colormap(cam_resized)
    base = np.asarray(original, dtype=np.float32)
    blended = base * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def explain(image, class_index):
    """image: PIL.Image. class_index: the class to explain (pass predict.py's
    own predicted class index) - ignored in favour of the single output
    neuron when predict.py's model is a wind-regression head, see
    GradCAM.generate()'s docstring. Returns a base64-encoded PNG data URL of
    the original image with the Grad-CAM heatmap overlaid.

    Raises on failure - this is intentional. server.py wraps every call to
    this function in try/except and degrades to a warning message rather
    than a broken response, the same defensive pattern already used for the
    optional satellite feature - a Grad-CAM bug must never take down the
    core /api/predict endpoint.
    """
    from predict import _model, _predict_transform  # local import: torch-heavy, only needed here

    tensor = _predict_transform(image.convert("RGB")).unsqueeze(0)

    with GradCAM(_model) as cam_tool:
        cam = cam_tool.generate(tensor, class_index)

    overlaid = overlay_heatmap(image, cam)
    buf = io.BytesIO()
    overlaid.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
