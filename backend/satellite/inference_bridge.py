"""Optional bridge from a cropped live-satellite image to the existing
predict.py model - OFF by default (see config.ENABLE_LIVE_INFERENCE) and
always clearly labelled when on.

Why off by default (model compatibility finding - the honest answer the
brief specifically asked for):

The model (EfficientNet-B0, backend/predict.py, backend/model.pth) was
fine-tuned only on the TCIR dataset: single-channel infrared "brightness
temperature-like" chips, roughly 201x201 pixels, EACH ONE CROPPED AND
CENTRED ON A SPECIFIC NAMED STORM by the TCIR dataset's own extraction
process, then resized to 224x224 and converted to a 3-channel tensor with
NO ImageNet normalisation. There is no published, verified mapping from a
Himawari (or any other) live full-disk quicklook image to that exact
per-storm-centred crop:

  1. Framing: a live regional crop (this module's output) is a wide view
     of the whole North Indian Ocean, not a tight crop centred on one
     storm's eye. Feeding a wide regional image where the model expects a
     tight storm-centred chip is a genuine train/inference mismatch, not
     just a resolution difference.
  2. Radiometry: TCIR pixel values come from calibrated brightness
     temperature, remapped to 8-bit. NICT's quicklook PNG is a rendered,
     enhanced-for-viewing composite with its own unknown/undocumented
     tone-mapping. There is no guarantee the two are on comparable scales.
  3. Never validated: the model's ~60% test accuracy and the confusion
     matrices in the model card are ALL on TCIR images. Nobody has
     measured how it behaves on a differently-sourced, differently
     processed image - not even once.

None of this means live imagery can't ever feed the model - it means doing
so validly needs (at minimum) locating a real storm's centre in the live
image, cropping tightly around it at a comparable ground resolution to
TCIR, and ideally re-validating accuracy on that pipeline before trusting
any number it produces. That is out of scope to fake here.

So: this module exists (per the brief, "if feasible implement a separate
compatible inference pipeline"), it is technically wired end-to-end, but it
ships disabled, and even when enabled it labels its output as experimental
and explicitly distinct from the validated TCIR-based prediction that
/api/predict and /api/demo already provide.
"""
from __future__ import annotations

from . import config


class InferenceUnavailable(Exception):
    pass


def run_experimental_inference(cropped_regional_image):
    """Runs the existing model on a cropped live-satellite image. Only
    called at all if config.ENABLE_LIVE_INFERENCE is True. Returns a dict
    that is deliberately shaped differently from the normal /api/predict
    response (extra `experimental` + `caveat` fields) so the frontend can
    never accidentally render it as if it were the validated prediction.
    """
    if not config.ENABLE_LIVE_INFERENCE:
        raise InferenceUnavailable(
            "Live-image inference is disabled (ENABLE_LIVE_INFERENCE=false). "
            "See backend/satellite/inference_bridge.py for why this is the "
            "default and what would need to change before enabling it.")
    try:
        from predict import predict  # the existing, unmodified model wrapper
    except Exception as exc:  # torch missing, model.pth missing, ...
        raise InferenceUnavailable(f"Model not available: {type(exc).__name__}: {exc}")

    category, confidence, probs = predict(cropped_regional_image)
    return {
        "experimental": True,
        "category": category,
        "confidence": round(float(confidence), 1),
        "probabilities": {k: round(float(v), 2) for k, v in probs.items()},
        "caveat": (
            "EXPERIMENTAL: this model was trained and validated only on "
            "TCIR storm-centred chips, not on wide regional live imagery. "
            "This result is not scientifically validated - do not treat it "
            "like the TCIR-based prediction elsewhere in this app."
        ),
    }
