"""
Fit a temperature-scaling value for PRISM-TC's classifier (backend/predict.py).

STALE as of the v3 wind-regression ensemble (model_v3b.pth + model_v4_seed1.pth)
----------------------------------------------------------------------------------
This script hard-codes the v2 architecture (5-class classifier, num_classes=5,
loads a single backend/model.pth) and temperature scaling, which rescales
softmax logits. v3's models have a single regression output and no logits to
rescale - predict.py derives its confidence a different way (see its
docstring) and does not call apply_temperature() at all. Do NOT run this
against model_v3b.pth/model_v4_seed1.pth; it would load incorrectly (wrong
num_classes) or produce a temperature.json that model_card.py already knows
to ignore for v3 (see model_card.py's get_model_card()). Left in place only
as a reference for if/when v2 is used again, or a future classification
(non-regression) variant is trained.

WHY YOU ARE RUNNING THIS, NOT THE ASSISTANT
--------------------------------------------
This script needs torch, timm and the real backend/model.pth - none of which
are available in the sandbox that wrote this code (no internet, no torch
installed, no model.pth). Run it yourself, once, on the same machine you run

    python backend/tests/smoke_test.py --real

on, since that's the machine that actually has all three.

WHAT IT DOES
------------
1. Loads backend/model.pth the same way predict.py does, but keeps the raw
   logits instead of only the final softmax probabilities.
2. Runs the model over a labelled set of images (default: backend/samples/,
   only 10 images - enough to prove the mechanism works end-to-end, but you
   should point --data at a larger held-out set if you have one). Ideally
   this should be the SAME held-out split used to measure the ~60% accuracy
   quoted in CLAUDE.md - fitting T on images the model was trained on, or on
   the same images used to report accuracy, would be circular and would
   understate how overconfident the model really is in the field.
3. Fits a single scalar temperature T with L-BFGS to minimize negative log
   likelihood on that set (Guo et al., 2017, "On Calibration of Modern
   Neural Networks" - the standard, minimally invasive way to fix
   overconfidence without touching model weights or changing which class
   wins argmax).
4. Prints Expected Calibration Error (ECE) before vs. after calibration, and
   writes backend/temperature.json. predict.py picks this file up
   automatically the next time the server starts - no further code changes
   needed. GET /api/model/transparency will also start reporting
   calibration.fitted = true.

USAGE
-----
    python backend/fit_calibration.py
    python backend/fit_calibration.py --data path/to/your_val_set.csv --images-dir path/to/images

--data must be a CSV with columns: filename,true_class
(true_class must be one of the 5 IMD category names exactly as they appear
in backend/logic.CLASS_ORDER, e.g. "Severe Cyclonic Storm").
"""

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import timm  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

import logic  # noqa: E402
from calibration import compute_ece, TEMPERATURE_PATH  # noqa: E402

MODEL_PATH = BACKEND_DIR / "model.pth"

# Must match predict.py exactly - no ImageNet normalization, see predict.py's docstring.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_raw_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=torch.device("cpu")))
    model.eval()
    return model


def collect_logits(model, df, images_dir):
    """Runs the model over every row and returns (logits, labels) as numpy arrays."""
    logits_list, labels_list = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = Path(images_dir) / row["filename"]
            image = Image.open(path).convert("RGB")
            tensor = TRANSFORM(image).unsqueeze(0)
            out = model(tensor)
            logits_list.append(out[0].numpy())
            labels_list.append(logic.CLASS_ORDER.index(row["true_class"]))
    return np.array(logits_list), np.array(labels_list)


def fit_temperature(logits, labels):
    """L-BFGS fit of a single scalar T minimizing NLL - the standard
    temperature-scaling procedure (Guo et al., 2017). Clamped to a small
    positive minimum defensively so a pathological fit can't produce a
    zero/negative temperature that would blow up apply_temperature()."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
    nll_loss = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_loss(logits_t / temperature.clamp(min=0.05), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.clamp(min=0.05).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(BACKEND_DIR / "samples" / "samples.csv"),
                        help="CSV with columns filename,true_class (default: backend/samples/samples.csv)")
    parser.add_argument("--images-dir", default=str(BACKEND_DIR / "samples"),
                        help="Folder the filenames in --data are relative to")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        print(f"ERROR: {MODEL_PATH} not found - copy the real model.pth into backend/ first.")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if len(df) < 30:
        print(f"WARNING: only {len(df)} labelled images in {args.data}. Temperature scaling "
              f"fits a single number, so this will still run, but the result will not be "
              f"very reliable on a set this small - use a larger held-out set if you have "
              f"one (see this script's docstring for why it should be held-out, not the "
              f"training set).\n")

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_raw_model()

    print(f"Running inference on {len(df)} images from {args.images_dir} ...")
    logits, labels = collect_logits(model, df, args.images_dir)

    probs_before = torch.softmax(torch.tensor(logits), dim=1).numpy()
    ece_before = compute_ece(probs_before, labels)

    print("Fitting temperature (L-BFGS, minimizing negative log likelihood) ...")
    temperature = fit_temperature(logits, labels)

    probs_after = torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
    ece_after = compute_ece(probs_after, labels)

    print(f"\nFitted temperature: T = {temperature:.3f}")
    print(f"ECE before calibration: {ece_before * 100:.1f}%")
    print(f"ECE after calibration:  {ece_after * 100:.1f}%")
    if ece_after > ece_before:
        print("NOTE: ECE got WORSE, not better. This can happen on a tiny or "
              "unrepresentative dataset (the 10-image samples/ set, for instance). Do not "
              "ship this temperature.json for a demo if this happens - re-run with a "
              "larger, genuinely held-out set instead.")

    TEMPERATURE_PATH.write_text(json.dumps({
        "temperature": round(temperature, 4),
        "fitted_on": str(args.data),
        "n_images": int(len(df)),
        "ece_before_pct": round(ece_before * 100, 2),
        "ece_after_pct": round(ece_after * 100, 2),
    }, indent=2))
    print(f"\nWrote {TEMPERATURE_PATH}")
    print("predict.py and GET /api/model/transparency will pick this up automatically "
          "the next time the server starts - no further code changes needed.")


if __name__ == "__main__":
    main()
