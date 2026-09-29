"""
Manual Grad-CAM demo/sanity-check - NOT an automated test.

Run this yourself on a machine with torch, timm and the real backend
model files (model_v3b.pth + model_v4_seed1.pth):

    python backend/gradcam_demo.py
    python backend/gradcam_demo.py --sample img_1791.png

It runs the real model on one sample image, generates a Grad-CAM overlay
(backend/gradcam.py), and saves it next to this script so you can look at it
and judge for yourself whether the highlighted region makes sense (e.g. does
it land on the storm's cloud structure, or on the image border / a
compression artifact?). It does not assert anything automatically - open the
saved PNG and eyeball it, the same way test_connection.py asks you to
eyeball test_output_imd.jpg for the satellite feature.
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default=None,
                        help="Filename under backend/samples/ (default: the first one found)")
    parser.add_argument("--out", default=str(BACKEND_DIR / "gradcam_demo_output.png"))
    args = parser.parse_args()

    model_paths = [BACKEND_DIR / "model_v3b.pth", BACKEND_DIR / "model_v4_seed1.pth"]
    missing = [str(p) for p in model_paths if not p.is_file()]
    if missing:
        print(f"ERROR: missing model file(s): {missing} - copy them into backend/ first.")
        sys.exit(1)

    from PIL import Image

    import gradcam
    from predict import predict

    samples_dir = BACKEND_DIR / "samples"
    if args.sample:
        sample_path = samples_dir / args.sample
    else:
        candidates = sorted(samples_dir.glob("*.png"))
        if not candidates:
            print(f"ERROR: no .png files found in {samples_dir}")
            sys.exit(1)
        sample_path = candidates[0]

    if not sample_path.is_file():
        print(f"ERROR: {sample_path} not found.")
        sys.exit(1)

    print(f"Loading {sample_path.name} ...")
    image = Image.open(sample_path)

    print("Running the real model to get its predicted class ...")
    category, confidence, all_probs = predict(image)
    print(f"Predicted: {category} ({confidence:.1f}% confidence)")

    from logic import CLASS_ORDER
    class_index = CLASS_ORDER.index(category)

    print("Generating Grad-CAM overlay for that predicted class ...")
    data_url = gradcam.explain(image, class_index)

    import base64
    payload = data_url.split(",", 1)[1]
    Path(args.out).write_bytes(base64.b64decode(payload))
    print(f"\nWrote {args.out}")
    print("Open it and judge for yourself: does the highlighted region land on the "
          "storm's cloud structure, or somewhere that doesn't make meteorological "
          "sense (image border, a compression artifact, empty ocean)? Grad-CAM is a "
          "coarse debugging aid, not proof of correct reasoning - see gradcam.py's "
          "docstring.")


if __name__ == "__main__":
    main()
