"""One-time export: model_v3b.pth + model_v4_seed1.pth -> ONNX.

Each .onnx has TWO outputs:
  - "logit"    : (B, 1) wind-regression output (x100 = knots, as in predict.py)
  - "features" : (B, 1280, 7, 7) post-activation map feeding global-average-pool
                 (== what forward_head pools; Grad-CAM target)

Also writes onnx_assets.npz with each member's classifier weight/bias for the
numpy CAM path. Needs torch+timm locally (dev-only; NOT a runtime dep).
"""
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import timm

OPSET = 18


class WindWithFeatures(torch.nn.Module):
    def __init__(self, member):
        super().__init__()
        self.m = member

    def forward(self, x):
        f = self.m.forward_features(x)
        y = self.m.forward_head(f)
        return y, f


def export(src: Path, dst: Path):
    sd = torch.load(str(src), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    assert sd["classifier.weight"].shape[0] == 1, "expected wind-regression head"
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    m.load_state_dict(sd)
    m.eval()
    wrap = WindWithFeatures(m)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrap, dummy, str(dst),
        input_names=["input"], output_names=["logit", "features"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"},
                      "features": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
        dynamo=False)
    w = sd["classifier.weight"].numpy().reshape(-1)
    b = float(sd["classifier.bias"].numpy().reshape(-1)[0])
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return w, b


def main():
    assets = {}
    for name in ("model_v3b", "model_v4_seed1"):
        w, b = export(BACKEND_DIR / f"{name}.pth", BACKEND_DIR / f"{name}.onnx")
        assets[f"{name}_w"] = w
        assets[f"{name}_b"] = np.array(b)
    out = BACKEND_DIR / "onnx_assets.npz"
    np.savez(out, **assets)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
