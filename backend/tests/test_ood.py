"""Automated OOD-guard tests (BUG 1 regression).

Covers, with zero torch/timm/network:
  1. All 10 bundled TCIR samples assess clean (no false rejects).
  2. 8 realistic non-cyclone images are REJECTED by the full server path
     (false accepts = 0): photo, face, screenshot, plain colour, text page,
     ordinary map, earth-from-space, noise.
  3. A rejection carries no class/wind/track/risk/Grad-CAM and names TCIR.

Run:  python backend/tests/test_ood.py
"""
import base64
import csv
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


RNG = np.random.default_rng(7)
S = (201, 201)


def _photo():
    x = np.zeros((201, 201, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    x[..., 0] = (xx * 200 + 30).astype(np.uint8)
    x[..., 1] = (yy * 150 + 40).astype(np.uint8)
    x[..., 2] = 90
    x[120:170, 30:120] = [60, 140, 60]
    x[100:125, 40:160] = [200, 60, 50]
    return Image.fromarray(x)


def _face():
    img = Image.new("RGB", S, (120, 140, 180))
    d = ImageDraw.Draw(img)
    d.ellipse([55, 30, 145, 150], fill=(210, 170, 140))
    d.ellipse([80, 70, 100, 90], fill=(40, 30, 25))
    d.ellipse([120, 70, 140, 90], fill=(40, 30, 25))
    d.arc([85, 100, 135, 135], 10, 170, fill=(150, 70, 70), width=4)
    d.rectangle([40, 10, 160, 35], fill=(70, 50, 35))
    return img


def _screenshot():
    img = Image.new("RGB", S, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 201, 22], fill=(40, 90, 180))
    for i, y in enumerate(range(35, 190, 12)):
        w = 150 - (i % 4) * 25
        d.rectangle([12, y, 12 + w, y + 6], fill=(30, 30, 30))
    d.rectangle([12, 150, 80, 175], fill=(60, 160, 80))
    return img


def _plain():
    return Image.new("RGB", S, (100, 150, 200))


def _textpage():
    img = Image.new("RGB", S, (250, 250, 248))
    d = ImageDraw.Draw(img)
    for y in range(15, 195, 9):
        d.rectangle([15, y, 186, y + 3], fill=(20, 20, 20))
    return img


def _maplike():
    x = np.zeros((201, 201, 3), dtype=np.uint8)
    x[:] = [235, 225, 190]
    x[0:60, :] = [140, 190, 120]
    x[150:, 100:] = [90, 150, 220]
    img = Image.fromarray(x)
    d = ImageDraw.Draw(img)
    for i in range(5):
        d.line([(0, 30 + i * 30), (201, 45 + i * 25)], fill=(200, 60, 60), width=2)
    return img


def _earth():
    x = np.zeros((201, 201, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:201, 0:201]
    disc = (xx - 100) ** 2 + (yy - 100) ** 2 < 70 ** 2
    x[disc] = [40, 90, 200]
    clouds = RNG.random((201, 201)) < 0.25
    x[disc & clouds] = [230, 230, 230]
    return Image.fromarray(x)


def _noise():
    return Image.fromarray(RNG.integers(0, 256, (201, 201, 3), dtype=np.uint8))


NON_CYCLONES = [
    ("random photo", _photo()),
    ("face", _face()),
    ("screenshot", _screenshot()),
    ("plain colour", _plain()),
    ("text page", _textpage()),
    ("ordinary map", _maplike()),
    ("earth from space", _earth()),
    ("noise", _noise()),
]

import ood_guard  # noqa: E402
import server  # noqa: E402

server.init_state()

# --- 1. no false rejects on real samples -------------------------------------
n_flagged = 0
for row in csv.DictReader(open(BACKEND / "samples" / "samples.csv")):
    level = ood_guard.assess(Image.open(BACKEND / "samples" / row["filename"]))["level"]
    if level == "likely_ood":
        n_flagged += 1
check("all 10 TCIR samples pass the guard (0 false rejects)", n_flagged == 0,
      f"flagged={n_flagged}")

# --- 2/3. every non-cyclone is rejected with a clean shape -------------------
n_accepted = []
for name, img in NON_CYCLONES:
    buf = BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    d = server.run_prediction({"latitude": 15.0, "longitude": 85.0, "image": uri})
    if not d.get("rejected"):
        n_accepted.append(name)
        check(f"rejects {name}", False,
              f"classified as {(d.get('prediction') or {}).get('category')}")
        continue
    check(f"rejects {name}", True)
    check(f"{name}: no prediction/track/probs keys",
          "prediction" not in d and "track" not in d and "probabilities" not in d, str(sorted(d)))
    check(f"{name}: no gradcam in meta",
          "gradcam_image" not in (d.get("meta") or {}))
    check(f"{name}: reason names TCIR", "TCIR" in str(d.get("reason", "")), str(d.get("reason")))
check("false-accept rate is 0/8", not n_accepted, f"accepted={n_accepted}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All OOD tests passed (10/10 pass, 8/8 rejected).")
