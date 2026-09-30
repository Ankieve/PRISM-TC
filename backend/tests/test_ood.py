"""Automated OOD-guard tests (BUG 1 regression).

Covers, with zero torch/timm/network:
  1. All 10 bundled TCIR samples assess clean (no false rejects).
  2. 43 synthetic non-cyclone images are REJECTED by the full server path
     (false accepts = 0): photos, faces, screenshots, documents, landscapes,
     objects, cartoons, plain colours, noise, maps, colour weather charts,
     gradients and patterns.
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


def _arr(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


_BAT = np.random.default_rng(11)


def _shot_dark_ui():
    a = np.full((201, 201, 3), 30)
    a[0:25, :] = [50, 50, 60]
    for i, y in enumerate(range(35, 190, 11)):
        a[y:y + 5, 12:150 - (i % 5) * 20] = 220
    return _arr(a)


def _shot_code():
    a = np.full((201, 201, 3), 40)
    cols = [[120, 220, 120], [220, 150, 100], [150, 180, 255], [220, 220, 220]]
    for i, y in enumerate(range(10, 195, 9)):
        a[y:y + 4, 8:190] = cols[i % 4]
    return _arr(a)


def _doc_receipt():
    a = np.full((201, 201, 3), 245)
    for y in range(12, 195, 8):
        a[y:y + 2, 20:175] = 25
    a[150:180, 60:140] = [30, 30, 200]
    return _arr(a)


def _doc_news():
    a = np.full((201, 201, 3), 235)
    a[:, 95:105] = 180
    for y in range(10, 195, 7):
        a[y:y + 2, 8:90] = 40
        a[y:y + 2, 110:193] = 40
    return _arr(a)


def _land_beach():
    xx, yy = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    a = np.zeros((201, 201, 3))
    a[..., 0] = 60 + 120 * yy
    a[..., 1] = 150 + 60 * (1 - yy)
    a[..., 2] = 200 - 80 * yy
    a[140:, :] = [240, 220, 150]
    return _arr(a)


def _land_forest():
    a = _BAT.integers(20, 90, (201, 201, 3)).astype(float)
    a[..., 1] += 40
    return _arr(a)


def _land_city():
    a = np.full((201, 201, 3), 130.0)
    for x in range(0, 201, 22):
        h = _BAT.integers(60, 170)
        a[201 - h:, x:x + 15] = [70, 75, 95]
    a[0:40, :] = [100, 160, 220]
    return _arr(a)


def _land_night():
    a = np.full((201, 201, 3), 12.0)
    pts = _BAT.integers(0, 201, (120, 2))
    a[pts[:, 0], pts[:, 1]] = 255
    a[150:, 60:140] = [200, 150, 60]
    return _arr(a)


def _obj_car():
    a = np.full((201, 201, 3), 180.0)
    a[90:140, 30:170] = [180, 30, 30]
    a[100:125, 45:80] = [200, 230, 255]
    a[100:125, 120:155] = [200, 230, 255]
    return _arr(a)


def _obj_ball():
    a = np.full((201, 201, 3), 220.0)
    yy, xx = np.mgrid[0:201, 0:201]
    a[(xx - 100) ** 2 + (yy - 100) ** 2 < 60 ** 2] = [230, 120, 40]
    return _arr(a)


def _obj_bottle():
    a = np.full((201, 201, 3), 200.0)
    a[40:170, 85:115] = [40, 120, 60]
    a[20:40, 92:108] = [150, 150, 150]
    return _arr(a)


def _obj_chair():
    a = np.full((201, 201, 3), 210.0)
    a[50:130, 60:140] = [120, 80, 40]
    a[130:185, 70:80] = [80, 55, 30]
    a[130:185, 120:130] = [80, 55, 30]
    return _arr(a)


def _cartoon(face_kind):
    im = Image.new("RGB", (201, 201),
                   (255, 220, 100) if face_kind == 0 else
                   (140, 200, 255) if face_kind == 1 else (200, 240, 200))
    d = ImageDraw.Draw(im)
    if face_kind == 0:
        d.ellipse([50, 40, 150, 150], outline=(0, 0, 0), width=5, fill=(255, 200, 150))
        d.ellipse([80, 80, 95, 95], fill=(0, 0, 0))
        d.ellipse([125, 80, 140, 95], fill=(0, 0, 0))
        d.arc([80, 105, 140, 140], 20, 160, fill=(0, 0, 0), width=5)
    elif face_kind == 1:
        d.rectangle([30, 100, 170, 150], outline=(0, 0, 0), width=5, fill=(255, 60, 60))
        d.ellipse([50, 140, 80, 170], outline=(0, 0, 0), width=5, fill=(50, 50, 50))
        d.ellipse([120, 140, 150, 170], outline=(0, 0, 0), width=5, fill=(50, 50, 50))
    else:
        d.ellipse([60, 60, 140, 150], outline=(0, 0, 0), width=5, fill=(255, 180, 80))
        d.polygon([(60, 70), (75, 30), (95, 60)], outline=(0, 0, 0), width=4,
                  fill=(255, 180, 80))
        d.polygon([(105, 60), (125, 30), (140, 70)], outline=(0, 0, 0), width=4,
                  fill=(255, 180, 80))
    return im


def _noise_gray():
    g = _BAT.normal(128, 45, (201, 201))
    return _arr(np.stack([g, g, g], axis=-1))


def _noise_sp():
    a = np.full((201, 201, 3), 128.0)
    m = _BAT.random((201, 201))
    a[m < 0.15] = 0
    a[m > 0.85] = 255
    return _arr(a)


def _map_topo():
    a = np.zeros((201, 201, 3))
    yy, xx = np.mgrid[0:201, 0:201]
    a[..., 1] = 100 + 80 * np.sin(xx / 12) * np.cos(yy / 15)
    a[..., 0] = 120 + 60 * np.cos(xx / 20)
    a[..., 2] = 90
    return _arr(a)


def _map_road():
    a = np.full((201, 201, 3), 240.0)
    for i in range(6):
        y = 20 + i * 30
        a[y:y + 4, :] = [255, 200, 60]
        x = 15 + i * 32
        a[:, x:x + 3] = [255, 255, 255]
    a[80:120, 80:120] = [150, 220, 150]
    return _arr(a)


def _wx_heatmap():
    yy, xx = np.mgrid[0:201, 0:201]
    v = np.sin(xx / 18) + np.cos(yy / 14) + np.sin((xx + yy) / 25)
    v = (v - v.min()) / (v.max() - v.min())
    a = np.zeros((201, 201, 3))
    a[..., 0] = np.clip(1.5 - np.abs(4 * v - 3), 0, 1) * 255
    a[..., 1] = np.clip(1.5 - np.abs(4 * v - 2), 0, 1) * 255
    a[..., 2] = np.clip(1.5 - np.abs(4 * v - 1), 0, 1) * 255
    return _arr(a)


def _wx_fronts():
    a = np.full((201, 201, 3), 220.0)
    yy, xx = np.mgrid[0:201, 0:201]
    a[(xx - yy) ** 2 < 150] = [40, 40, 220]
    a[(xx + yy - 200) ** 2 < 150] = [220, 40, 40]
    return _arr(a)


def _wx_sat_color():
    a = _BAT.normal(150, 35, (201, 201, 3))
    a[..., 0] += 30
    a[..., 2] -= 20
    return _arr(a)


def _gradient_gray():
    a = np.tile(np.linspace(0, 255, 201), (201, 1))
    return _arr(np.stack([a, a, a], axis=-1))


def _gradient_color():
    a = np.tile(np.linspace(0, 255, 201), (201, 1))
    return _arr(np.stack([a, 255 - a, np.full_like(a, 128)], axis=-1))


def _checker():
    a = (np.indices((201, 201)).sum(axis=0) // 12 % 2 * 255).astype(float)
    return _arr(np.stack([a, a, a], axis=-1))


def _stripes():
    a = np.zeros((201, 201, 3))
    a[:, ::16] = [220, 40, 60]
    a[:, 8::16] = [40, 60, 220]
    return _arr(a)


def _qr_like():
    a = (_BAT.random((25, 25)) < 0.5).astype(float) * 255
    a = np.kron(a, np.ones((8, 8)))[:201, :201]
    return _arr(np.stack([a, a, a], axis=-1))


NON_CYCLONES += [
    ("screenshot dark UI", _shot_dark_ui()),
    ("screenshot code", _shot_code()),
    ("document receipt", _doc_receipt()),
    ("document newspaper", _doc_news()),
    ("landscape beach", _land_beach()),
    ("landscape forest", _land_forest()),
    ("landscape city", _land_city()),
    ("landscape night", _land_night()),
    ("object car", _obj_car()),
    ("object ball", _obj_ball()),
    ("object bottle", _obj_bottle()),
    ("object chair", _obj_chair()),
    ("cartoon face", _cartoon(0)),
    ("cartoon car", _cartoon(1)),
    ("cartoon cat", _cartoon(2)),
    ("plain black", Image.new("RGB", (201, 201), (0, 0, 0))),
    ("plain gray128", Image.new("RGB", (201, 201), (128, 128, 128))),
    ("plain white", Image.new("RGB", (201, 201), (255, 255, 255))),
    ("plain red", Image.new("RGB", (201, 201), (220, 40, 40))),
    ("plain green", Image.new("RGB", (201, 201), (40, 180, 70))),
    ("plain blue", Image.new("RGB", (201, 201), (50, 100, 230))),
    ("plain sky", Image.new("RGB", (201, 201), (150, 200, 235))),
    ("noise gray", _noise_gray()),
    ("noise salt-pepper", _noise_sp()),
    ("map topo", _map_topo()),
    ("map road", _map_road()),
    ("weather heatmap", _wx_heatmap()),
    ("weather fronts", _wx_fronts()),
    ("weather sat color", _wx_sat_color()),
    ("gradient gray", _gradient_gray()),
    ("gradient color", _gradient_color()),
    ("checkerboard", _checker()),
    ("stripes", _stripes()),
    ("qr-like", _qr_like()),
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
check("false-accept rate is 0/43", not n_accepted, f"accepted={n_accepted}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All OOD tests passed (10/10 pass, 43/43 rejected).")
