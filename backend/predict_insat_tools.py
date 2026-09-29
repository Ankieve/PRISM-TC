"""
predict_insat_tools.py - SIH26070 cyclone intensity toolkit (EfficientNet-B0, timm)

NOT used by the live server. backend/predict.py (a separate, smaller file) is
what server.py and gradcam.py actually import, and it only ever receives a
plain uploaded/sample image - never a raw MOSDAC .h5 file. This module is
kept as a standalone toolkit/CLI for offline work: scoring a folder of INSAT
.h5 files against IBTrACS, calibrating the optional OOD feature check, or
experimenting with the ensemble - none of which the server's request path
needs today. Wiring live MOSDAC .h5 prediction into the server would be a
new feature, not part of this integration.

Supports two kinds of checkpoint (detected automatically in load_model):
  - WIND model  (1 output) - predicts max wind in knots, converted to IMD class;
                ImageNet normalization; 4-rotation test-time averaging.
                RECOMMENDED: ensemble of model_v3b.pth + model_v4_seed1.pth
                Storm-wise test (12 unseen storms, 520 images): 53.7% exact,
                93.1% within one class, macro-F1 0.487, MAE 10.2 kt, train-test
                gap 6 pts (vs the old v2 model's 42.4-point gap on the same
                test). See backend/model_card.py for the full comparison,
                including where v2 actually still wins (live Biparjoy exact
                accuracy) - it is not a strict win on every number.
  - CLASS model (model.pth = v2, 5 outputs) - predicts the class directly;
                no normalization, no rotation averaging.
                Storm-wise test: 54.8% exact, 87.9% within one class.

Two input paths:
  1. predict_image()  - TCIR-style images (resize 224; ImageNet-normalized for
                        a wind model, NOT normalized for the old class model -
                        see _tf() below, this is chosen automatically)
  2. predict_insat()  - INSAT-3DR L1C ASIA_MER .h5 files from MOSDAC (same
                        automatic normalization choice)

INSAT-3DR preprocessing (the crop/brightness-temperature step, independent of
the normalization choice above) was matched to the TCIR training images and
validated on Cyclone Biparjoy (Jun 2023) - 36 real frames, IBTrACS ground
truth:
  - storm-centred crop of 374 x 374 px on the ASIA_MER grid (~14 deg, like TCIR)
  - TIR1 brightness temperature clipped to 180-310 K, scaled so COLD = BLACK
  - resized to 224
  - v2 (no normalization): 83.3% exact / 94.4% within-one-class
  - ensemble (ImageNet-normalized): 75.0% exact / 100% within-one-class / 4.9 kt MAE
  (see backend/model_card.py's "live_biparjoy_validation" for the honest
  read of these two numbers side by side)

"Not a cyclone" detection below (check_image / check_insat_crop) is a
SEPARATE heuristic from backend/ood_guard.py, which is what actually gates
images in the live server today (tuned there from real TCIR samples and a
real false-positive case - see its own docstring). The checks here exist for
this standalone toolkit's own use (e.g. scoring a batch of .h5 files with no
ood_guard involved) and are deliberately NOT also run inside backend/predict.py's
predict(), to avoid two differently-tuned guards disagreeing on the same
image. If INSAT/.h5 prediction is ever wired into the live server, decide
then which of the two guards (or both) should run for that path.

Every prediction in this module goes through:
  a. Image checks   - rejects colour photos and blank/flat images
                      (satellite IR images are grayscale with real structure)
  b. INSAT checks   - rejects crops with no deep convective cloud (nothing colder
                      than ~240 K) or where the storm centre is outside the scan
  c. Feature check  - (optional, after calibrate_ood) rejects images whose deep
                      features look unlike the cyclone images used for calibration
  d. Confidence     - low top-class probability is flagged as 'low_confidence'
                      (flagged, not rejected)

Usage (command line):
  python predict.py --model model.pth --image storm.png [--ood ood_ref.npz]
  python predict.py --model model.pth --h5 3RIMG_..._L1C_ASIA_MER_V01R00.h5 --lat 23.3 --lon 68.6
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms as T

CLASSES = [
    # NOTE: kept as exactly 'Depression' (not 'Depression/Deep Depression') to
    # match backend/logic.py's CLASS_ORDER and backend/predict.py - the two
    # need never diverge, since a mismatch there is what would have broken
    # every prediction request had this file's labels been used directly by
    # the server (see backend/predict.py's own docstring).
    'Depression',
    'Cyclonic Storm',
    'Severe Cyclonic Storm',
    'Very Severe Cyclonic Storm',
    'Extremely Severe/Super Cyclone',
]

# IMD grade -> class index (for evaluation against IBTrACS NEWDELHI_GRADE)
IMD_TO_CLASS = {'D': 0, 'DD': 0, 'CS': 1, 'SCS': 2, 'VSCS': 3, 'ESCS': 4, 'SuCS': 4}

# INSAT-3DR settings (validated on Biparjoy - change only with re-testing)
HALF_PX = 187            # half crop size in pixels -> 374 px ~ 14 deg on ASIA_MER
BT_MIN, BT_MAX = 180.0, 310.0

# "Not a cyclone" thresholds (heuristics - tune if they reject real storms)
MAX_COLOUR_SPREAD = 8.0   # mean |R-G|+|G-B| in 0-255 units; grayscale images are ~0
MIN_GRAY_STD = 5.0        # flat/blank images have almost no variation
COLD_BT_K = 240.0         # deep convective cloud tops are colder than this
MIN_COLD_FRACTION = 0.01  # at least 1% of the crop must be that cold
MIN_VALID_FRACTION = 0.5  # at least half the crop must be inside the satellite scan
LOW_CONFIDENCE = 0.5      # top-class probability below this is flagged (class model)
MAX_TTA_SPREAD_KT = 10.0  # wind model: rotations disagreeing by more than this is flagged

WIND_THRESHOLDS_KT = [34, 48, 64, 90]   # IMD class boundaries

# Class model (v2): NO Normalize.  Wind model (v3b): ImageNet Normalize.
_TF = T.Compose([T.Resize((224, 224)), T.ToTensor()])
_TF_NORM = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def _tf(model):
    return _TF_NORM if getattr(model, 'kind', 'class') == 'wind' else _TF


def wind_to_class(kt):
    return int(np.digitize(kt, WIND_THRESHOLDS_KT))


# ---------------------------------------------------------------- model
class _WindEnsemble(torch.nn.Module):
    """Averages the wind output of several wind models."""
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)

    # feature check (calibrate_ood / _embed) uses the first member
    def forward_features(self, x):
        return self.members[0].forward_features(x)

    def forward_head(self, x, pre_logits=False):
        return self.members[0].forward_head(x, pre_logits=pre_logits)


def _load_one(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    n_out = sd['classifier.weight'].shape[0]
    if n_out == 1:
        kind = 'wind'
    elif n_out == len(CLASSES):
        kind = 'class'
    else:
        raise ValueError(f'{path}: {n_out} outputs; expected 1 (wind) or {len(CLASSES)} (class)')
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=n_out)
    model.load_state_dict(sd)
    return model, kind


def load_model(path='model.pth', device=None):
    """Load a checkpoint, or several wind checkpoints as an ensemble.
    path: one file, a list of files, or a comma-separated string, e.g.
          'model_v3b.pth,model_v4_seed1.pth'   (recommended ensemble)
    Returns (model, device)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    paths = path.split(',') if isinstance(path, str) else list(path)
    loaded = [_load_one(p.strip()) for p in paths]
    kinds = {k for _, k in loaded}
    if len(loaded) == 1:
        model, kind = loaded[0]
    elif kinds == {'wind'}:
        model, kind = _WindEnsemble([m for m, _ in loaded]), 'wind'
    else:
        raise ValueError('An ensemble can only combine wind models (1 output each)')
    model.eval().to(device)
    model.kind = kind
    return model, device


def _rejected(reason, **extra):
    return {'is_cyclone': False, 'reason': reason, 'class': None, 'index': None,
            'wind_kt': None, 'confidence': None, 'probs': None, 'low_confidence': None, **extra}


@torch.no_grad()
def _embed(model, img, device):
    """Unit-length feature vector from the layer before the classifier."""
    x = _tf(model)(img.convert('RGB')).unsqueeze(0).to(device)
    f = model.forward_head(model.forward_features(x), pre_logits=True)
    return F.normalize(f, dim=1)[0].cpu().numpy()


@torch.no_grad()
def _run(model, img, device, ood_ref=None):
    # c. feature-distance check
    if ood_ref is not None:
        sim = float(_embed(model, img, device) @ ood_ref['centroid'])
        if sim < ood_ref['threshold']:
            return _rejected(f'image features unlike cyclone images '
                             f'(similarity {sim:.3f} < {ood_ref["threshold"]:.3f})',
                             similarity=round(sim, 4))
    x = _tf(model)(img.convert('RGB')).unsqueeze(0).to(device)

    if model.kind == 'wind':
        # average over 4 rotations (cyclones look alike at any rotation)
        winds = np.array([float(model(torch.rot90(x, k, (2, 3)))[0, 0]) * 100 for k in range(4)])
        kt = float(winds.mean())
        i = wind_to_class(kt)
        spread = float(winds.max() - winds.min())
        return {
            'is_cyclone': True,
            'reason': 'ok',
            'class': CLASSES[i],
            'index': i,
            'wind_kt': round(kt, 1),
            'tta_spread_kt': round(spread, 1),
            'confidence': None,
            'probs': None,
            'low_confidence': bool(spread > MAX_TTA_SPREAD_KT),   # d. flag only
        }

    p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    i = int(p.argmax())
    return {
        'is_cyclone': True,
        'reason': 'ok',
        'class': CLASSES[i],
        'index': i,
        'wind_kt': None,
        'confidence': round(float(p[i]), 4),
        'probs': {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
        'low_confidence': bool(p[i] < LOW_CONFIDENCE),   # d. flag only
    }


# ---------------------------------------------------------------- a. image checks
def check_image(img):
    """Returns (ok, reason). Rejects colour photos and blank images."""
    arr = np.asarray(img.convert('RGB'), dtype=np.float32)
    spread = (np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean())
    if spread > MAX_COLOUR_SPREAD:
        return False, f'colour image (spread {spread:.1f}) - satellite IR images are grayscale'
    if arr.mean(-1).std() < MIN_GRAY_STD:
        return False, 'blank or flat image - no cloud structure'
    return True, 'ok'


# ---------------------------------------------------------------- optional c. calibration
def calibrate_ood(model, images, device='cpu', out_path='ood_ref.npz', percentile=1.0, margin=0.05):
    """Learn what 'cyclone-like' features look like from known cyclone images.
    images: list of PIL images or file paths - ideally a few hundred TCIR training images
            covering all classes. Calibrating on one storm only makes the check too strict.
    threshold = (percentile of calibration similarities) - margin"""
    imgs = [Image.open(i) if isinstance(i, (str, os.PathLike)) else i for i in images]
    E = np.stack([_embed(model, im, device) for im in imgs])
    centroid = E.mean(0)
    centroid /= np.linalg.norm(centroid)
    sims = E @ centroid
    threshold = float(np.percentile(sims, percentile) - margin)
    np.savez(out_path, centroid=centroid, threshold=threshold)
    print(f'Calibrated on {len(imgs)} images | similarity min {sims.min():.3f}, '
          f'median {np.median(sims):.3f} | threshold {threshold:.3f} -> {out_path}')
    return {'centroid': centroid, 'threshold': threshold}


def load_ood(path='ood_ref.npz'):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {'centroid': d['centroid'], 'threshold': float(d['threshold'])}


# ---------------------------------------------------------------- path 1: TCIR-style images
def predict_image(model, image, device='cpu', ood_ref=None):
    """image: file path or PIL.Image (TCIR-style, already storm-centred)."""
    img = Image.open(image) if isinstance(image, (str, os.PathLike)) else image
    ok, reason = check_image(img)
    if not ok:
        return _rejected(reason)
    return _run(model, img, device, ood_ref)


# ---------------------------------------------------------------- path 2: INSAT-3DR .h5
def _attr(f, *words):
    for k, v in f.attrs.items():
        if all(w in k.lower().replace('_', ' ') for w in words):
            return float(np.ravel(v)[0])
    raise KeyError(f'File attribute containing {words} not found')


def _merc(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _south_up(f):
    """True if row 0 is the southern edge (not the case for standard ASIA_MER)."""
    if 'Y' in f:
        y = f['Y'][()]
        return y[0] < y[-1]
    return False


def _locate(f, lat, lon, shape):
    """Storm centre (lat, lon) -> (row, col) on the Mercator ASIA_MER grid."""
    H, W = shape
    top, bot = _attr(f, 'upper', 'lat'), _attr(f, 'lower', 'lat')
    left, right = _attr(f, 'left', 'lon'), _attr(f, 'right', 'lon')
    r = (_merc(top) - _merc(lat)) / (_merc(top) - _merc(bot)) * (H - 1)
    c = (lon - left) / (right - left) * (W - 1)
    if _south_up(f):
        r = (H - 1) - r
    return int(round(r)), int(round(c))


def insat_time(h5_path):
    """Scan time from a filename like 3RIMG_15JUN2023_1907_L1C_ASIA_MER_V01R00.h5"""
    name = os.path.basename(h5_path)
    return datetime.strptime('_'.join(name.split('_')[1:3]), '%d%b%Y_%H%M')


def insat_crop_bt(h5_path, center_lat, center_lon, half_px=HALF_PX):
    """Storm-centred brightness-temperature crop (K). Outside the scan = NaN."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        counts = f['IMG_TIR1'][()].squeeze()
        lut = f['IMG_TIR1_TEMP'][()]
        r, c = _locate(f, center_lat, center_lon, counts.shape)
    bt = lut[np.clip(counts, 0, len(lut) - 1)].astype('float32')
    bt[bt > 400] = np.nan                                   # fill values
    return np.pad(bt, half_px, constant_values=np.nan)[r:r + 2 * half_px, c:c + 2 * half_px]


def check_insat_crop(crop):
    """b. Returns (ok, reason, stats). Needs deep convective cloud near the centre."""
    valid = np.isfinite(crop)
    frac_valid = valid.mean()
    stats = {'valid_fraction': round(float(frac_valid), 3)}
    if frac_valid < MIN_VALID_FRACTION:
        return False, 'storm centre is outside the satellite scan area', stats
    vals = crop[valid]
    cold = float((vals < COLD_BT_K).mean())
    stats.update(min_bt_k=round(float(vals.min()), 1), cold_fraction=round(cold, 3))
    if cold < MIN_COLD_FRACTION:
        return False, (f'no deep convective cloud (only {cold:.1%} of pixels below '
                       f'{COLD_BT_K:.0f} K) - not a cyclone'), stats
    return True, 'ok', stats


def insat_to_image(h5_path, center_lat, center_lon, half_px=HALF_PX):
    """Build the TCIR-matched input image from an INSAT-3DR L1C ASIA_MER file."""
    crop = insat_crop_bt(h5_path, center_lat, center_lon, half_px)
    return _crop_to_image(crop)


def _crop_to_image(crop):
    crop = np.nan_to_num(crop, nan=BT_MAX)
    img = (np.clip((crop - BT_MIN) / (BT_MAX - BT_MIN), 0, 1) * 255).astype('uint8')  # cold = black
    return Image.fromarray(img).convert('RGB')


def predict_insat(model, h5_path, center_lat, center_lon, device='cpu', ood_ref=None):
    """Classify one INSAT-3DR file given the storm centre (e.g. from IBTrACS)."""
    crop = insat_crop_bt(h5_path, center_lat, center_lon)
    ok, reason, stats = check_insat_crop(crop)
    if not ok:
        out = _rejected(reason, **stats)
    else:
        out = _run(model, _crop_to_image(crop), device, ood_ref)
        out.update(stats)
    out['time'] = insat_time(h5_path).isoformat()
    return out


# ---------------------------------------------------------------- optional: centre from IBTrACS
def ibtracs_center(csv_path, storm_name, season, when):
    """Interpolated (lat, lon, wind_kt) for a storm at datetime `when`.
    csv_path: IBTrACS CSV, e.g. ibtracs.NI.list.v04r01.csv"""
    import pandas as pd
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    b = df[(df.NAME == storm_name.upper()) & (df.SEASON.astype(str) == str(season))].copy()
    if b.empty:
        raise ValueError(f'{storm_name} {season} not found in {csv_path}')
    b['time'] = pd.to_datetime(b.ISO_TIME)
    for col in ['LAT', 'LON', 'NEWDELHI_WIND']:
        b[col] = pd.to_numeric(b[col], errors='coerce')
    b = b.dropna(subset=['LAT', 'LON']).sort_values('time')
    t = b.time.values.astype('datetime64[s]').astype(float)
    x = np.datetime64(when, 's').astype(float)
    wind = np.interp(x, t, b.NEWDELHI_WIND.ffill().bfill()) if b.NEWDELHI_WIND.notna().any() else np.nan
    return float(np.interp(x, t, b.LAT)), float(np.interp(x, t, b.LON)), float(wind)


# ---------------------------------------------------------------- CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SIH26070 cyclone intensity prediction')
    ap.add_argument('--model', default='model_v3b.pth,model_v4_seed1.pth', help='one checkpoint, or comma-separated wind checkpoints (default: the recommended ensemble)')
    ap.add_argument('--ood', default='ood_ref.npz', help='feature-check file from calibrate_ood (optional)')
    ap.add_argument('--image', help='TCIR-style image file')
    ap.add_argument('--h5', help='INSAT-3DR L1C ASIA_MER .h5 file')
    ap.add_argument('--lat', type=float, help='storm centre latitude (with --h5)')
    ap.add_argument('--lon', type=float, help='storm centre longitude (with --h5)')
    a = ap.parse_args()

    model, dev = load_model(a.model)
    ood = load_ood(a.ood)
    if a.image:
        res = predict_image(model, a.image, dev, ood)
    elif a.h5:
        if a.lat is None or a.lon is None:
            ap.error('--h5 needs --lat and --lon (storm centre)')
        res = predict_insat(model, a.h5, a.lat, a.lon, dev, ood)
    else:
        ap.error('give --image or --h5')

    if not res['is_cyclone']:
        print(f"NOT A CYCLONE IMAGE: {res['reason']}")
    elif res['wind_kt'] is not None:
        flag = '  [low confidence]' if res['low_confidence'] else ''
        print(f"{res['class']}  (estimated max wind {res['wind_kt']:.0f} kt, "
              f"rotation spread {res['tta_spread_kt']:.0f} kt){flag}")
    else:
        flag = '  [low confidence]' if res['low_confidence'] else ''
        print(f"{res['class']}  (confidence {res['confidence']:.2f}){flag}")
        for c, p in res['probs'].items():
            print(f'  {c:32s} {p:.3f}')


