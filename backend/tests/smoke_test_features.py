"""Smoke test for the newer features (model transparency, OOD guard,
SST/wind-shear environment rule, backtest mode).  Run:
    python backend/tests/smoke_test_features.py

Uses a FAKE predict() (no torch/model.pth needed), same pattern as
smoke_test.py, so it runs anywhere including a sandbox with no internet.
"""
import base64
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import types
fake = types.ModuleType("predict")
order = ["Depression", "Cyclonic Storm", "Severe Cyclonic Storm",
         "Very Severe Cyclonic Storm", "Extremely Severe/Super Cyclone"]


def _fake_predict(img):
    probs = {c: 5.0 for c in order}
    probs["Severe Cyclonic Storm"] = 80.0
    return "Severe Cyclonic Storm", 80.0, probs


fake.predict = _fake_predict
sys.modules["predict"] = fake

import server  # noqa: E402

httpd = server.make_server("127.0.0.1", 0)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"
fails = []


def call(path, data=None, method=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=None if data is None else (data if isinstance(data, bytes) else json.dumps(data).encode()),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------- model transparency
s, b, _ = call("/api/model/transparency")
card = json.loads(b)
check("transparency 200", s == 200)
# v3 ensemble metrics (accuracy_exact_pct replaces v2's accuracy_pct - see
# model_card.py's docstring on why the schema changed with the model).
check("transparency has metrics", card.get("metrics", {}).get("accuracy_exact_pct") == 53.7, str(card)[:200])
check("transparency has failure modes", len(card.get("known_failure_modes", [])) >= 3)
check("transparency has 5 classes", len(card.get("classes", [])) == 5)

# ---------------------------------------------------------------------- OOD guard
s, b, samples_raw = call("/api/samples")
samples = json.loads(b)
img_bytes = (BACKEND / "samples" / samples[0]["filename"]).read_bytes()
uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()

s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": uri})
d = json.loads(b)
check("real sample not flagged OOD", s == 200 and d["meta"]["ood"]["level"] == "none", str(d["meta"].get("ood")))

# a solid black 224x224 PNG should be flagged likely_ood and REJECTED outright
# (no guessed category at all - see server.py's likely_ood early-return, added
# after a real user reported an electric bill and a random internet photo
# both coming back as confident cyclone categories).
from PIL import Image
import io as _io
blank = Image.new("RGB", (224, 224), (0, 0, 0))
buf = _io.BytesIO()
blank.save(buf, format="PNG")
blank_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": blank_uri})
d = json.loads(b)
check("solid black image is rejected outright, not classified",
      s == 200 and d.get("rejected") is True, str(d)[:300])
check("rejected response carries the ood assessment at the top level",
      d.get("ood", {}).get("level") == "likely_ood", str(d.get("ood")))
check("rejected response has no prediction/probabilities keys to accidentally trust",
      "prediction" not in d and "probabilities" not in d, str(d)[:300])
check("rejected response gives a plain-English reason", bool(d.get("reason")), str(d.get("reason")))

# A bright, near-white, low-color "document" image (mimics a real user report:
# a WhatsApp screenshot of a bill/table, brightness ~244, near-zero color, only
# moderate contrast) used to slip through as a soft "warning" and still get
# classified - the strong brightness pad was tightened (55 -> 35) specifically
# to catch this shape of image. See ood_guard.py's _STRONG_PAD comment.
import numpy as _np
doc_arr = _np.full((500, 400, 3), 250, dtype=_np.uint8)
for _y in range(0, 500, 25):
    doc_arr[_y:_y + 2, 30:370] = [60, 60, 65]
doc_img = Image.fromarray(doc_arr)
doc_buf = _io.BytesIO()
doc_img.save(doc_buf, format="PNG")
doc_uri = "data:image/png;base64," + base64.b64encode(doc_buf.getvalue()).decode()
s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": doc_uri})
d = json.loads(b)
check("bright low-color document image is rejected outright (not just warned)",
      s == 200 and d.get("rejected") is True, str(d)[:300])
check("document rejection reason cites brightness", "brightness" in str(d.get("ood", {}).get("reasons", [])).lower(),
      str(d.get("ood")))

# Compound-signal case: two independently-borderline (but not individually
# extreme) measurements should together escalate to likely_ood even though
# neither alone would cross the strong threshold.
compound_arr = _np.full((300, 300, 3), 232, dtype=_np.int16)  # brightness warning-band, not strong
compound_arr += _np.random.RandomState(0).randint(-8, 8, size=(300, 300, 1))
compound_arr = _np.clip(compound_arr, 0, 255).astype(_np.uint8)
compound_img = Image.fromarray(compound_arr)
compound_buf = _io.BytesIO()
compound_img.save(compound_buf, format="PNG")
compound_uri = "data:image/png;base64," + base64.b64encode(compound_buf.getvalue()).decode()
s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": compound_uri})
d = json.loads(b)
check("compound borderline-brightness+low-contrast image is not silently accepted",
      s == 200 and (d.get("rejected") is True), str(d)[:300])

# --------------------------------------------------- SST / wind-shear rule
s, b, _ = call("/api/predict", {
    "latitude": 12, "longitude": 88, "image": uri, "sst": 29.5, "wind_shear": 5,
})
d = json.loads(b)
check("favorable environment verdict", d["prediction"]["environment_favorability"] == "favorable", str(d["prediction"]))
check("environment source tagged rule-based", d["sources"]["environment_favorability"] == "rule-based")

s, b, _ = call("/api/predict", {
    "latitude": 12, "longitude": 88, "image": uri, "sst": 24.0, "wind_shear": 25,
})
d = json.loads(b)
check("unfavorable environment verdict", d["prediction"]["environment_favorability"] == "unfavorable", str(d["prediction"]))

s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": uri})
d = json.loads(b)
check("no sst/shear -> environment verdict is null", d["prediction"]["environment_favorability"] is None)
check("no sst/shear -> source is n/a", d["sources"]["environment_favorability"] == "n/a")

s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": uri, "sst": 999})
check("400 on out-of-range sst", s == 400)

# ------------------------------------------------------------------- backtest
s, b, _ = call("/api/backtest/storms")
storms = json.loads(b)
check("backtest storms 200", s == 200 and isinstance(storms, list) and len(storms) > 0, str(storms)[:200])
check("backtest storm has fields", all(k in storms[0] for k in ("storm_id", "start_time", "category", "lat", "lon")))

storm_id = storms[0]["storm_id"]
s, b, _ = call("/api/backtest/run", {"storm_id": storm_id}, method="POST")
d = json.loads(b)
check("backtest run 200", s == 200, str(d)[:300])
check("backtest excludes itself", d.get("analog_storm_used") != storm_id, str(d.get("analog_storm_used")))
check("backtest has track error", isinstance(d.get("track_error_km", {}).get("mean"), (int, float)))
check("backtest has wind error", isinstance(d.get("wind_error_kt", {}).get("mean"), (int, float)))

s, b, _ = call("/api/backtest/run", {"storm_id": "not-a-real-storm"}, method="POST")
check("backtest 400 on unknown storm", s == 400, str(b))

s, b, _ = call("/api/backtest/run", {}, method="POST")
check("backtest 400 on missing storm_id", s == 400)

# ------------------------------------------------------------------- Grad-CAM
# This sandbox has no torch, so gradcam.py fails to import - server.py must
# degrade to a warning, never a crash or a 500. This exercises exactly that
# defensive path (the same one a real deployment would use if torch/model.pth
# were fine but Grad-CAM itself raised for some other reason).
s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": uri, "gradcam": True})
d = json.loads(b)
check("gradcam request never crashes /api/predict", s == 200, str(d)[:300])
check("gradcam unavailable produces a warning, not a fake image",
      "gradcam_image" not in d["meta"] and any("Grad-CAM" in w for w in d["meta"]["warnings"]),
      str(d["meta"].get("warnings")))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASSED")
