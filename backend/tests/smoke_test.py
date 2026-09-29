"""Smoke test for the PRISM-TC backend.  Run:  python backend/tests/smoke_test.py

Uses a FAKE predict() when torch/model.pth are missing, so it works anywhere.
Pass --real to require the real model instead.
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

if "--real" not in sys.argv:
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
    req = urllib.request.Request(BASE + path, method=method,
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


s, b, h = call("/api/health")
check("health 200", s == 200 and json.loads(b)["status"] in ("ONLINE", "DEMO MODE"))
check("CORS header", h.get("Access-Control-Allow-Origin") == "*")

s, b, _ = call("/api/samples")
samples = json.loads(b)
check("samples list", s == 200 and len(samples) == 10 and "url" in samples[0])

for i in range(3):
    s, b, _ = call("/api/demo")
    d = json.loads(b)
    check(f"demo #{i} ok", s == 200 and d["prediction"]["category"] and len(d["track"]) == 9 and len(d["forecast"]) == 9, str(d)[:200])
check("demo has past track", len(d["past_track"]) >= 1)
check("demo sources marked", d["sources"]["track"] == "simulated" and d["sources"]["category"] in ("model", "placeholder"))
check("probabilities sum ~100", abs(sum(d["probabilities"].values()) - 100) < 1.5)

s, b, _ = call("/api/predict", {"latitude": 15.5, "longitude": 85, "wind": 75, "pressure": 980})
d = json.loads(b)
check("rule-based predict (no image)", s == 200 and d["meta"]["mode"] == "rule-based" and d["prediction"]["category"] == "Very Severe Cyclonic Storm" and d["prediction"]["confidence"] is None, str(d)[:300])
check("rule-based warns", any("No image" in w for w in d["meta"]["warnings"]))

img = (BACKEND / "samples" / samples[0]["filename"]).read_bytes()
uri = "data:image/png;base64," + base64.b64encode(img).decode()
s, b, _ = call("/api/predict", {"latitude": 12, "longitude": 88, "image": uri})
d = json.loads(b)
check("upload predict", s == 200 and d["meta"]["mode"] == "image" and d["prediction"]["wind_display"], str(d)[:200])
check("track starts at requested position", d["track"][0] == [12, 88], str(d["track"][0]))

s, b, _ = call("/api/predict", {"sample": samples[1]["filename"]})
check("sample predict", s == 200)

# bad input must return 400, never crash
for name, payload in [
    ("bad latitude", {"latitude": 999}),
    ("text latitude", {"latitude": "abc"}),
    ("nan wind", {"wind": float("nan")}),
    ("bad image", {"image": "data:image/png;base64,AAAA"}),
    ("bad sample", {"sample": "../../etc/passwd"}),
    ("list body", [1, 2, 3]),
]:
    raw = json.dumps(payload).encode()
    s, b, _ = call("/api/predict", raw)
    check(f"400 on {name}", s == 400 and "error" in json.loads(b), f"{s} {b[:100]}")
s, b, _ = call("/api/predict", b"not json")
check("400 on non-JSON", s == 400)
s, b, _ = call("/api/predict", b"")
check("400 on empty body", s == 400)
s, b, _ = call("/api/nothing")
check("404 unknown api", s == 404)

# static file safety
s, b, _ = call("/")
check("index served (if present)", s in (200, 404))
for p in ["/backend/server.py", "/../backend/server.py", "/%2e%2e/backend/logic.py",
          "/backend/predict.py", "/samples/samples.csv", "/samples/../server.py",
          "/backend/data/track_data.csv", "/model.pth"]:
    s, b, _ = call(p)
    check(f"blocked {p}", s == 404, f"{s}")
s, b, _ = call("/samples/" + samples[0]["filename"])
check("sample image served", s == 200 and b[:4] == b"\x89PNG")

httpd.shutdown()
print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
