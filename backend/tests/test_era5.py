"""Offline tests for the ERA5 module and its wiring into logic.py/server.py.

Deliberately does NOT need cdsapi, xarray, or network access - this sandbox
has neither, and that is exactly the "not configured" path this module must
handle gracefully. What IS fully tested here, with zero mocking needed:

  1. logic.environment_favors_intensification's new humidity modifier (pure
     Python, fully deterministic).
  2. era5.status()/is_available() correctly reporting "not installed" when
     cdsapi/xarray are absent (true in this sandbox).
  3. era5.fetch_environment() raising ERA5NotConfigured (not crashing, not
     hanging) when the optional packages are missing.
  4. The /api/era5/status and /api/era5/fetch HTTP routes returning a clean,
     well-formed error instead of a 500 or a hung connection.
  5. /api/predict still works normally and now actually uses a supplied
     humidity value in the environment verdict (previously accepted-but-unused).

Run:  python backend/tests/test_era5.py
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------- 1. logic.py -----
import logic  # noqa: E402

check("no sst/shear -> None",
      logic.environment_favors_intensification(None, None) is None)
check("warm+low-shear -> favorable",
      logic.environment_favors_intensification(29.0, 5.0) == "favorable")
check("warm+low-shear but DRY (humidity<40) -> downgraded to neutral",
      logic.environment_favors_intensification(29.0, 5.0, humidity_pct=25.0) == "neutral")
check("warm+low-shear + moist (humidity>=40) -> stays favorable",
      logic.environment_favors_intensification(29.0, 5.0, humidity_pct=70.0) == "favorable")
check("cold water -> unfavorable regardless of humidity",
      logic.environment_favors_intensification(24.0, 5.0, humidity_pct=90.0) == "unfavorable")
check("high shear -> unfavorable regardless of humidity",
      logic.environment_favors_intensification(29.0, 25.0, humidity_pct=90.0) == "unfavorable")
check("neutral band unaffected by humidity (no upgrade)",
      logic.environment_favors_intensification(27.0, 15.0, humidity_pct=90.0) == "neutral")
check("humidity=None behaves exactly as before",
      logic.environment_favors_intensification(29.0, 5.0, None) == "favorable")

# ---------------------------------------------------------- 2/3. era5.py ----
import era5  # noqa: E402

check("cdsapi/xarray are not installed in this sandbox (expected)",
      era5.cdsapi is None or era5.xr is None)
check("is_available() is False without the optional packages",
      era5.is_available() is False)

st = era5.status()
check("status() reports packages_installed=False", st["packages_installed"] is False)
check("status() reports ready=False", st["ready"] is False)
check("status() carries a non-empty import_error message", bool(st.get("import_error")))

try:
    era5.fetch_environment(15.5, 85.0)
    check("fetch_environment raises when not configured", False, "did not raise")
except era5.ERA5NotConfigured as exc:
    check("fetch_environment raises ERA5NotConfigured cleanly", True)
    check("...with an actionable message mentioning cdsapi/xarray/credentials",
          any(w in str(exc).lower() for w in ("cdsapi", "xarray", "credential", "cdsapirc")))
except Exception as exc:  # noqa: BLE001
    check("fetch_environment raises ERA5NotConfigured (not some other exception)",
          False, f"raised {type(exc).__name__}: {exc}")

# pure-python helpers that don't need cdsapi/xarray at all
box = era5._area_box(15.5, 85.0)
check("_area_box returns [N, W, S, E] around the point",
      box[0] > 15.5 > box[2] and box[3] > 85.0 > box[1], box)
check("_round_grid snaps to the 0.25 deg ERA5 grid",
      era5._round_grid(15.37) == 15.25, era5._round_grid(15.37))

# ---------------------------------------------------- 4/5. HTTP end-to-end --
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


def call(path, data=None, method=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=None if data is None else (data if isinstance(data, bytes) else json.dumps(data).encode()),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


status_code, body = call("/api/era5/status")
check("/api/era5/status responds 200 even though ERA5 isn't installed",
      status_code == 200, (status_code, body))
check("/api/era5/status body says ready=False", body.get("ready") is False, body)

status_code, body = call("/api/era5/fetch?lat=15.5&lon=85.0")
check("/api/era5/fetch returns 503 (not configured) instead of crashing",
      status_code == 503, (status_code, body))
check("/api/era5/fetch error mentions it's not configured",
      body.get("configured") is False, body)

status_code, body = call("/api/era5/fetch")
check("/api/era5/fetch with no lat/lon -> 400, not 500",
      status_code == 400, (status_code, body))

status_code, body = call("/api/health")
check("/api/health includes era5_available=False (packages missing)",
      body.get("era5_available") is False, body)

status_code, body = call("/api/predict", data={
    "latitude": 15.5, "longitude": 85.0, "sample": "does-not-matter-if-fails-first",
})
# don't care if this exact sample exists; just confirm predict is unaffected by ERA5 import
status_code, body = call("/api/predict", data={
    "latitude": 15.5, "longitude": 85.0, "wind": 90,
    "sst": 29.0, "wind_shear": 5.0, "humidity": 25.0,
})
check("/api/predict (rule-based, no image) still returns 200 with ERA5 wired in",
      status_code == 200, (status_code, body))
if status_code == 200:
    check("dry-air humidity=25 downgrades an otherwise-favorable verdict to neutral",
          body["prediction"]["environment_favorability"] == "neutral", body["prediction"])
    check("meta reports environment_input_source='input' (typed values, not ERA5)",
          body["meta"].get("environment_input_source") == "input", body["meta"])

status_code, body = call("/api/predict", data={
    "latitude": 15.5, "longitude": 85.0, "wind": 90,
    "sst": 29.0, "wind_shear": 5.0, "humidity": 70.0,
    "vorticity": 0.00015, "environment_source": "era5",
})
check("humidity=70 keeps the verdict favorable",
      status_code == 200 and body["prediction"]["environment_favorability"] == "favorable",
      (status_code, body.get("prediction")))
check("environment_source='era5' is echoed back honestly in meta",
      body.get("meta", {}).get("environment_input_source") == "era5", body.get("meta"))
check("vorticity is surfaced as an informational warning, not silently dropped",
      any("vorticity" in w.lower() for w in body.get("meta", {}).get("warnings", [])),
      body.get("meta", {}).get("warnings"))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All ERA5 tests passed.")
