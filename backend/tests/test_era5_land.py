"""Land-point ERA5 tests: SST null (never zero/invented), shear/humidity kept,
nearest-ocean context, not_applicable verdict + tags, old seeds unchanged.

Uses small synthetic NetCDF files (no network, no credentials).
Run:  python backend/tests/test_era5_land.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


import era5  # noqa: E402
import logic  # noqa: E402

xr = None
try:
    import xarray as _xr
    xr = _xr
except ImportError:
    pass
check("xarray available for synthetic NetCDF", xr is not None)
if xr is None:
    print("SKIPPED (no xarray)")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="era5land_"))
LAT, LON = 20.1, 93.5


def make_point(ocean=True):
    sst = np.array([[302.0 if ocean else np.nan]])
    ds_s = xr.Dataset(
        {"sea_surface_temperature": (("latitude", "longitude"), sst)},
        coords={"latitude": [LAT], "longitude": [LON]})
    p = TMP / ("ocean_pt.nc" if ocean else "land_pt.nc")
    ds_s.to_netcdf(str(p))
    return p


def make_pressure():
    n = {"u": 10.0, "v": 2.0, "r": 80.0, "vo": 1e-5}
    data = {}
    for var, base in n.items():
        arr = np.zeros((3, 1, 1))
        arr[0, 0, 0] = base + (5.0 if var in ("u", "v") else 0.0)  # 200hPa offset
        arr[1, 0, 0] = base  # 700
        arr[2, 0, 0] = base  # 850
        data[{"u": "u_component_of_wind", "v": "v_component_of_wind",
              "r": "relative_humidity", "vo": "vorticity"}[var]] = (
            ("pressure_level", "latitude", "longitude"), arr)
    ds = xr.Dataset(data, coords={"pressure_level": [200, 700, 850],
                                  "latitude": [LAT], "longitude": [LON]})
    p = TMP / "pressure.nc"
    ds.to_netcdf(str(p))
    return p


def make_box():
    lats = np.arange(19.5, 20.75, 0.25)
    lons = np.arange(93.0, 94.25, 0.25)
    grid = np.full((len(lats), len(lons)), np.nan)
    grid[-1, -1] = 301.0  # one ocean cell at (20.5, 94.0)
    ds = xr.Dataset({"sea_surface_temperature": (("latitude", "longitude"), grid)},
                    coords={"latitude": lats, "longitude": lons})
    p = TMP / "box.nc"
    ds.to_netcdf(str(p))
    return p


P_OCEAN = make_point(True)
P_LAND = make_point(False)
P_PRES = make_pressure()
P_BOX = make_box()

# --- 1. land extract: null SST + reason, real others, no raise ----------------
land = era5._extract(P_LAND, P_PRES, LAT, LON)
check("land point no longer raises", True)
check("land SST is null (not zero, not invented)",
      land["sst_c"] is None, land["sst_c"])
check("land reason recorded", land.get("sst_note") == "over land, SST not applicable"
      and land.get("land") is True, land)
check("land shear/humidity/vorticity still real",
      land["wind_shear_kt"] == round(float(np.hypot(5, 5)) * 1.94384, 1)
      and land["humidity_pct"] == 80.0
      and abs(land["vorticity_850_s1"] - 1e-5) < 1e-12, land)

ocean = era5._extract(P_OCEAN, P_PRES, LAT, LON)
check("ocean point unchanged (land False, real SST)",
      ocean["land"] is False and ocean["sst_c"] == round(302.0 - 273.15, 2), ocean)

# --- 2. nearest ocean ----------------------------------------------------------
near = era5._nearest_ocean_sst(P_BOX, LAT, LON)
check("nearest ocean cell found with coords+distance",
      near is not None and near["lat"] == 20.5 and near["lon"] == 94.0
      and near["sst_c"] == round(301.0 - 273.15, 2) and near["distance_km"] > 0, near)
empty = TMP / "empty.nc"
xr.Dataset({"sea_surface_temperature": (("latitude", "longitude"),
                                        np.full((2, 2), np.nan))},
           coords={"latitude": [20.0, 20.25], "longitude": [93.5, 93.75]}
           ).to_netcdf(str(empty))
check("all-land box returns None (omitted, never invented)",
      era5._nearest_ocean_sst(empty, LAT, LON) is None)

# --- 3. full _fetch_live with mocked retrieve ----------------------------------
calls = []
real_retrieve = era5._retrieve


def fake_retrieve(client, dataset, request):
    calls.append(dataset)
    area = request.get("area", [])
    if dataset == "reanalysis-era5-single-levels" and len(area) == 4 and area[0] != area[2]:
        return P_BOX
    if dataset == "reanalysis-era5-single-levels":
        return P_LAND
    return P_PRES


era5._retrieve = fake_retrieve
era5._make_client = lambda: object()
try:
    full = era5._fetch_live(LAT, LON, {"area": [LAT, LON, LAT, LON]}, {"x": 1})
finally:
    era5._retrieve = real_retrieve
check("live land fetch returns shear+null SST+nearest ocean",
      full["sst_c"] is None and full["land"] is True
      and full.get("nearest_ocean_sst", {}).get("lat") == 20.5, full)
check("box follow-up requested exactly once",
      calls.count("reanalysis-era5-single-levels") == 2
      and calls.count("reanalysis-era5-pressure-levels") == 1, calls)

# --- 4. verdict / tags ----------------------------------------------------------
check("land+null SST -> not_applicable (never favorable/unfavorable)",
      logic.environment_favors_intensification(None, 5.0, 70.0, sst_land=True)
      == "not_applicable")
check("null SST without land flag stays None (not enough data)",
      logic.environment_favors_intensification(None, 5.0, 70.0) is None)
check("typed sst=0 still numeric (unfavorable), unchanged behaviour",
      logic.environment_favors_intensification(0.0, 5.0, 70.0) == "unfavorable")

tracks = logic.TrackDB(BACKEND / "data" / "track_data.csv")
resp = logic.build_response(
    class_index=2, category="Severe Cyclonic Storm", confidence=50.0,
    probs={c: 20.0 for c in logic.CLASS_ORDER},
    lat=LAT, lon=LON, wind_input=56, pressure_input=980, mode="image",
    source="model", tracks=tracks, sst=None, wind_shear=5.0, humidity=70.0,
    vorticity=1e-5, environment_source="era5", sst_land=True)
check("response verdict is not_applicable",
      resp["prediction"]["environment_favorability"] == "not_applicable",
      resp["prediction"])
check("tag stays rule-based (accurate, not MODEL)",
      resp["sources"]["environment_favorability"] == "rule-based", resp["sources"])
check("warning says over-land not-applicable",
      any("Over land" in w for w in resp["meta"]["warnings"]),
      resp["meta"]["warnings"])

# --- 5. old seeds unchanged ------------------------------------------------------
import json
ok_old = 0
for p in sorted((BACKEND / "data" / "era5" / "seed").glob("*.json")):
    d = json.loads(p.read_text())
    if not d.get("land") and d.get("sst_c") is not None and "wind_shear_kt" in d:
        ok_old += 1
check("earlier 7 ocean seeds load unchanged (real SST, no land flag)", ok_old == 7, ok_old)

# --- 6. HTTP: sst_land flag -------------------------------------------------------
import threading
import urllib.request
import server  # noqa: E402
httpd = server.make_server("127.0.0.1", 0)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_address[1]}"
body = json.dumps({"latitude": LAT, "longitude": LON, "wind": 56,
                   "wind_shear": 5.0, "humidity": 70.0, "sst_land": True}).encode()
req = urllib.request.Request(base + "/api/predict", data=body,
                             headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req).read())
check("HTTP sst_land -> not_applicable verdict",
      d["prediction"]["environment_favorability"] == "not_applicable", d["prediction"])
httpd.shutdown()

# --- 7. RSS ----------------------------------------------------------------------
try:
    import psutil
    peak = psutil.Process().memory_info().rss / 1048576
    check("RSS under 300 MB", peak < 300, f"{peak:.1f}")
except ImportError:
    print("psutil missing - RSS not measured")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All land-point tests passed.")
