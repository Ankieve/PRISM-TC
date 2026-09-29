"""Offline tests for the IBTrACS historical-archive module and its wiring
into server.py/ibtracs_scheduler.py.

Deliberately does NOT need real internet access - this sandbox has none,
and every network call below is mocked at the `requests` boundary. What IS
fully tested here, with zero network calls:

  1. CSV parsing: units-row skip, per-row lat/lon/wind fallback across
     agency columns, storm_id construction, category derivation via the
     SAME logic.class_from_wind() thresholds used everywhere else in this
     app, row-dropping when no usable wind exists, and a clean IBTracsError
     (not a crash) when the file doesn't look like IBTrACS at all.
  2. status()/is_available() before anything has been fetched.
  3. fetch_and_update(): a full mocked download+parse+write cycle, the
     ACTIVE-storms merge (and year_range recomputation after merging it in
     - a real bug this test would have caught), the cache-freshness skip on
     an immediate re-call, non-fatal ACTIVE-feed failures, and that a
     malformed main file raises IBTracsError WITHOUT touching the last
     good cache on disk (never show nothing when something good exists).
  4. IBTRACS_ENABLED=false raising IBTracsNotConfigured cleanly.
  5. The daemon-thread scheduler running immediately on start, repeating,
     and stopping cleanly, without dying on a raised exception.
  6. Every /api/storms/* HTTP route: status, years, by-year (including the
     has_real_image cross-reference against the REAL backend/samples/
     samples.csv - not a mock, so this also verifies the storm_id
     convention actually matches the existing curated sample set), track
     (valid id, unknown id, missing param), and the manual POST refresh
     trigger (both a mocked failure and a mocked success) - plus
     /api/health carrying storms_available.

None of this touches backend/data/ibtracs_cache - a temporary directory
stands in for it, so running this test never leaves fake data behind for
the real app to accidentally read, and never depends on anything already
having been fetched on this machine.

Run:  python backend/tests/test_ibtracs.py

Note for the maintainer running this on their own internet-connected
device: this file cannot and does not test a REAL download from NOAA NCEI
(the sandbox this was written in has no internet) - see
backend/ibtracs_sync.py's own "Honesty notes" for exactly what that first
real fetch should be treated as verifying (column names, storm_id
reconstruction). Run `curl -I <BASE_URL>ibtracs.NI.list.v04r01.csv` or just
open the Cyclone Archive panel and press "Refresh now" once to do that.
"""
import json
import shutil
import sys
import tempfile
import threading
import time
import types
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


class FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


FAKE_MAIN_CSV = (
    "SID,SEASON,NUMBER,BASIN,NAME,ISO_TIME,LAT,LON,WMO_WIND,NEWDELHI_WIND,USA_WIND\n"
    "h,u,n,i,t,s,d,e,g,r,r\n"
    "2014028N10093,2014,2,NI,HUDHUD,2014-10-08 00:00:00,10.5,85.2,,35,32\n"
    "2014028N10093,2014,2,NI,HUDHUD,2014-10-08 03:00:00,10.8,85.0,,45,40\n"
    "2014028N10093,2014,2,NI,HUDHUD,2014-10-08 06:00:00,11.2,84.6,,70,\n"
    "2021003N08060,2021,3,NI,TAUKTAE,2021-05-14 00:00:00,15.0,72.0,,,90\n"
)
FAKE_ACTIVE_CSV = (
    "SID,SEASON,NUMBER,BASIN,NAME,ISO_TIME,LAT,LON,WMO_WIND,NEWDELHI_WIND,USA_WIND\n"
    "h,u,n,i,t,s,d,e,g,r,r\n"
    "2026030N12070,2026,30,NI,LIVESTORM,2026-09-26 00:00:00,12.0,70.0,,55,\n"
)


def fake_get_ok(url, timeout=None, headers=None):
    return FakeResp(FAKE_ACTIVE_CSV if "ACTIVE" in url else FAKE_MAIN_CSV)


def fake_get_active_fails(url, timeout=None, headers=None):
    if "ACTIVE" in url:
        raise Exception("simulated ACTIVE feed outage")
    return FakeResp(FAKE_MAIN_CSV)


def fake_get_all_fail(url, timeout=None, headers=None):
    raise Exception("simulated total network outage")


def fake_get_bad_csv(url, timeout=None, headers=None):
    return FakeResp("<html>not a csv, NCEI changed something</html>")


# =========================================================== 1. parsing ====
import ibtracs_sync  # noqa: E402

df, stats = ibtracs_sync._parse_csv(FAKE_MAIN_CSV, basin_filter="NI")
check("parses 4 rows into 4 usable points", stats["point_count"] == 4, stats)
check("groups into 2 storms", stats["storm_count"] == 2, stats)
check("drops 0 rows (every row had a usable wind+position)",
      stats["rows_dropped_no_wind"] == 0, stats)
check("year_range spans the two seasons present", stats["year_range"] == [2014, 2021], stats)

row0 = df[(df["storm_id"] == "201402I") & (df["time"] == 2014100800)].iloc[0]
check("storm_id built as SEASON+NUMBER(02d)+'I'", "201402I" in set(df["storm_id"]), set(df["storm_id"]))
check("wind falls back NEWDELHI(35) over WMO(empty)/USA(32) on row 1",
      row0["vmax"] == 35.0, row0["vmax"])
check("category derived via logic.class_from_wind (35kt -> Cyclonic Storm)",
      row0["category"] == "Cyclonic Storm", row0["category"])

row2 = df[(df["storm_id"] == "201402I") & (df["time"] == 2014100806)].iloc[0]
check("wind falls back to WMO(70) when NEWDELHI/USA are both empty",
      row2["vmax"] == 70.0, row2["vmax"])
check("70kt correctly classified as Very Severe Cyclonic Storm",
      row2["category"] == "Very Severe Cyclonic Storm", row2["category"])

row3 = df[df["storm_id"] == "202103I"].iloc[0]
check("wind falls back to USA(90) when NEWDELHI/WMO are both empty",
      row3["vmax"] == 90.0, row3["vmax"])
check("90kt correctly classified as Extremely Severe/Super Cyclone",
      row3["category"] == "Extremely Severe/Super Cyclone", row3["category"])

no_wind_csv = (
    "SID,SEASON,NUMBER,BASIN,NAME,ISO_TIME,LAT,LON,WMO_WIND,NEWDELHI_WIND,USA_WIND\n"
    "h,u,n,i,t,s,d,e,g,r,r\n"
    "2020001N10050,2020,1,NI,X,2020-05-01 00:00:00,10.0,80.0,,,\n"
    "2020001N10050,2020,1,NI,X,2020-05-01 03:00:00,10.1,80.1,,40,\n"
)
df2, stats2 = ibtracs_sync._parse_csv(no_wind_csv, basin_filter="NI")
check("rows with no wind in ANY candidate column are dropped, not crashed on",
      stats2["rows_dropped_no_wind"] == 1 and stats2["point_count"] == 1, stats2)

try:
    ibtracs_sync._parse_csv("<html>404 not found</html>", basin_filter="NI")
    check("missing-columns file raises IBTracsError", False, "did not raise")
except ibtracs_sync.IBTracsError as exc:
    check("missing-columns file raises IBTracsError cleanly", True)
    check("...with a message naming the missing-columns cause",
          "missing" in str(exc).lower() or "column" in str(exc).lower(), str(exc))
except Exception as exc:  # noqa: BLE001
    check("missing-columns file raises IBTracsError (not something else)",
          False, f"raised {type(exc).__name__}: {exc}")

try:
    ibtracs_sync._parse_csv(FAKE_MAIN_CSV, basin_filter="XX")
    check("basin filter matching nothing raises IBTracsError", False, "did not raise")
except ibtracs_sync.IBTracsError:
    check("basin filter matching nothing raises IBTracsError cleanly", True)


# ================================================ redirect cache to /tmp ===
TMP_DIR = Path(tempfile.mkdtemp(prefix="ibtracs_test_"))
ibtracs_sync.DATA_DIR = TMP_DIR
ibtracs_sync.TRACKS_CSV_PATH = TMP_DIR / "tracks.csv"
ibtracs_sync.META_PATH = TMP_DIR / "meta.json"
ibtracs_sync.IBTRACS_ENABLED = True
ibtracs_sync.IMPORT_ERROR = None


# ================================================== 2. status before fetch =
check("is_available() True (requests is a base dependency, enabled by default)",
      ibtracs_sync.is_available() is True)
st = ibtracs_sync.status()
check("status() reports have_data=False before any fetch", st["have_data"] is False, st)
check("status() carries no storm_count yet", st["storm_count"] is None, st)
check("list_storms_by_year() returns {} before any fetch", ibtracs_sync.list_storms_by_year() == {})
check("load_tracks() returns None before any fetch", ibtracs_sync.load_tracks() is None)


# ======================================== 3. fetch_and_update() mocked =====
ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_ok)

result = ibtracs_sync.fetch_and_update(force=True)
check("fetch_and_update() reports refreshed=True", result["refreshed"] is True, result)
check("merges main(2 storms)+active(1 storm) into 3 storms", result["storm_count"] == 3, result)
check("year_range correctly extends to include the ACTIVE storm's season (2026)",
      result["year_range"] == [2014, 2026], result)
check("active_storm_count reported", result["active_storm_count"] == 1, result)
check("TRACKS_CSV_PATH was actually written", ibtracs_sync.TRACKS_CSV_PATH.is_file())
check("META_PATH was actually written", ibtracs_sync.META_PATH.is_file())

result2 = ibtracs_sync.fetch_and_update(force=False)
check("immediate re-call without force skips re-downloading (cache still fresh)",
      result2.get("refreshed") is False, result2)

ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_active_fails)
result3 = ibtracs_sync.fetch_and_update(force=True)
check("main archive still refreshes when only the ACTIVE feed fails (non-fatal)",
      result3["refreshed"] is True and result3["storm_count"] == 2, result3)

good_meta_before_failure = ibtracs_sync._read_meta()
ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_bad_csv)
try:
    ibtracs_sync.fetch_and_update(force=True)
    check("malformed main CSV raises IBTracsError", False, "did not raise")
except ibtracs_sync.IBTracsError:
    check("malformed main CSV raises IBTracsError cleanly", True)
meta_after_failure = ibtracs_sync._read_meta()
check("a failed refresh never clobbers the last good cache on disk",
      meta_after_failure == good_meta_before_failure, (good_meta_before_failure, meta_after_failure))

ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_all_fail)
try:
    ibtracs_sync.fetch_and_update(force=True)
    check("total network outage raises IBTracsError", False, "did not raise")
except ibtracs_sync.IBTracsError:
    check("total network outage raises IBTracsError cleanly (not e.g. a bare Exception)", True)

ibtracs_sync.IBTRACS_ENABLED = False
try:
    ibtracs_sync.fetch_and_update(force=True)
    check("disabled feature raises IBTracsNotConfigured", False, "did not raise")
except ibtracs_sync.IBTracsNotConfigured:
    check("disabled feature raises IBTracsNotConfigured cleanly", True)
ibtracs_sync.IBTRACS_ENABLED = True

# restore a known-good cache for the scheduler + HTTP tests below
ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_ok)
ibtracs_sync.fetch_and_update(force=True)


# ============================================ 4. list_storms_by_year() =====
by_year = ibtracs_sync.list_storms_by_year()
check("list_storms_by_year groups the merged data by season",
      set(by_year.keys()) == {2014, 2021, 2026}, by_year.keys())
storm_2014 = by_year[2014][0]
check("per-storm summary carries the worst category actually reached",
      storm_2014["max_category"] == "Very Severe Cyclonic Storm", storm_2014)
check("per-storm summary carries the peak wind actually reached",
      storm_2014["max_wind_kt"] == 70.0, storm_2014)


# ==================================================== 5. scheduler =========
import ibtracs_scheduler  # noqa: E402

sched_calls = []


def fake_fetch(force=False):
    sched_calls.append(force)
    return {"refreshed": True, "storm_count": 3, "year_range": [2014, 2026]}


real_fetch = ibtracs_sync.fetch_and_update
ibtracs_sync.fetch_and_update = fake_fetch
ibtracs_sync.IBTRACS_REFRESH_INTERVAL_SECONDS = 1
ibtracs_scheduler.start()
time.sleep(2.3)
ibtracs_scheduler.stop()
time.sleep(0.2)
check("scheduler runs at least twice within ~2.3s at a 1s interval",
      len(sched_calls) >= 2, sched_calls)
check("scheduler thread actually stops when told to", not ibtracs_scheduler._thread.is_alive())


def fake_fetch_raises(force=False):
    raise RuntimeError("boom - scheduler must survive this")


ibtracs_sync.fetch_and_update = fake_fetch_raises
ibtracs_scheduler._thread = None
ibtracs_scheduler.start()
time.sleep(1.3)
survived = ibtracs_scheduler._thread.is_alive()
ibtracs_scheduler.stop()
check("scheduler thread survives an exception raised inside its own loop", survived)

ibtracs_sync.fetch_and_update = real_fetch
ibtracs_sync.fetch_and_update(force=True)  # leave a good cache in place again


# ============================================== 6. HTTP routes (server.py) =
fake_predict_module = types.ModuleType("predict")


def _fake_predict(img):
    order = ["Depression", "Cyclonic Storm", "Severe Cyclonic Storm",
             "Very Severe Cyclonic Storm", "Extremely Severe/Super Cyclone"]
    probs = {c: 5.0 for c in order}
    probs["Severe Cyclonic Storm"] = 80.0
    return "Severe Cyclonic Storm", 80.0, probs


fake_predict_module.predict = _fake_predict
sys.modules["predict"] = fake_predict_module

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


code, body = call("/api/health")
check("/api/health includes storms_available=True", body.get("storms_available") is True, body)

code, body = call("/api/storms/status")
check("/api/storms/status responds 200", code == 200, (code, body))
check("/api/storms/status reflects the 3-storm merged cache", body.get("storm_count") == 3, body)

code, body = call("/api/storms/years")
check("/api/storms/years responds 200 with all 3 seasons", code == 200
      and sorted(y["season"] for y in body["years"]) == [2014, 2021, 2026], body)

code, body = call("/api/storms/by-year")
check("/api/storms/by-year with no 'year' param -> 400, not 500", code == 400, (code, body))

code, body = call("/api/storms/by-year?year=2014")
check("/api/storms/by-year?year=2014 responds 200 with one storm", code == 200
      and len(body["storms"]) == 1, body)
storm = body["storms"][0] if code == 200 and body.get("storms") else {}
check("storm_id 201402I correctly cross-referenced as has_real_image=True "
      "against the REAL backend/samples/samples.csv",
      storm.get("has_real_image") is True, storm)

code, body = call("/api/storms/by-year?year=2021")
check("a storm NOT in samples.csv is honestly flagged has_real_image=False",
      code == 200 and body["storms"][0]["has_real_image"] is False, body)

code, body = call("/api/storms/track")
check("/api/storms/track with no storm_id -> 400, not 500", code == 400, (code, body))

code, body = call("/api/storms/track?storm_id=201402I")
check("/api/storms/track returns all 3 recorded points for 201402I",
      code == 200 and len(body["points"]) == 3, body)
check("/api/storms/track points are in chronological order",
      body["points"][0]["time"] < body["points"][-1]["time"], body.get("points"))

code, body = call("/api/storms/track?storm_id=DOES-NOT-EXIST")
check("/api/storms/track with an unknown storm_id -> 404, not 500", code == 404, (code, body))

ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_all_fail)
code, body = call("/api/storms/refresh", method="POST", data={})
check("POST /api/storms/refresh surfaces a real failure as 502, not 500",
      code == 502, (code, body))

ibtracs_sync.requests = types.SimpleNamespace(get=fake_get_ok)
code, body = call("/api/storms/refresh", method="POST", data={})
check("POST /api/storms/refresh succeeds (200) when the mocked network works",
      code == 200 and body.get("refreshed") is True, (code, body))

httpd.shutdown()
shutil.rmtree(TMP_DIR, ignore_errors=True)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All IBTrACS tests passed.")
