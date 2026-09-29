"""Smoke test for the /api/satellite/* endpoints.

Runs the REAL server (fake model, like smoke_test.py) with the satellite
scheduler DISABLED (SATELLITE_ENABLED=false), so this test is self-contained
and does not depend on internet access - it checks that every endpoint
responds sensibly with NO satellite data yet, that a manual POST /update
against an unreachable network fails cleanly (ERROR, not a crash/500), and
that the server's existing routes are completely unaffected throughout.

Run:  python backend/tests/smoke_test_satellite.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Fake model, same trick as smoke_test.py, so this runs anywhere.
import types
fake = types.ModuleType("predict")


def _fake_predict(image):
    return "Depression", 42.0, {"Depression": 42.0, "Cyclonic Storm": 20.0,
                                 "Severe Cyclonic Storm": 20.0,
                                 "Very Severe Cyclonic Storm": 10.0,
                                 "Extremely Severe/Super Cyclone": 8.0}


fake.predict = _fake_predict
sys.modules["predict"] = fake

# Point the satellite data dir at a throwaway temp folder so this test never
# touches (or depends on) real captured data from a previous run.
_tmp_data_dir = tempfile.mkdtemp(prefix="prism_tc_sat_smoke_")
os.environ["SATELLITE_DATA_DIR"] = _tmp_data_dir
os.environ["SATELLITE_ENABLED"] = "false"  # this test drives updates manually

import server  # noqa: E402

HOST, PORT = "127.0.0.1", 8791
BASE = f"http://{HOST}:{PORT}"

_passed = 0
_failed = 0


def check(name, condition):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def post(path, body=b""):
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    httpd = server.make_server(HOST, PORT)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        check("satellite import succeeded (module loaded without error)",
              server.SATELLITE_IMPORT_ERROR is None)

        status, body = get("/api/satellite/status")
        check("GET /api/satellite/status -> 200", status == 200)
        check("Before any update: status is NO_DATA", body.get("status") == "NO_DATA")

        status, body = get("/api/satellite/metadata")
        check("GET /api/satellite/metadata -> 200", status == 200)

        status, body = get("/api/satellite/image")
        check("GET /api/satellite/image -> 404 before any capture (not a 500/crash)", status == 404)

        status, body = get("/api/satellite/history")
        check("GET /api/satellite/history -> 200 with an empty list before any update",
              status == 200 and body == [])

        # This sandboxed test environment has no route to the public
        # internet, so a manual update MUST fail cleanly, not crash the
        # server and not return a fabricated success.
        status, body = post("/api/satellite/update")
        check("POST /api/satellite/update -> 200 (endpoint itself works even though the fetch fails)",
              status == 200)
        check("...and correctly reports ERROR status rather than faking success",
              body.get("status") == "ERROR")
        check("...with a non-empty error message", bool(body.get("error")))

        status, body = get("/api/satellite/status")
        check("Status after a failed update is ERROR, not silently NO_DATA or LIVE",
              status == 200 and body.get("status") == "ERROR")

        # The existing, previously-working routes must be completely
        # unaffected by any of the above.
        status, body = get("/api/health")
        check("Existing /api/health route still works", status == 200)
        status, body = get("/api/samples")
        check("Existing /api/samples route still works", status == 200 and isinstance(body, list))
        status, body = get("/api/demo")
        check("Existing /api/demo route still works", status == 200 and "prediction" in body)

    finally:
        httpd.shutdown()

    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
