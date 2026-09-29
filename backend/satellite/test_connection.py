#!/usr/bin/env python3
"""Real connectivity test - RUN THIS ON A MACHINE WITH REAL INTERNET ACCESS.

This could not be executed from the sandboxed environment this project was
built in (it has no route to the public internet - not even to pypi.org).
Everything in this file is written to genuinely test the live pipeline;
none of its "PASS" results are hard-coded.

Usage (from the PRISM-TC-Frontend folder):
    python backend/satellite/test_connection.py

Exit code 0 = all checks passed. Non-zero = see the printed failure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import satellite...` style below works

from satellite import config  # noqa: E402
from satellite.sources.imd_insat import IMDInsatSource  # noqa: E402
from satellite.sources.base import SourceError  # noqa: E402


def _step(n, total, text):
    print(f"[{n}/{total}] {text}")


def main() -> int:
    total = 5
    source = IMDInsatSource()

    _step(1, total, f"Reaching {config.IMD_INSAT_IR1_URL} ...")
    try:
        meta = source.get_latest_metadata()
    except SourceError as exc:
        print(f"    FAILED at stage '{exc.stage}': {exc}")
        print("    -> Check your internet connection, or that IMD hasn't moved/renamed "
              "this file (see sources/imd_insat.py).")
        return 1
    print(f"    OK - satellite={meta.satellite}")
    print(f"    Timestamp: {meta.observation_time_utc} UTC (source: {meta.extra.get('timestamp_source')})")
    if meta.extra.get("timestamp_source") == "fetch_time_fallback":
        print("    NOTE: the server did not send a Last-Modified header, so this is the "
              "time WE fetched it, not a confirmed satellite scan time. The exact scan "
              "window is printed inside the image itself (a GMT/IST banner at the top) - "
              "open test_output_imd.jpg yourself and compare if you need the exact time.")

    _step(2, total, "Checking observation freshness ...")
    try:
        from datetime import datetime, timezone
        obs = datetime.strptime(meta.observation_time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - obs).total_seconds() / 60
        print(f"    {age_min:.1f} minutes since {'last fetch' if meta.extra.get('timestamp_source') == 'fetch_time_fallback' else 'reported update'}.")
    except Exception as exc:  # noqa: BLE001
        print(f"    Could not compute age: {exc}")

    _step(3, total, "Downloading the image ...")
    t0 = time.time()
    try:
        image, raw = source.download(meta)
    except SourceError as exc:
        print(f"    FAILED at stage '{exc.stage}': {exc}")
        return 1
    print(f"    OK - {image.width}x{image.height}px, {len(raw)} bytes, {time.time() - t0:.1f}s")

    _step(4, total, "Validating image ...")
    if image.width < 100 or image.height < 100:
        print("    FAILED: image implausibly small.")
        return 1
    out_path = Path(__file__).resolve().parent / "test_output_imd.jpg"
    image.save(out_path)
    print(f"    OK - saved: {out_path}")
    print("    >>> IMPORTANT: open test_output_imd.jpg yourself and confirm it shows India, "
          "the Arabian Sea and the Bay of Bengal, with a real IR cloud picture (not a blank "
          "or broken image). This is the one thing this script cannot verify for you.")
    print("    Note: this source is used as-is (no programmatic crop) - see "
          "sources/imd_insat.py's region_description for its documented approximate extent.")

    _step(5, total, "Running the full pipeline (pipeline.run_once) ...")
    from satellite import pipeline  # noqa: E402  (import here: needs config already loaded)
    state = pipeline.run_once()
    if state.get("status") == "ERROR":
        print(f"    FAILED: {state.get('error')}")
        return 1
    print(f"    OK - status={state.get('status')}, image at data/satellite/latest/latest.png")

    print("\nAll checks passed. Start the server (python backend/server.py) and open "
          "http://localhost:8000, or curl http://localhost:8000/api/satellite/status, "
          "to see this wired into the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
