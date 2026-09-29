"""Async ERA5 job tests (BUG 2 regression): immediate return, dedup, hung-child
kill + retry, seed/cache/live order, RSS < 300 MB. No network, no credentials.

The hung-child test uses the ERA5_TEST_HANG hook (read by the worker child),
so it exercises the REAL spawn + terminate path on every OS. This module is
guarded so spawn can safely re-import it.

Run:  python backend/tests/test_era5_jobs.py
"""
import os
import sys
import threading
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond:
        fails.append(name)


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1048576
    except ImportError:
        return None


def main():
    import era5  # noqa: E402

    peak = (rss_mb() or 0)

    def track():
        nonlocal peak
        r = rss_mb()
        if r:
            peak = max(peak, r)

    # --- 1. submit returns immediately ---------------------------------------
    t0 = time.time()
    job = era5.submit_fetch(15.5, 85.0)
    dt = time.time() - t0
    check("submit returns immediately (<3s, never held open)", dt < 3, f"{dt:.2f}s")
    track()
    check("no-seed + live-off job fails with a clear message (never invented)",
          job["status"] == "failed" and "seed" in job["error"].lower()
          and "ERA5_LIVE" in job["error"], str(job)[:160])

    # --- 2. seed-first order --------------------------------------------------
    t0 = time.time()
    seed_job = era5.submit_fetch(16.1, 68.1)  # committed seed 16.00_68.00
    check("seed point returns done immediately", seed_job["status"] == "done"
          and time.time() - t0 < 3, str(seed_job)[:160])
    check("seed result is real reanalysis with valid_time",
          bool(seed_job["result"].get("valid_time_utc"))
          and seed_job["result"].get("seed") is True, str(seed_job["result"])[:160])
    track()

    # --- 3. dedup + hung-child kill + retry -----------------------------------
    era5.ERA5_LIVE = True
    era5.ERA5_REQUEST_TIMEOUT_SECONDS = 3
    os.environ["ERA5_TEST_HANG"] = "60"
    try:
        j1 = era5.submit_fetch(10.0, 70.0)
        j2 = era5.submit_fetch(10.0, 70.0)
        check("second identical request attaches (same job, no new fetch)",
              j1["job_id"] == j2["job_id"] and j2.get("attached") is True
              and j1["status"] in ("queued", "running"), (j1["job_id"], j2.get("attached")))
        deadline = time.time() + 90
        cur = era5.get_job(j1["job_id"])
        while cur["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(0.5)
            cur = era5.get_job(j1["job_id"])
        check("hung child killed by hard timeout -> job failed, worker alive",
              cur["status"] == "failed" and "did not respond within 3s" in cur["error"]
              and (era5._WORKER_THREAD is not None and era5._WORKER_THREAD.is_alive()),
              str(cur)[:200])
    finally:
        del os.environ["ERA5_TEST_HANG"]
        era5.ERA5_LIVE = False
    track()

    # retry after the kill works (seed path, instant)
    retry = era5.submit_fetch(16.1, 68.1)
    check("next fetch works right after the kill", retry["status"] == "done",
          str(retry)[:120])
    track()

    # --- 4. stale-job expiry ---------------------------------------------------
    now = time.time()
    stale = {"job_id": "stale-test", "key": "k", "lat": 0.0, "lon": 0.0,
             "when_iso": "x", "created_utc": "x", "status": "running",
             "result": None, "error": None, "created": now,
             "updated": now, "started": now - (era5.ERA5_REQUEST_TIMEOUT_SECONDS + 130)}
    with era5._JOBS_LOCK:
        era5._JOBS["stale-test"] = stale
    got = era5.get_job("stale-test")
    check("stale running job expires to failed (never hangs)",
          got is not None and got["status"] == "failed", str(got)[:160])
    with era5._JOBS_LOCK:
        era5._JOBS.pop("stale-test", None)
    track()

    print(f"\npeak RSS during job sequence: {peak:.1f} MB" if peak else "\npsutil missing - RSS not measured")
    if peak:
        check("RSS stays under 300 MB", peak < 300, f"{peak:.1f}")

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        sys.exit(1)
    print("All ERA5 job tests passed.")


if __name__ == "__main__":
    main()
