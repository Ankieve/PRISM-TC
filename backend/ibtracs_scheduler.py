"""Background thread that keeps the IBTrACS historical-track cache
(backend/ibtracs_sync.py) refreshed once a day, so the Year -> Storm
browser stays current "day to day" without anyone needing to restart the
server or manually re-fetch.

Same defensive shape as backend/satellite/scheduler.py: runs one cycle
immediately on startup (so the browser has real data - or a clear error -
right away instead of waiting a full day), then waits
ibtracs_sync.IBTRACS_REFRESH_INTERVAL_SECONDS between attempts. Any
exception inside the loop is caught and logged; the thread itself never
dies, and a failed refresh simply leaves yesterday's good data in place
(ibtracs_sync.fetch_and_update never deletes the last good file on
failure).
"""
from __future__ import annotations

import logging
import threading

import ibtracs_sync

log = logging.getLogger("ibtracs")

_thread = None
_stop_event = threading.Event()


def _loop():
    while not _stop_event.is_set():
        try:
            result = ibtracs_sync.fetch_and_update()
            if result.get("refreshed"):
                log.info("Daily refresh done: %s storms, seasons %s.",
                          result.get("storm_count"), result.get("year_range"))
        except ibtracs_sync.IBTracsNotConfigured as exc:
            log.info("IBTrACS sync disabled, scheduler will keep idling: %s", exc)
        except Exception as exc:  # noqa: BLE001 - the scheduler must never die
            log.error("Unhandled error in IBTrACS scheduler loop: %s", exc)
        _stop_event.wait(ibtracs_sync.IBTRACS_REFRESH_INTERVAL_SECONDS)


def start():
    global _thread
    if not ibtracs_sync.IBTRACS_ENABLED:
        log.info("IBTrACS feature disabled (IBTRACS_ENABLED=false). Skipping scheduler.")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, name="ibtracs-scheduler", daemon=True)
    _thread.start()
    log.info("Scheduler started (refreshing every %ds, basin=%s).",
              ibtracs_sync.IBTRACS_REFRESH_INTERVAL_SECONDS, ibtracs_sync.IBTRACS_BASIN)


def stop():
    _stop_event.set()
