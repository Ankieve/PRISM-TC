"""Background polling thread. Started (optionally) from server.py's
__main__ block. Runs pipeline.run_once() on a fixed interval, matched to
the actual product cadence (see config.UPDATE_INTERVAL_SECONDS) rather than
an arbitrary short number - polling a 10-minute product every few seconds
would just hammer the source for no benefit.

Designed to never take the process down: any exception inside the loop is
caught and logged, and the thread keeps running on the same schedule.
"""
from __future__ import annotations

import logging
import threading
import time

from . import config, pipeline

log = logging.getLogger("satellite")

_thread = None
_stop_event = threading.Event()


def _loop():
    # Run one cycle immediately on startup so the dashboard has something
    # (or a clear error) right away, instead of waiting a full interval.
    while not _stop_event.is_set():
        try:
            pipeline.run_once()
        except Exception as exc:  # noqa: BLE001 - the scheduler must never die
            log.error("Unhandled error in satellite scheduler loop: %s", exc)
        _stop_event.wait(config.UPDATE_INTERVAL_SECONDS)


def start():
    global _thread
    if not config.SATELLITE_ENABLED:
        log.info("Satellite feature disabled (SATELLITE_ENABLED=false in .env). Skipping scheduler.")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, name="satellite-scheduler", daemon=True)
    _thread.start()
    log.info("Scheduler started (polling every %ds, source=%s).",
              config.UPDATE_INTERVAL_SECONDS, config.SATELLITE_SOURCE)


def stop():
    _stop_event.set()
