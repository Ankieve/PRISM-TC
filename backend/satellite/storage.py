"""Local JSON/file storage for the satellite pipeline. No database - a
handful of small JSON files plus the image files themselves, matching the
"avoid unnecessary infrastructure" requirement. Writes are atomic (write to
a temp file, then rename) so a crash mid-write can't corrupt state.json and
leave the whole app unable to start.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config

_LOCK = threading.Lock()

_DEFAULT_STATE = {
    "status": "NO_DATA",       # NO_DATA | UPDATING | LIVE | STALE | ERROR
    "source": None,
    "observation_time_utc": None,   # when the satellite took the image
    "received_time_utc": None,      # when OUR pipeline finished downloading it
    "processed_time_utc": None,     # when preprocessing/crop finished
    "channel": None,
    "product": None,
    "image_path": None,             # path under data/satellite/latest/
    "prediction": None,             # only set if ENABLE_LIVE_INFERENCE is on
    "error": None,
    "error_stage": None,
    "last_check_utc": None,
    "last_success_utc": None,
}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def read_state() -> dict:
    with _LOCK:
        if not config.STATE_FILE.is_file():
            return dict(_DEFAULT_STATE)
        try:
            data = json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
            merged = dict(_DEFAULT_STATE)
            merged.update(data)
            return merged
        except (json.JSONDecodeError, OSError):
            # Corrupted state file must never crash the app or the frontend.
            return dict(_DEFAULT_STATE)


def write_state(patch: dict) -> dict:
    with _LOCK:
        current = dict(_DEFAULT_STATE)
        if config.STATE_FILE.is_file():
            try:
                current.update(json.loads(config.STATE_FILE.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        current.update(patch)
        _atomic_write_json(config.STATE_FILE, current)
        return current


def append_history(entry: dict, max_entries: Optional[int] = None) -> None:
    max_entries = max_entries or config.MAX_HISTORY_ENTRIES
    with _LOCK:
        history = []
        if config.HISTORY_FILE.is_file():
            try:
                history = json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(entry)
        history = history[-max_entries:]
        _atomic_write_json(config.HISTORY_FILE, history)


def read_history(limit: int = 50) -> list:
    with _LOCK:
        if not config.HISTORY_FILE.is_file():
            return []
        try:
            history = json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                return []
            return history[-limit:][::-1]  # newest first
        except (json.JSONDecodeError, OSError):
            return []


def already_have_observation(observation_time_utc: str) -> bool:
    """Duplicate detection: have we already successfully processed this
    exact observation timestamp?"""
    state = read_state()
    return state.get("observation_time_utc") == observation_time_utc and state.get("status") in (
        "LIVE", "STALE",
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
