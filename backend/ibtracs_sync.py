"""Real historical cyclone-track sync (IBTrACS, NOAA NCEI).

What this is
------------
IBTrACS (International Best Track Archive for Climate Stewardship) is NOAA
NCEI's public, free, no-account-needed global "best track" archive - the
same kind of REAL observed-position/observed-intensity record used by
backend/data/track_data.csv (the small 75-storm set the AI-prediction demo
and backtest already ship with), just the FULL North Indian Ocean record
instead of a hand-picked subset, downloaded fresh instead of bundled once.

This module does NOT touch backend/data/track_data.csv. That file is a
curated 75-storm set cross-referenced against the 10 real TCIR sample
images in backend/samples/ (see samples.csv) and the AI-prediction demo/
backtest are built around it - overwriting it would silently break which
storms can run "Run AI Prediction" vs "Historical track only". Instead this
module downloads into its own file (see TRACKS_CSV_PATH) that a *separate*
year-by-year browsing feature reads from, so both data sources coexist
without either one clobbering the other.

Coverage note: NCEI's own climatology tables describe North Indian Ocean
("NI" basin) coverage as reliable from roughly 1981 onward; sparser records
exist further back (older seasons may have fewer points, coarser positions,
or a wind estimate reconstructed after the fact rather than observed at the
time) - IBTrACS ships them anyway with no explicit reliability flag per row,
so this module does not filter by year; the year_range reported by status()
is simply whatever seasons are present in the downloaded file.

Honesty notes (read before trusting a number this returns)
------------------------------------------------------------
1. This code could not be exercised against the real NCEI server from the
   sandbox this was written in (no internet access there). The base URL,
   file names (ibtracs.NI.list.v04r01.csv / ibtracs.ACTIVE.list.v04r01.csv)
   and column names below (SID, NAME, SEASON, NUMBER, ISO_TIME, BASIN, the
   plain LAT/LON columns, the agency-prefixed *_WIND columns) match NOAA
   NCEI's published IBTrACS v04r01 CSV format and the Google Earth Engine
   catalog page for this dataset at the time of writing, but have not been
   confirmed against a live download. The very first real fetch (on a
   machine with internet) should be treated as the actual verification -
   if column names have changed, this will raise IBTracsError with the
   parse failure rather than silently producing wrong data (see
   _parse_csv). Same "unverified, confirm before trusting" flag already
   used for the Himawari satellite source and for era5.py.
2. storm_id is constructed here as f"{SEASON}{NUMBER:02d}I" to match the
   convention already used by backend/data/track_data.csv and
   backend/samples/samples.csv (e.g. "201402I"), not IBTrACS's own SID
   field (which looks like "2014028N10093"). This reconstruction is an
   assumption inherited from how the existing curated files were built; it
   has not been independently verified against IBTrACS documentation. It
   only matters for cross-referencing against the existing 10 samples - if
   it is ever wrong, the practical effect is just that a sample fails to
   match its storm in the browser, not incorrect track data.
3. Per-row lat/lon/wind values fall back across several candidate IBTrACS
   columns (see LAT_COLS / LON_COLS / WIND_COLS) because North Indian Ocean
   storms are tracked by RSMC New Delhi but IBTrACS also carries a WMO
   consensus position and (for some storms) a USA/JTWC position, and not
   every column is populated for every row. A row with no usable wind value
   in ANY candidate column is dropped (category cannot be derived without
   one) - this is disclosed in the written metadata as rows_dropped_no_wind.
4. This is entirely optional and safe to leave off. If IBTRACS_ENABLED is
   false, or the download/parse fails, every function below either returns
   a clear status or raises IBTracsNotConfigured / IBTracsError - it never
   crashes the server, and the last successfully-parsed file on disk is
   left untouched so the site keeps showing the most recent good data
   instead of nothing (same defensive pattern as backend/satellite/ and
   backend/era5.py).
5. This has zero new pip dependencies - it uses `requests`, already listed
   in backend/requirements.txt for backend/satellite/.

What "update day to day" means here
------------------------------------
IBTrACS's own main archive is republished by NCEI roughly 3x/week (not
truly daily), plus a separate "ACTIVE" file limited to storms active in the
last 7 days for near-real-time status. This module's scheduler (see
ibtracs_scheduler below / backend/ibtracs_scheduler.py) re-downloads once
per IBTRACS_REFRESH_INTERVAL_SECONDS (default: once every 24h) and merges
in the ACTIVE subset - so the SITE checks for new data daily even though
the upstream source itself does not change every single day. When nothing
new has been published, a refresh simply reproduces the same file.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from logic import CLASS_ORDER, class_from_wind

BACKEND_DIR = Path(__file__).resolve().parent

log = logging.getLogger("ibtracs")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[IBTrACS] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


class IBTracsNotConfigured(Exception):
    """Feature disabled via env var - not a real failure."""


class IBTracsError(Exception):
    """A real download/parse attempt was made and it failed."""


# ------------------------------------------------------------------ config --
def _bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


IBTRACS_ENABLED = _bool("IBTRACS_ENABLED", True)
# North Indian Ocean - the basin covering India and its surrounding seas
# (Arabian Sea + Bay of Bengal). IBTrACS's other basin codes (NA, EP, WP,
# SI, SP, SA) are not fetched by default; override if a different basin
# is ever wanted.
IBTRACS_BASIN = os.environ.get("IBTRACS_BASIN", "NI").strip().upper()
# NCEI republishes IBTrACS ~3x/week, not truly daily - refreshing once every
# 24h is already more often than the source usually changes; lower this if
# a faster check is wanted, it just costs a redundant download most days.
IBTRACS_REFRESH_INTERVAL_SECONDS = _int("IBTRACS_REFRESH_INTERVAL_SECONDS", 24 * 3600)
IBTRACS_REQUEST_TIMEOUT_SECONDS = _int("IBTRACS_REQUEST_TIMEOUT_SECONDS", 60)

BASE_URL = ("https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r01/access/csv/")

DATA_DIR = Path(os.environ.get("IBTRACS_DATA_DIR", BACKEND_DIR / "data" / "ibtracs_cache"))
TRACKS_CSV_PATH = DATA_DIR / f"ibtracs_{IBTRACS_BASIN.lower()}_tracks.csv"
META_PATH = DATA_DIR / f"ibtracs_{IBTRACS_BASIN.lower()}_meta.json"

# Candidate column names tried in order, per row - see Honesty note 3.
LAT_COLS = ["LAT", "NEWDELHI_LAT", "WMO_LAT", "USA_LAT"]
LON_COLS = ["LON", "NEWDELHI_LON", "WMO_LON", "USA_LON"]
WIND_COLS = ["NEWDELHI_WIND", "WMO_WIND", "USA_WIND"]

try:
    import requests  # type: ignore
    IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - should never happen, it's a base dependency
    requests = None
    IMPORT_ERROR = f"requests not installed ({type(_exc).__name__}: {_exc})"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def is_available() -> bool:
    return IBTRACS_ENABLED and requests is not None


def _read_meta() -> dict | None:
    if not META_PATH.is_file():
        return None
    try:
        return json.loads(META_PATH.read_text())
    except (OSError, ValueError):
        return None


def status() -> dict:
    """Cheap, side-effect-free status for GET /api/storms/status - never
    makes a network call."""
    meta = _read_meta()
    now = time.time()
    age_seconds = (now - meta["fetched_at_epoch"]) if meta else None
    return {
        "enabled": IBTRACS_ENABLED,
        "import_error": IMPORT_ERROR,
        "basin": IBTRACS_BASIN,
        "have_data": meta is not None and TRACKS_CSV_PATH.is_file(),
        "storm_count": meta.get("storm_count") if meta else None,
        "point_count": meta.get("point_count") if meta else None,
        "year_range": meta.get("year_range") if meta else None,
        "rows_dropped_no_wind": meta.get("rows_dropped_no_wind") if meta else None,
        "active_storm_count": meta.get("active_storm_count") if meta else None,
        "last_refreshed_utc": meta.get("fetched_at_utc") if meta else None,
        "age_seconds": age_seconds,
        "stale": age_seconds is not None and age_seconds > IBTRACS_REFRESH_INTERVAL_SECONDS * 2,
        "refresh_interval_seconds": IBTRACS_REFRESH_INTERVAL_SECONDS,
        "source_url": meta.get("source_url") if meta else None,
        "note": ("Historical track archive (NOAA IBTrACS), not a forecast. "
                 "The upstream archive itself is republished a few times a "
                 "week, not every single day - see backend/ibtracs_sync.py."),
    }


# --------------------------------------------------------------- download --
def _download_csv(filename: str) -> str:
    if requests is None:
        raise IBTracsNotConfigured(IMPORT_ERROR or "requests is not available.")
    url = BASE_URL + filename
    try:
        resp = requests.get(
            url,
            timeout=IBTRACS_REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "PRISM-TC-cyclone-dashboard/1.0 (educational hackathon project)"},
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise IBTracsError(f"Could not download {url}: {type(exc).__name__}: {exc}") from exc
    return resp.text


def _first_valid(row: "pd.Series", cols: list[str]):
    for c in cols:
        if c in row.index:
            v = row[c]
            try:
                if pd.notna(v):
                    return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _parse_csv(raw_text: str, basin_filter: str | None) -> tuple["pd.DataFrame", dict]:
    """Turn a raw IBTrACS CSV (header row + units row + data rows) into the
    storm_id,time,lat,lon,vmax,category,name,season schema. Returns
    (dataframe, stats). Raises IBTracsError if the file does not look like
    IBTrACS at all (e.g. an HTML error page instead of a CSV)."""
    try:
        # Row 0 = header, row 1 = units (e.g. "YYYY-MM-DD HH:MM:SS", "deg",
        # "kts") - not real data, must be skipped.
        df = pd.read_csv(io.StringIO(raw_text), skiprows=[1], low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise IBTracsError(f"Downloaded file did not parse as CSV: {exc}") from exc

    required = {"SID", "SEASON", "NUMBER", "ISO_TIME", "BASIN"}
    missing = required - set(df.columns)
    if missing:
        raise IBTracsError(
            f"Downloaded IBTrACS file is missing expected columns {sorted(missing)} - "
            "NCEI may have changed the CSV format since this module was written "
            "(see backend/ibtracs_sync.py Honesty note 1).")

    if basin_filter:
        df = df[df["BASIN"].astype(str).str.strip() == basin_filter]

    rows = []
    dropped_no_wind = 0
    for _, row in df.iterrows():
        wind_kt = _first_valid(row, WIND_COLS)
        if wind_kt is None:
            dropped_no_wind += 1
            continue
        lat = _first_valid(row, LAT_COLS)
        lon = _first_valid(row, LON_COLS)
        if lat is None or lon is None:
            dropped_no_wind += 1  # no usable position either - can't plot it
            continue
        try:
            season = int(float(row["SEASON"]))
            number = int(float(row["NUMBER"]))
        except (TypeError, ValueError):
            continue
        storm_id = f"{season}{number:02d}I"
        try:
            dt = pd.to_datetime(row["ISO_TIME"])
        except Exception:  # noqa: BLE001
            continue
        cat_index = class_from_wind(wind_kt)
        rows.append({
            "storm_id": storm_id,
            "time": int(dt.strftime("%Y%m%d%H")),
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "vmax": round(wind_kt, 1),
            "category": CLASS_ORDER[cat_index],
            "name": str(row.get("NAME", "NOT_NAMED")).strip() or "NOT_NAMED",
            "season": season,
        })

    if not rows:
        raise IBTracsError(
            "Parsed 0 usable rows out of the downloaded file - either the basin "
            f"filter ({basin_filter!r}) matched nothing or every row was missing "
            "wind/position data. Treat this as a real failure, not empty data.")

    out = pd.DataFrame(rows).drop_duplicates(subset=["storm_id", "time"])
    out = out.sort_values(["storm_id", "time"]).reset_index(drop=True)
    stats = {
        "point_count": int(len(out)),
        "storm_count": int(out["storm_id"].nunique()),
        "rows_dropped_no_wind": int(dropped_no_wind),
        "year_range": [int(out["season"].min()), int(out["season"].max())],
    }
    return out, stats


# ------------------------------------------------------------ public API ---
def fetch_and_update(force: bool = False) -> dict:
    """The one function the scheduler (and a manual /api/storms/refresh
    route) calls. Downloads the main basin file + the ACTIVE-storms subset,
    parses both, writes TRACKS_CSV_PATH + META_PATH, and returns the fresh
    status(). Raises IBTracsNotConfigured / IBTracsError on failure - on
    failure, the previous good TRACKS_CSV_PATH/META_PATH are left untouched
    so the site keeps serving the last real data instead of nothing.
    """
    if not is_available():
        raise IBTracsNotConfigured(
            IMPORT_ERROR or "IBTrACS sync is disabled (IBTRACS_ENABLED=false).")

    if not force:
        meta = _read_meta()
        if meta is not None:
            age = time.time() - meta.get("fetched_at_epoch", 0)
            if age < IBTRACS_REFRESH_INTERVAL_SECONDS and TRACKS_CSV_PATH.is_file():
                return {**status(), "refreshed": False,
                        "reason": "cached data is still within the refresh interval"}

    log.info("Fetching ibtracs.%s.list.v04r01.csv from NCEI...", IBTRACS_BASIN)
    main_text = _download_csv(f"ibtracs.{IBTRACS_BASIN}.list.v04r01.csv")
    main_df, main_stats = _parse_csv(main_text, basin_filter=IBTRACS_BASIN)

    active_storm_count = None
    try:
        log.info("Fetching ibtracs.ACTIVE.list.v04r01.csv (near-real-time, last 7 days)...")
        active_text = _download_csv("ibtracs.ACTIVE.list.v04r01.csv")
        active_df, _ = _parse_csv(active_text, basin_filter=IBTRACS_BASIN)
        active_storm_count = int(active_df["storm_id"].nunique())
        # Merge any brand-new points from the active feed in with the main
        # set (e.g. a storm currently forming that the tri-weekly main
        # archive hasn't picked up yet) - de-duplicated by (storm_id, time).
        main_df = (pd.concat([main_df, active_df], ignore_index=True)
                   .drop_duplicates(subset=["storm_id", "time"])
                   .sort_values(["storm_id", "time"]).reset_index(drop=True))
        main_stats["point_count"] = int(len(main_df))
        main_stats["storm_count"] = int(main_df["storm_id"].nunique())
        main_stats["year_range"] = [int(main_df["season"].min()), int(main_df["season"].max())]
    except IBTracsError as exc:
        # The ACTIVE feed is a nice-to-have for near-real-time freshness; if
        # it fails for any reason, the main historical archive we already
        # have is still real and still worth keeping - don't fail the whole
        # refresh over it.
        log.warning("ACTIVE-storms fetch failed (non-fatal, keeping main archive only): %s", exc)

    ensure_dirs()
    main_df.to_csv(TRACKS_CSV_PATH, index=False)
    meta = {
        "basin": IBTRACS_BASIN,
        "source_url": BASE_URL + f"ibtracs.{IBTRACS_BASIN}.list.v04r01.csv",
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_at_epoch": time.time(),
        "active_storm_count": active_storm_count,
        **main_stats,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("Wrote %d points across %d storms (seasons %s-%s) to %s",
              meta["point_count"], meta["storm_count"],
              meta["year_range"][0], meta["year_range"][1], TRACKS_CSV_PATH.name)
    return {**status(), "refreshed": True}


def load_tracks() -> "pd.DataFrame | None":
    """Read the cached parsed CSV, or None if nothing has been fetched yet.
    Never makes a network call - safe to call from a request handler."""
    if not TRACKS_CSV_PATH.is_file():
        return None
    try:
        return pd.read_csv(TRACKS_CSV_PATH)
    except (OSError, ValueError) as exc:
        log.warning("Could not read cached tracks file: %s", exc)
        return None


def list_storms_by_year() -> dict:
    """{season: [ {storm_id, name, start, end, point_count, max_category,
    max_wind_kt}, ... ]} built entirely from the cached file - no network
    call. Returns {} if nothing has been fetched yet. Consumed by the
    Year -> Storm browser (frontend) via a small server.py route."""
    df = load_tracks()
    if df is None or df.empty:
        return {}
    result: dict = {}
    for storm_id, g in df.groupby("storm_id"):
        g = g.sort_values("time")
        season = int(g["season"].iloc[0])
        cat_idx = max(CLASS_ORDER.index(c) for c in g["category"])
        result.setdefault(season, []).append({
            "storm_id": storm_id,
            "name": g["name"].iloc[0],
            "start_time": int(g["time"].iloc[0]),
            "end_time": int(g["time"].iloc[-1]),
            "point_count": int(len(g)),
            "max_category": CLASS_ORDER[cat_idx],
            "max_wind_kt": float(g["vmax"].max()),
        })
    for season in result:
        result[season].sort(key=lambda s: s["start_time"])
    return dict(sorted(result.items()))
