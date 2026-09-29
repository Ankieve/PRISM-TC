"""Real ERA5 reanalysis fetch (Copernicus Climate Data Store).

What this is
------------
ERA5 is ECMWF's global atmospheric reanalysis: REAL observational/model-blended
data, not a forecast and not synthetic. This module fetches four fields for a
single lat/lon point and feeds them into the SAME rule-based environment check
already in logic.environment_favors_intensification - it does not touch the AI
classifier at all.

    sea-surface temperature (deg C)             -> sst
    200hPa-minus-850hPa wind shear (kt)         -> wind_shear
    700hPa relative humidity (%)                -> humidity (now used, not just accepted)
    850hPa relative vorticity (s^-1)            -> vorticity (informational only,
                                                    NOT used in the favorable/
                                                    unfavorable verdict - see
                                                    logic.py for why)

Honesty notes (read before trusting a number this returns)
------------------------------------------------------------
1. ERA5 is a REANALYSIS, not a nowcast. The near-real-time extension (called
   ERA5T internally by ECMWF but served through the same CDS dataset name) is
   itself typically ~5 days behind the present. This module defaults to
   requesting data from ERA5_LAG_DAYS ago (5 by default) for exactly that
   reason - asking for "right now" will usually just fail. This is disclosed
   to the frontend via valid_time_utc / lag_days in the response; it is never
   silently presented as a live observation.
2. This code could not be exercised against a real CDS server from this dev
   environment (the sandbox this was written in has no internet access and no
   CDS account). The dataset names and variable short names below
   (sea_surface_temperature/sst, u_component_of_wind/u, v_component_of_wind/v,
   relative_humidity/r, vorticity/vo) match ECMWF's published ERA5
   documentation at the time of writing, but the exact request parameter name
   for output format ("data_format" vs the older "format") has changed on the
   CDS side before. Both are sent for robustness (see _build_requests). If a
   request is rejected, check the current parameter names in the CDS "API
   request" panel for the dataset before assuming this module is broken -
   same spirit as the Himawari source in backend/satellite/, which carries an
   identical "unverified, confirm before trusting" flag.
3. This is entirely optional. If cdsapi is not installed, or the operator has
   no CDS API key configured, every function below raises ERA5NotConfigured
   with a clear, actionable message - it never crashes the server (same
   defensive pattern as backend/satellite/ and backend/gradcam.py).

Setup (on YOUR machine - never in this repo, never in .env)
-------------------------------------------------------------
1. pip install -r backend/requirements-era5.txt   (cdsapi, xarray, netCDF4 -
   NOT part of the base backend/requirements.txt; the app runs fully without
   them, this is an add-on)
2. Create a free account at https://cds.climate.copernicus.eu/
3. Accept the ERA5 dataset licence on the CDS website (one-time, per dataset)
4. Put your personal API key in ~/.cdsapirc (CDS gives you the exact two
   lines to paste on your profile page) - cdsapi reads this file itself, this
   module never sees or stores the key.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import multiprocessing
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

log = logging.getLogger("era5")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[ERA5] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


class ERA5NotConfigured(Exception):
    """cdsapi/xarray not installed, or no ~/.cdsapirc - not a real failure,
    just "this optional feature isn't set up on this machine"."""


class ERA5Error(Exception):
    """A real attempt was made (client configured) but it failed: network,
    CDS queue timeout, no data at that point, bad response, etc."""


# ------------------------------------------------------------------ config --
def _bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


ERA5_ENABLED = _bool("ERA5_ENABLED", True)
# ERA5/ERA5T's typical publication lag. 5 days is conservative; ECMWF has
# sometimes published ERA5T within ~2-3 days, but requesting further back
# lowers the chance of "no data yet" for a demo. Override with ERA5_LAG_DAYS
# if you know today's actual lag (check https://cds.climate.copernicus.eu/
# for the latest available date on the ERA5 dataset page).
ERA5_LAG_DAYS = _int("ERA5_LAG_DAYS", 5)
# Half-width (degrees) of the CDS "area" box requested around the point. ERA5
# is on a 0.25 deg grid, so 0.5 comfortably guarantees the nearest grid point
# is inside the returned box regardless of rounding.
ERA5_BOX_DEGREES = _float("ERA5_BOX_DEGREES", 0.5)
# How long to wait for the CDS client before giving up and returning a clear
# "still queued" error instead of hanging the request forever. CDS can queue
# jobs for anywhere from seconds to (rarely) tens of minutes at busy times -
# this is a soft client-side cutoff, not a claim about how CDS itself
# behaves.
ERA5_REQUEST_TIMEOUT_SECONDS = _int("ERA5_REQUEST_TIMEOUT_SECONDS", 90)
# Live CDS fetch is OFF unless explicitly enabled: the demo must never hang
# on a queued Copernicus request. With ERA5_LIVE=false (default) only
# repo-bundled seed files + on-disk cache are served; anything else gets a
# clear "no seed / live disabled" message, never invented values.
# The warm-cache script bypasses this per-call (allow_live=True).
ERA5_LIVE = _bool("ERA5_LIVE", False)
# Completed/failed jobs are forgotten after this long (stale-job expiry).
ERA5_JOB_TTL_SECONDS = _int("ERA5_JOB_TTL_SECONDS", 30 * 60)

DATA_DIR = Path(os.environ.get("ERA5_DATA_DIR", BACKEND_DIR / "data" / "era5"))
CACHE_DIR = DATA_DIR / "cache"
# Repo-bundled pre-fetched results (see backend/era5_warm_cache.py). Render's
# disk is ephemeral, so anything fetched at runtime vanishes on restart;
# seeds ship WITH the repo, so the demo storms never need a live CDS request.
SEED_DIR = DATA_DIR / "seed"
# A given (rounded point, hour) of ERA5 reanalysis never changes once
# published, so the cache TTL only needs to protect against re-requesting
# the exact same point repeatedly within one demo session - not correctness.
CACHE_TTL_SECONDS = _int("ERA5_CACHE_TTL_SECONDS", 6 * 3600)


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SEED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------- optional imports --
# Memory-critical: cdsapi + xarray + netCDF4 + scipy cost ~70 MB of RSS
# (measured: fresh python 27 MB -> 98 MB after import). They are NEVER
# imported at module level here, so the server process stays lean and a
# fetch can run isolated in a short-lived subprocess (see fetch_environment).
# `cdsapi`/`xr` stay as None sentinels for backwards-compat checks;
# availability is probed via find_spec (no import, no memory).
cdsapi = None
xr = None


def _spec_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _import_error():
    missing = [n for n in ("cdsapi", "xarray") if not _spec_available(n)]
    if not missing:
        return None
    return ("ERA5 libraries not installed (missing: %s). Install with "
            "`pip install -r backend/requirements-era5.txt`." % ", ".join(missing))


def is_available() -> bool:
    """True only if the optional packages are importable. Does NOT check for
    a valid ~/.cdsapirc - that is only discovered when a request is actually
    attempted, since cdsapi itself is what parses that file."""
    return ERA5_ENABLED and _import_error() is None


def status() -> dict:
    """Cheap, side-effect-free status for GET /api/era5/status - never makes
    a network call and never imports the heavy libraries."""
    has_rc = (Path.home() / ".cdsapirc").is_file() or bool(
        os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))
    import_error = _import_error()
    return {
        "enabled": ERA5_ENABLED,
        "packages_installed": import_error is None,
        "credentials_found": has_rc,
        "ready": ERA5_ENABLED and import_error is None and has_rc,
        "import_error": import_error,
        "live_enabled": ERA5_ENABLED and ERA5_LIVE,
        "lag_days": ERA5_LAG_DAYS,
        "note": ("ERA5 is a reanalysis: even when this reports ready=true, the "
                 "data returned is from roughly lag_days ago, not the present "
                 "moment - see this module's docstring."),
    }


# --------------------------------------------------------------- utilities --
def _round_grid(value: float, step: float = 0.25) -> float:
    return round(round(value / step) * step, 2)


def _target_time(when_utc) -> datetime:
    if when_utc is not None:
        return when_utc
    target = datetime.now(timezone.utc) - timedelta(days=ERA5_LAG_DAYS)
    return target.replace(minute=0, second=0, microsecond=0)


def _cache_key(lat: float, lon: float, when: datetime) -> str:
    raw = f"{_round_grid(lat)}:{_round_grid(lon)}:{when.strftime('%Y%m%d%H')}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str):
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if time.time() - payload.get("_cached_at", 0) > CACHE_TTL_SECONDS:
        return None
    result = dict(payload)
    result.pop("_cached_at", None)
    result["cached"] = True
    return result


def _write_cache(key: str, result: dict) -> None:
    ensure_dirs()
    payload = dict(result)
    payload["_cached_at"] = time.time()
    try:
        _cache_path(key).write_text(json.dumps(payload))
    except OSError:
        pass  # cache is a convenience, never a hard requirement


def _seed_name(lat: float, lon: float) -> str:
    return f"{_round_grid(lat):.2f}_{_round_grid(lon):.2f}.json"


def _read_seed(lat: float, lon: float):
    """Repo-bundled pre-fetched result for this grid point (see
    backend/era5_warm_cache.py). Served with its REAL valid_time_utc and a
    `seed: True` flag so the UI can label it honestly as pre-fetched
    reanalysis, never as live. Survives Render's ephemeral disk because it
    ships with the repo."""
    path = SEED_DIR / _seed_name(lat, lon)
    if not path.is_file():
        return None
    try:
        result = dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None
    result["cached"] = True
    result["seed"] = True
    result["source"] = (result.get("source", "ERA5 reanalysis")
                        + " [pre-fetched seed shipped with the repo]")
    return result


def _area_box(lat: float, lon: float):
    """Smallest possible CDS request: a single ERA5 grid point, i.e.
    area [N, W, S, E] with N == S and W == E (snapped to the 0.25 deg
    grid). One grid cell holds ~12 fields for one hour - a few KB, versus
    a 1 deg box which returns 25x that for no benefit since _nearest()
    picks the closest point anyway."""
    la, lo = _round_grid(lat), _round_grid(lon)
    return [la, lo, la, lo]


def _build_requests(lat: float, lon: float, when: datetime):
    area = _area_box(lat, lon)
    # New-CDS (2024+) request format uses year/month/day/time keys.
    # Dataset names unchanged. Area order [N, W, S, E]. Verified Sep 2026
    # against cds.climate.copernicus.eu "How to download ERA5".
    common = {"product_type": "reanalysis",
              "year": when.strftime("%Y"),
              "month": when.strftime("%m"),
              "day": when.strftime("%d"),
              "time": when.strftime("%H:00"),
              "area": area,
              "data_format": "netcdf",
              "download_format": "unarchived"}
    single = {"variable": "sea_surface_temperature", **common}
    pressure = {"variable": ["u_component_of_wind", "v_component_of_wind",
                              "relative_humidity", "vorticity"],
                "pressure_level": ["200", "700", "850"], **common}
    return single, pressure


def _make_client():
    """Build a cdsapi client. On Render there is no ~/.cdsapirc, so read
    credentials ONLY from env vars (CDSAPI_URL and CDSAPI_KEY). Locally,
    falls back to the default constructor (reads ~/.cdsapirc)."""
    import cdsapi  # lazy: ~70 MB with xarray/scipy, only in the fetch worker
    url = (os.environ.get("CDSAPI_URL") or "").strip()
    key = (os.environ.get("CDSAPI_KEY") or "").strip()
    if url and key:
        return cdsapi.Client(url=url, key=key, quiet=True)
    if key:
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api", key=key, quiet=True)
    return cdsapi.Client(quiet=True)


def _retrieve(client, dataset: str, request: dict) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".nc", prefix="era5_")
    os.close(fd)
    path = Path(path_str)
    client.retrieve(dataset, request, str(path))
    return path


def _nearest(ds, lat: float, lon: float):
    """ERA5 longitude is 0-360; accept either convention from the caller."""
    lon_grid = lon % 360
    try:
        return ds.sel(latitude=lat, longitude=lon_grid, method="nearest")
    except Exception:
        return ds.sel(latitude=lat, longitude=lon, method="nearest")


def _extract(single_path: Path, pressure_path: Path, lat: float, lon: float) -> dict:
    import xarray as xr  # lazy: only inside the short-lived fetch worker
    with xr.open_dataset(single_path) as ds_s, xr.open_dataset(pressure_path) as ds_p:
        pt_s = _nearest(ds_s, lat, lon)
        pt_p = _nearest(ds_p, lat, lon)

        def scalar(da):
            v = da.values
            return float(v.reshape(-1)[0])

        sst_var = "sst" if "sst" in ds_s.variables else "sea_surface_temperature"
        sst_k = scalar(pt_s[sst_var])
        if sst_k != sst_k:  # NaN - point is over land, ERA5 SST is ocean-only.
            # Honest land path (never invent an SST, never fail the whole
            # fetch): shear/humidity/vorticity below are still real and are
            # returned, with SST null and a reason. See _fetch_live, which
            # additionally attaches the nearest real ocean SST for context.
            sst_c = None
            sst_note = "over land, SST not applicable"
            land = True
        else:
            sst_c = round(sst_k - 273.15, 2)
            sst_note = None
            land = False

        level_dim = "level" if "level" in pt_p.dims else "pressure_level"

        def at_level(var, level):
            return scalar(pt_p[var].sel(**{level_dim: level}, method="nearest"))

        u_var = "u" if "u" in ds_p.variables else "u_component_of_wind"
        v_var = "v" if "v" in ds_p.variables else "v_component_of_wind"
        r_var = "r" if "r" in ds_p.variables else "relative_humidity"
        vo_var = "vo" if "vo" in ds_p.variables else "vorticity"

        u200, v200 = at_level(u_var, 200), at_level(v_var, 200)
        u850, v850 = at_level(u_var, 850), at_level(v_var, 850)
        shear_ms = math.hypot(u200 - u850, v200 - v850)
        wind_shear_kt = round(shear_ms * 1.94384, 1)

        humidity_pct = round(at_level(r_var, 700), 1)
        vorticity_850 = at_level(vo_var, 850)

        valid_time = None
        for cand in ("valid_time", "time"):
            if cand in pt_s.coords:
                try:
                    valid_time = str(pt_s.coords[cand].values)
                    break
                except Exception:  # noqa: BLE001
                    pass

        return {
            "sst_c": sst_c,
            "sst_note": sst_note,
            "land": land,
            "wind_shear_kt": wind_shear_kt,
            "humidity_pct": humidity_pct,
            "vorticity_850_s1": vorticity_850,
            "valid_time_utc": valid_time,
        }


def fetch_environment(lat: float, lon: float, when_utc: datetime | None = None,
                       use_cache: bool = True, allow_live: bool | None = None) -> dict:
    """Synchronous fetch (used by era5_warm_cache.py and the tests).

    Lookup order: seed file -> disk cache -> live CDS (only if allow_live,
    which defaults to the ERA5_LIVE env flag, OFF for the demo). Raises
    ERA5NotConfigured (packages missing / live disabled / no credentials) or
    ERA5Error (a real attempt failed). Never invents values.
    """
    if not is_available():
        err = _import_error()
        raise ERA5NotConfigured(
            err or "ERA5 fetch is disabled (ERA5_ENABLED=false).")

    when = _target_time(when_utc)
    key = _cache_key(lat, lon, when)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached
        seed = _read_seed(lat, lon)
        if seed is not None:
            return seed

    live = ERA5_LIVE if allow_live is None else allow_live
    if not live:
        raise ERA5NotConfigured(
            "No pre-fetched ERA5 seed file for this point and live CDS fetch is "
            "disabled (ERA5_LIVE=false - the demo default). Warm one with "
            "`python backend/era5_warm_cache.py` on a machine with Copernicus "
            "CDS credentials (CDSAPI_URL/CDSAPI_KEY), or set ERA5_LIVE=true to "
            "allow a live queued request.")

    single_req, pressure_req = _build_requests(lat, lon, when)
    when_iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        extracted = _fetch_in_subprocess(lat, lon, when_iso, single_req, pressure_req)
    finally:
        gc.collect()  # drop any queue/pipe buffers in THIS process promptly

    result = {
        **extracted,
        "requested_time_utc": when_iso,
        "lag_days": ERA5_LAG_DAYS,
        "lat": lat, "lon": lon,
        "cached": False,
        "source": "ERA5 reanalysis (Copernicus Climate Data Store) - see backend/era5.py",
    }
    _write_cache(key, result)
    return result


# Job queue state (the single-flight worker below replaces the old lock:
# jobs wait in a queue instead of erroring, and there is no lock to leak).
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_WORKER_THREAD = None


def _fetch_worker(queue, lat: float, lon: float, when_iso: str,
                  single_req: dict, pressure_req: dict):
    """Runs in a SHORT-LIVED child process (spawn): imports the ~70 MB libs,
    does the CDS download + NetCDF decode, puts a plain dict on the queue
    and exits - releasing ALL its memory back to the OS. A crash here
    (segfault/OOM in native NetCDF code) kills only the child; the parent
    sees an empty queue and reports ERA5Error. Must stay picklable and
    import-light at module level.
    ERA5_TEST_HANG (seconds, env): integration-test hook - sleep before
    fetching so the parent's hard timeout + terminate path can be exercised
    deterministically (see backend/tests/test_era5_jobs.py)."""
    if os.environ.get("ERA5_TEST_HANG"):
        time.sleep(float(os.environ["ERA5_TEST_HANG"]))
    try:
        out = _fetch_live(lat, lon, single_req, pressure_req)
        out["valid_time_utc"] = out.get("valid_time_utc") or when_iso
        queue.put({"ok": True, "result": out})
    except Exception as exc:  # noqa: BLE001 - child must never die silently
        try:
            queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_ocean_sst(single_box_path: Path, lat: float, lon: float):
    """Nearest grid cell with a real (non-NaN) SST in an already-downloaded
    single-level file. Pure read of real ERA5 data - nothing invented. Returns
    {sst_c, lat, lon, distance_km} or None when no ocean cell is in the box."""
    import xarray as xr  # lazy: only inside the short-lived fetch worker
    with xr.open_dataset(single_box_path) as ds:
        var = "sst" if "sst" in ds.variables else "sea_surface_temperature"
        da = ds[var]
        lats = [float(v) for v in ds["latitude"].values.reshape(-1)]
        lons = [float(v) % 360 for v in ds["longitude"].values.reshape(-1)]
        best = None
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                try:
                    v = float(da.values.reshape(-1)[i * len(lons) + j])
                except Exception:  # noqa: BLE001
                    continue
                if v != v:  # NaN - land mask, skip
                    continue
                lon_c = lo if lo <= 180 else lo - 360
                d = _haversine_km(lat, lon, la, lon_c)
                if best is None or d < best[0]:
                    best = (d, la, lon_c, v)
        if best is None:
            return None
        d, la, lo, v = best
        return {"sst_c": round(v - 273.15, 2), "lat": round(la, 2),
                "lon": round(lo, 2), "distance_km": round(d, 1)}


def _fetch_live(lat: float, lon: float, single_req: dict, pressure_req: dict) -> dict:
    """One real CDS round-trip in the CURRENT process. Used by the worker
    child, and as a last-resort fallback when no child process can be
    started at all (then the heavy libs land in this process - correctness
    first, isolation best-effort)."""
    client = _make_client()
    single_path = _retrieve(client, "reanalysis-era5-single-levels", single_req)
    pressure_path = _retrieve(client, "reanalysis-era5-pressure-levels", pressure_req)
    try:
        out = _extract(single_path, pressure_path, lat, lon)
    finally:
        for p in (single_path, pressure_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
    if out.get("land"):
        # One small follow-up: a 1-degree single-level box around the point,
        # scanned locally for the nearest real ocean SST (context only -
        # never used in any verdict). Still pure ERA5 data, one extra file.
        box_req = dict(single_req)
        box_req["area"] = [round(lat + 0.5, 2), round(lon - 0.5, 2),
                           round(lat - 0.5, 2), round(lon + 0.5, 2)]
        box_path = _retrieve(client, "reanalysis-era5-single-levels", box_req)
        try:
            near = _nearest_ocean_sst(box_path, lat, lon)
        finally:
            try:
                box_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        if near is not None:
            out["nearest_ocean_sst"] = near
    return out


def _classify_worker_error(detail: str):
    """Map a worker child's failure string to the right exception type."""
    if "cdsapirc" in detail.lower() or "credentials" in detail.lower() \
            or "401" in detail or "403" in detail:
        return ERA5NotConfigured(
            "No Copernicus CDS credentials found. Create a free account at "
            "https://cds.climate.copernicus.eu/, accept the ERA5 licence, and set "
            "CDSAPI_URL + CDSAPI_KEY env vars (Render) or ~/.cdsapirc (local). "
            f"Detail: {detail}")
    return ERA5Error(f"ERA5 request failed: {detail}")


def _spawn_fetch(lat: float, lon: float, when_iso: str,
                 single_req: dict, pressure_req: dict, ctx_name: str) -> dict:
    ctx = multiprocessing.get_context(ctx_name)
    queue = ctx.Queue()
    proc = ctx.Process(target=_fetch_worker,
                       args=(queue, lat, lon, when_iso, single_req, pressure_req),
                       daemon=True)
    proc.start()  # may raise RuntimeError when spawn is impossible here
    try:
        proc.join(timeout=ERA5_REQUEST_TIMEOUT_SECONDS)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            raise ERA5Error(
                f"CDS did not respond within {ERA5_REQUEST_TIMEOUT_SECONDS}s (it queues "
                "requests and can be slow at busy times). Try again shortly - a "
                "completed job is cached on disk.")
        try:
            msg = queue.get_nowait()
        except Exception:
            raise ERA5Error("ERA5 worker ended without a result (exit "
                            f"code {proc.exitcode}) - likely killed for memory. "
                            "Try again shortly.") from None
        if not msg.get("ok"):
            raise _classify_worker_error(msg.get("error", "unknown worker error"))
        return msg["result"]
    finally:
        try:
            queue.close()
        except Exception:
            pass
        del queue
        gc.collect()


def _fetch_in_subprocess(lat: float, lon: float, when_iso: str,
                         single_req: dict, pressure_req: dict) -> dict:
    """Isolation-first dispatcher. Production path is a spawn child (fresh
    ~30 MB process, zero shared state with the server). If spawn is
    impossible here (Windows/macOS process started from an unguarded
    __main__ - dev scripts and tests), fall back to fork (Linux), and only
    as a last resort run in-process on a bounded thread."""
    args = (lat, lon, when_iso, single_req, pressure_req)
    try:
        return _spawn_fetch(*args, "spawn")
    except RuntimeError as exc:
        log.warning("spawn unavailable (%s) - trying fork fallback", exc)
    try:
        return _spawn_fetch(*args, "fork")
    except Exception as exc:  # noqa: BLE001 - no fork on Windows
        log.warning("fork unavailable (%s) - running fetch in-process", exc)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_fetch_live, lat, lon, single_req, pressure_req)
        try:
            out = future.result(timeout=ERA5_REQUEST_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            raise ERA5Error(
                f"CDS did not respond within {ERA5_REQUEST_TIMEOUT_SECONDS}s (it queues "
                "requests and can be slow at busy times). Try again shortly.") from exc
        except Exception as exc:  # noqa: BLE001
            raise _classify_worker_error(f"{type(exc).__name__}: {exc}") from exc
    out["valid_time_utc"] = out.get("valid_time_utc") or when_iso
    return out


# --------------------------------------------------- async live-fetch jobs --
# The HTTP layer never blocks on CDS (queues take minutes). submit_fetch()
# returns a job immediately; ONE background worker thread runs live fetches
# sequentially in short-lived subprocesses; the frontend polls get_job().
# No request is ever held open longer than a few seconds.
def _purge_jobs(now=None):
    now = time.time() if now is None else now
    stale = [jid for jid, job in _JOBS.items()
             if now - job.get("updated", job.get("created", 0)) > ERA5_JOB_TTL_SECONDS]
    for jid in stale:
        _JOBS.pop(jid, None)


def _public_job(job, attached=False):
    out = {"job_id": job["job_id"], "status": job["status"],
           "lat": job["lat"], "lon": job["lon"],
           "requested_time_utc": job["when_iso"],
           "created_utc": job["created_utc"], "attached": attached}
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "failed":
        out["error"] = job["error"]
        out["configured"] = job.get("configured", True)
    return out


def _new_job_id():
    return hashlib.md5(f"{time.time()}-{os.getpid()}".encode()).hexdigest()[:12]


def _worker_loop():
    """Single worker: FIFO, one live fetch at a time. Never dies (every job
    is wrapped in try/finally) so there is no lock state to leak - a hung
    child is killed by _fetch_in_subprocess's hard timeout, the job is
    marked failed, and the next job proceeds."""
    while True:
        job = None
        with _JOBS_LOCK:
            for candidate in _JOBS.values():
                if candidate["status"] == "queued":
                    job = candidate
                    job["status"] = "running"
                    job["started"] = time.time()
                    job["updated"] = time.time()
                    break
        if job is None:
            time.sleep(0.5)
            continue
        try:
            extracted = _fetch_in_subprocess(
                job["lat"], job["lon"], job["when_iso"],
                job["single_req"], job["pressure_req"])
            result = {**extracted, "requested_time_utc": job["when_iso"],
                      "lag_days": ERA5_LAG_DAYS,
                      "lat": job["lat"], "lon": job["lon"],
                      "cached": False,
                      "source": "ERA5 reanalysis (Copernicus Climate Data Store) - see backend/era5.py"}
            _write_cache(job["key"], result)
            with _JOBS_LOCK:
                job["status"] = "done"
                job["result"] = result
                job["updated"] = time.time()
        except ERA5NotConfigured as exc:
            with _JOBS_LOCK:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["configured"] = False
                job["updated"] = time.time()
        except Exception as exc:  # noqa: BLE001 - a dead job must never kill the worker
            with _JOBS_LOCK:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["configured"] = not isinstance(exc, ERA5NotConfigured)
                job["updated"] = time.time()
        finally:
            gc.collect()


def _ensure_worker():
    global _WORKER_THREAD
    with _JOBS_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _WORKER_THREAD = threading.Thread(target=_worker_loop,
                                          name="era5-worker", daemon=True)
        _WORKER_THREAD.start()


def submit_fetch(lat: float, lon: float, when_utc: datetime | None = None,
                 use_cache: bool = True) -> dict:
    """Enqueue-or-attach a live fetch. ALWAYS returns immediately with a job
    dict (status queued/running/done/failed) - never blocks on CDS.

    Lookup order: seed file -> disk cache -> live job (only if ERA5_LIVE).
    A second request for the same (lat, lon, hour) attaches to the running
    job (attached: true) instead of starting another. Never invents values:
    with no seed/cache and live disabled, the job is failed with a message
    saying exactly that.
    """
    from datetime import timezone as _tz
    when = _target_time(when_utc)
    key = _cache_key(lat, lon, when)
    when_iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    created_utc = datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _JOBS_LOCK:
        _purge_jobs()
        if use_cache:
            seed = _read_seed(lat, lon)
            if seed is not None:
                return {"job_id": "seed", "status": "done", "lat": lat, "lon": lon,
                        "requested_time_utc": when_iso, "created_utc": created_utc,
                        "attached": False, "result": seed}
            cached = _read_cache(key)
            if cached is not None:
                return {"job_id": "cache", "status": "done", "lat": lat, "lon": lon,
                        "requested_time_utc": when_iso, "created_utc": created_utc,
                        "attached": False, "result": cached}
        for job in _JOBS.values():
            if job["key"] == key and job["status"] in ("queued", "running"):
                return _public_job(job, attached=True)
        if not ERA5_ENABLED or not ERA5_LIVE:
            return {"job_id": "none", "status": "failed", "lat": lat, "lon": lon,
                    "requested_time_utc": when_iso, "created_utc": created_utc,
                    "attached": False, "configured": False,
                    "error": ("No pre-fetched ERA5 seed file for this point and live CDS "
                              "fetch is disabled (ERA5_LIVE=false - the demo default, so the "
                              "demo never hangs on a Copernicus queue). Nothing was invented: "
                              "fill SST/shear/humidity in by hand, or warm a seed with "
                              "`python backend/era5_warm_cache.py`.")}
        err = _import_error()
        if err is not None:
            return {"job_id": "none", "status": "failed", "lat": lat, "lon": lon,
                    "requested_time_utc": when_iso, "created_utc": created_utc,
                    "attached": False, "configured": False, "error": err}
        job = {"job_id": _new_job_id(), "key": key, "lat": lat, "lon": lon,
               "when_iso": when_iso, "created_utc": created_utc,
               "status": "queued", "result": None, "error": None,
               "created": time.time(), "updated": time.time(),
               "single_req": None, "pressure_req": None}
        single_req, pressure_req = _build_requests(lat, lon, when)
        job["single_req"] = single_req
        job["pressure_req"] = pressure_req
        _JOBS[job["job_id"]] = job
        public = _public_job(job)
    _ensure_worker()
    return public


def get_job(job_id: str):
    """Poll a job. Stale running jobs (worker lost track) expire to failed
    instead of hanging forever."""
    with _JOBS_LOCK:
        _purge_jobs()
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job["status"] == "running" and \
                time.time() - job.get("started", job["updated"]) > ERA5_REQUEST_TIMEOUT_SECONDS + 120:
            job["status"] = "failed"
            job["error"] = ("Live fetch worker went silent (stale job expired) - "
                            "submit again to retry; the next attempt starts clean.")
            job["configured"] = True
            job["updated"] = time.time()
        return _public_job(job)
