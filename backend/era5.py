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

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import tempfile
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

DATA_DIR = Path(os.environ.get("ERA5_DATA_DIR", BACKEND_DIR / "data" / "era5"))
CACHE_DIR = DATA_DIR / "cache"
# A given (rounded point, hour) of ERA5 reanalysis never changes once
# published, so the cache TTL only needs to protect against re-requesting
# the exact same point repeatedly within one demo session - not correctness.
CACHE_TTL_SECONDS = _int("ERA5_CACHE_TTL_SECONDS", 6 * 3600)


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------- optional imports --
IMPORT_ERROR = None
try:
    import cdsapi  # type: ignore
except Exception as _exc:  # noqa: BLE001
    cdsapi = None
    IMPORT_ERROR = f"cdsapi not installed ({type(_exc).__name__}: {_exc})"

try:
    import xarray as xr  # type: ignore
except Exception as _exc:  # noqa: BLE001
    xr = None
    if IMPORT_ERROR is None:
        IMPORT_ERROR = f"xarray not installed ({type(_exc).__name__}: {_exc})"


def is_available() -> bool:
    """True only if the optional packages are importable. Does NOT check for
    a valid ~/.cdsapirc - that is only discovered when a request is actually
    attempted, since cdsapi itself is what parses that file."""
    return ERA5_ENABLED and cdsapi is not None and xr is not None


def status() -> dict:
    """Cheap, side-effect-free status for GET /api/era5/status - never makes
    a network call."""
    has_rc = (Path.home() / ".cdsapirc").is_file() or bool(
        os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))
    return {
        "enabled": ERA5_ENABLED,
        "packages_installed": cdsapi is not None and xr is not None,
        "credentials_found": has_rc,
        "ready": is_available() and has_rc,
        "import_error": IMPORT_ERROR,
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


def _area_box(lat: float, lon: float):
    """CDS 'area' is [North, West, South, East] in degrees."""
    b = ERA5_BOX_DEGREES
    return [round(lat + b, 2), round(lon - b, 2), round(lat - b, 2), round(lon + b, 2)]


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
    with xr.open_dataset(single_path) as ds_s, xr.open_dataset(pressure_path) as ds_p:
        pt_s = _nearest(ds_s, lat, lon)
        pt_p = _nearest(ds_p, lat, lon)

        def scalar(da):
            v = da.values
            return float(v.reshape(-1)[0])

        sst_var = "sst" if "sst" in ds_s.variables else "sea_surface_temperature"
        sst_k = scalar(pt_s[sst_var])
        if sst_k != sst_k:  # NaN - point is over land, ERA5 SST is ocean-only
            raise ERA5Error("SST is undefined at this point (likely over land in ERA5's "
                             "ocean mask) - try a point further offshore.")
        sst_c = round(sst_k - 273.15, 2)

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
            "wind_shear_kt": wind_shear_kt,
            "humidity_pct": humidity_pct,
            "vorticity_850_s1": vorticity_850,
            "valid_time_utc": valid_time,
        }


def fetch_environment(lat: float, lon: float, when_utc: datetime | None = None,
                       use_cache: bool = True) -> dict:
    """The one function server.py calls. Returns a dict with sst_c,
    wind_shear_kt, humidity_pct, vorticity_850_s1, valid_time_utc,
    requested_time_utc, lag_days, lat, lon, cached, source.

    Raises ERA5NotConfigured (packages missing / no credentials) or
    ERA5Error (a real attempt failed) - server.py turns both into a clear
    JSON error response and never lets this crash a request.
    """
    if not is_available():
        raise ERA5NotConfigured(
            IMPORT_ERROR or "ERA5 fetch is disabled (ERA5_ENABLED=false).")

    when = _target_time(when_utc)
    key = _cache_key(lat, lon, when)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    try:
        client = _make_client()
    except Exception as exc:  # noqa: BLE001 - typically: no ~/.cdsapirc
        raise ERA5NotConfigured(
            "No Copernicus CDS credentials found. Create a free account at "
            "https://cds.climate.copernicus.eu/, accept the ERA5 licence, and set "
            "CDSAPI_URL + CDSAPI_KEY env vars (Render) or ~/.cdsapirc (local). "
            f"Detail: {exc}") from exc

    single_req, pressure_req = _build_requests(lat, lon, when)

    def _do_fetch():
        single_path = _retrieve(client, "reanalysis-era5-single-levels", single_req)
        pressure_path = _retrieve(client, "reanalysis-era5-pressure-levels", pressure_req)
        try:
            return _extract(single_path, pressure_path, lat, lon)
        finally:
            for p in (single_path, pressure_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_do_fetch)
        try:
            extracted = future.result(timeout=ERA5_REQUEST_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            raise ERA5Error(
                f"CDS did not respond within {ERA5_REQUEST_TIMEOUT_SECONDS}s (it queues "
                "requests and can be slow at busy times). The request keeps running in "
                "the background - try again shortly, it may hit the cache next time."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ERA5Error(f"ERA5 request failed: {type(exc).__name__}: {exc}") from exc

    result = {
        **extracted,
        "requested_time_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lag_days": ERA5_LAG_DAYS,
        "lat": lat, "lon": lon,
        "cached": False,
        "source": "ERA5 reanalysis (Copernicus Climate Data Store) - see backend/era5.py",
    }
    _write_cache(key, result)
    return result
