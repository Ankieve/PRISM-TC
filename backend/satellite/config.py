"""Configuration for the satellite module, read from environment variables
(and a local .env file if python-dotenv style parsing is available - we
avoid a hard dependency on the python-dotenv package and parse it ourselves
with the standard library, since it is a tiny format).

Never hard-code credentials here. Everything that can be secret comes from
the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR.parent / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, # comments). Does not
    override variables already set in the real environment, so a value
    exported in the shell always wins over the file."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ENV_FILE)


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- general ---
SATELLITE_ENABLED = _bool("SATELLITE_ENABLED", True)
# Which source class to use as primary. "imd_insat" (India's own INSAT-3D/3DS
# via IMD's public site) is the default: it is free, needs no account, and
# - unlike Himawari - its single frame covers the WHOLE North Indian Ocean
# (Arabian Sea and Bay of Bengal both) in one image. See satellite/README.md
# "Why IMD/INSAT instead of Himawari" for the full story.
SATELLITE_SOURCE = os.environ.get("SATELLITE_SOURCE", "imd_insat")
# Fallback source, tried only if the primary fails outright. Leave blank to
# disable fallback. "himawari_nict" remains here but is UNVERIFIED - its
# exact current tile URL scheme could not be confirmed live (see README) -
# so it is not enabled as the default fallback. "eumetsat_iodc" needs
# EUMETSAT_* credentials below.
SATELLITE_FALLBACK_SOURCE = os.environ.get("SATELLITE_FALLBACK_SOURCE", "")

# IMD's INSAT-3D/3DS Asia-sector quicklook does not publish a documented
# refresh cadence, but INSAT-3D/3DS's normal full-disk scan interval is
# around 30 minutes. Polling every 60 seconds would just re-fetch the same
# unchanged file repeatedly - poll less often than that.
UPDATE_INTERVAL_SECONDS = _int("UPDATE_INTERVAL", 300)  # 5 minutes

# How old the latest successfully processed observation can be before the
# frontend must show STALE instead of LIVE. ~30 min scan cadence plus
# publication latency plus polling slack.
STALE_AFTER_SECONDS = _int("SATELLITE_STALE_AFTER_SECONDS", 60 * 40)  # 40 min

REQUEST_TIMEOUT_SECONDS = _int("SATELLITE_REQUEST_TIMEOUT", 20)
DOWNLOAD_RETRIES = _int("SATELLITE_DOWNLOAD_RETRIES", 3)

# The ML model (EfficientNet-B0) was trained only on TCIR chips: a single
# IR channel, brightness-temperature-like grayscale, ~201x201 px, tightly
# cropped and centred on a named storm, with NO documented geolocation for
# the crop other than "centred on the storm". A near-real-time full-disk
# picture is a different product (see satellite/README.md "Model
# compatibility" section for the full reasoning). Because of that mismatch,
# running predict() on live imagery is scientifically unverified and is
# OFF by default. Flip this only once you understand and accept that the
# result is not validated the way the TCIR-based prediction is.
ENABLE_LIVE_INFERENCE = _bool("ENABLE_LIVE_INFERENCE", False)

# North Indian Ocean bounding box (degrees). Covers the Arabian Sea and the
# Bay of Bengal with margin. Adjust in .env if you want a tighter/looser box.
REGION_LAT_MIN = float(os.environ.get("SATELLITE_LAT_MIN", "-5"))
REGION_LAT_MAX = float(os.environ.get("SATELLITE_LAT_MAX", "30"))
REGION_LON_MIN = float(os.environ.get("SATELLITE_LON_MIN", "40"))
REGION_LON_MAX = float(os.environ.get("SATELLITE_LON_MAX", "100"))

# ------------------------------------------------------------- IMD/INSAT ---
# India Meteorological Department's public Asia-sector Thermal Infrared-1
# quicklook from INSAT-3D/3DS. A single static image, no login, no query
# parameters. Confirmed reachable manually on 2026-09-23 (see README).
IMD_INSAT_IR1_URL = os.environ.get(
    "IMD_INSAT_IR1_URL", "https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg")

# ---------------------------------------------------------- Himawari/NICT ---
# UNVERIFIED (see README "Why IMD/INSAT instead of Himawari"): NICT's
# real-time mirror's exact current tile-file naming could not be confirmed
# live from a real browser during this build, so this source is kept for
# reference/future work but is not the default primary or fallback. This is
# a rendered full-disk quicklook product, not raw calibrated data - see
# README for what that means for accuracy, even once the URL scheme issue
# is resolved.
HIMAWARI_BASE_URL = os.environ.get(
    "HIMAWARI_BASE_URL", "https://himawari8-dl.nict.go.jp/himawari8/img/D531106")
# Tile grid level: "4d" = 4x4 grid of tiles (16 tiles). Higher levels exist
# (8d, 20d) but cost more bandwidth/time per poll; 4d is enough to identify
# cloud structure over the North Indian Ocean region for a dashboard.
HIMAWARI_LEVEL = os.environ.get("HIMAWARI_LEVEL", "4d")
HIMAWARI_TILE_PX = _int("HIMAWARI_TILE_PX", 550)

# ---------------------------------------------------------- EUMETSAT IODC ---
# Optional fallback. Requires a FREE EUMETSAT Earth Observation Portal
# account (https://eoportal.eumetsat.int/) - registration is free, no card,
# but it is not anonymous like the Himawari source above. Leave blank to
# skip this source entirely.
EUMETSAT_CONSUMER_KEY = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
EUMETSAT_CONSUMER_SECRET = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")
# EUMETSAT Data Store collection id for the product you want. We do not
# hard-code a guessed id: confirm the exact id for "Meteosat-8 (IODC) Level
# 1.5 image data" on https://data.eumetsat.int and put it here. Implemented
# but UNVERIFIED from this environment - see README.
EUMETSAT_COLLECTION_ID = os.environ.get("EUMETSAT_COLLECTION_ID", "")

# ---------------------------------------------------------------- MOSDAC ---
# Deliberately NOT implemented as a live automated source. See
# satellite/sources/mosdac.py for exactly why, and what a teammate with
# MOSDAC credentials would need to finish it.
MOSDAC_USERNAME = os.environ.get("MOSDAC_USERNAME", "")
MOSDAC_PASSWORD = os.environ.get("MOSDAC_PASSWORD", "")

# ------------------------------------------------------------- storage ----
DATA_DIR = Path(os.environ.get("SATELLITE_DATA_DIR", BACKEND_DIR / "data" / "satellite"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LATEST_DIR = DATA_DIR / "latest"
METADATA_DIR = DATA_DIR / "metadata"
HISTORY_FILE = METADATA_DIR / "history.json"
STATE_FILE = METADATA_DIR / "state.json"
MAX_HISTORY_ENTRIES = _int("SATELLITE_MAX_HISTORY", 200)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, LATEST_DIR, METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
