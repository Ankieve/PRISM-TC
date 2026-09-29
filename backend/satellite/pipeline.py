"""Orchestrates one check-and-update cycle:

    check newest available observation
    compare timestamp with last processed observation
    download only if new
    validate
    preprocess (crop to North Indian Ocean)
    run inference (only if ENABLE_LIVE_INFERENCE)
    store result
    update frontend state

Never raises - a failure at any stage is caught, logged, and written into
state.json as status=ERROR with the failing stage, so the rest of the app
(the existing, working /api/predict and /api/demo) is completely unaffected
if the satellite feature breaks.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from PIL import Image, UnidentifiedImageError

from . import config, inference_bridge, region, storage
from .inference_bridge import InferenceUnavailable
from .sources.base import NotConfiguredError, SourceError
from .sources.eumetsat_iodc import EumetsatIodcSource
from .sources.himawari_nict import HimawariNictSource
from .sources.imd_insat import IMDInsatSource
from .sources.mosdac import MosdacSource

log = logging.getLogger("satellite")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Satellite] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

_SOURCES = {
    "imd_insat": IMDInsatSource,
    "himawari_nict": HimawariNictSource,
    "eumetsat_iodc": EumetsatIodcSource,
    "mosdac": MosdacSource,
}


def _build_source(name: str):
    cls = _SOURCES.get(name)
    if cls is None:
        raise NotConfiguredError(f"Unknown SATELLITE_SOURCE '{name}'.")
    return cls()


def _is_stale(observation_time_utc: str) -> bool:
    try:
        obs = datetime.strptime(observation_time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return True
    age = (datetime.now(timezone.utc) - obs).total_seconds()
    return age > config.STALE_AFTER_SECONDS


def _validate_image(image: Image.Image) -> None:
    if image.width < 100 or image.height < 100:
        raise SourceError(f"Downloaded image is implausibly small ({image.width}x{image.height}).")


def run_once() -> dict:
    """Runs one full cycle. Returns the resulting state dict. Safe to call
    from the scheduler thread or directly from a manual POST /api/satellite/update."""
    config.ensure_dirs()
    storage.write_state({"status": "UPDATING", "last_check_utc": storage.now_iso()})
    log.info("Checking for new observation...")

    source_name = config.SATELLITE_SOURCE
    fallback_name = config.SATELLITE_FALLBACK_SOURCE
    attempted = []

    for name in [source_name] + ([fallback_name] if fallback_name else []):
        if not name:
            continue
        attempted.append(name)
        try:
            result = _run_with_source(name)
            return result
        except SourceError as exc:
            log.warning("Source '%s' failed at stage '%s': %s", name, exc.stage, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - never let this crash the app
            log.error("Unexpected error with source '%s': %s", name, exc)
            continue

    state = storage.write_state({
        "status": "ERROR",
        "error": f"All configured sources failed: {attempted}",
        "error_stage": "source",
    })
    return state


def _run_with_source(source_name: str) -> dict:
    source = _build_source(source_name)
    if not source.is_configured():
        raise NotConfiguredError(f"Source '{source_name}' is missing required configuration.")

    metadata = source.get_latest_metadata()
    log.info("New observation found: %s at %s", metadata.source, metadata.observation_time_utc)

    if storage.already_have_observation(metadata.observation_time_utc):
        log.info("Already have this observation - skipping download (duplicate detection).")
        state = storage.read_state()
        state["status"] = "STALE" if _is_stale(metadata.observation_time_utc) else "LIVE"
        state["last_check_utc"] = storage.now_iso()
        return storage.write_state(state)

    log.info("Downloading...")
    t0 = time.time()
    full_image, raw_bytes = source.download(metadata)
    log.info("Download complete (%d bytes, %.1fs).", len(raw_bytes), time.time() - t0)

    _validate_image(full_image)

    raw_path = config.RAW_DIR / f"{source_name}_{metadata.observation_time_utc.replace(':', '')}.png"
    raw_path.write_bytes(raw_bytes if raw_bytes else b"")
    if not raw_bytes:
        full_image.save(raw_path)  # sources that return a stitched-only image

    log.info("Preprocessing...")
    if source.region_crop_supported:
        cropped, coverage = region.crop_north_indian_ocean(
            full_image, config.REGION_LAT_MIN, config.REGION_LAT_MAX,
            config.REGION_LON_MIN, config.REGION_LON_MAX)
        if cropped is None:
            raise SourceError(
                "The North Indian Ocean box is entirely outside this source's visible disk "
                "(check SATELLITE_LAT_MIN/MAX, SATELLITE_LON_MIN/MAX in .env, or switch source).")
        if not coverage.fully_visible:
            log.warning(
                "Only %.0f%% of the configured North Indian Ocean box is visible from %s "
                "(the rest is beyond the satellite's horizon - this is expected for the "
                "western Arabian Sea from Himawari; see backend/satellite/README.md).",
                coverage.fraction_visible * 100, source_name)
        region_info = {
            "lat_min": config.REGION_LAT_MIN, "lat_max": config.REGION_LAT_MAX,
            "lon_min": config.REGION_LON_MIN, "lon_max": config.REGION_LON_MAX,
            "coverage_fraction": round(coverage.fraction_visible, 2),
            "fully_visible": coverage.fully_visible,
            "description": None,
        }
    else:
        # This source's image is not in a geostationary-disk projection (or
        # its exact projection parameters haven't been calibrated), so we
        # do not attempt to compute a custom crop - see base.py's
        # region_crop_supported docstring. The full downloaded frame is
        # used as-is, honestly described rather than falsely claiming a
        # precise coverage_fraction we haven't actually computed.
        cropped = full_image
        log.info("Source '%s' provides a pre-framed regional image; using it as-is "
                  "(no programmatic crop - see its region_description).", source_name)
        region_info = {
            "lat_min": None, "lat_max": None, "lon_min": None, "lon_max": None,
            "coverage_fraction": None,
            "fully_visible": None,
            "description": source.region_description,
        }

    processed_path = config.PROCESSED_DIR / f"{source_name}_{metadata.observation_time_utc.replace(':', '')}.png"
    cropped.save(processed_path)

    latest_path = config.LATEST_DIR / "latest.png"
    cropped.save(latest_path)

    prediction = None
    if config.ENABLE_LIVE_INFERENCE:
        log.info("Running inference...")
        try:
            prediction = inference_bridge.run_experimental_inference(cropped)
            log.info("Prediction complete.")
        except InferenceUnavailable as exc:
            log.warning("Inference skipped: %s", exc)
            prediction = {"experimental": True, "error": str(exc)}

    received_time = storage.now_iso()
    status = "STALE" if _is_stale(metadata.observation_time_utc) else "LIVE"

    new_state = {
        "status": status,
        "source": metadata.source,
        "satellite": metadata.satellite,
        "product": metadata.product,
        "channel": metadata.channel,
        "observation_time_utc": metadata.observation_time_utc,
        "received_time_utc": received_time,
        "processed_time_utc": storage.now_iso(),
        "image_path": "/api/satellite/image",
        "prediction": prediction,
        "error": None,
        "error_stage": None,
        "last_check_utc": storage.now_iso(),
        "last_success_utc": received_time,
        "region": region_info,
    }
    storage.write_state(new_state)
    storage.append_history({
        "source": metadata.source,
        "observation_time_utc": metadata.observation_time_utc,
        "received_time_utc": received_time,
        "status": status,
    })
    log.info("Frontend state updated.")
    return new_state
