"""INSAT-3D/3DS real-time Infrared quicklook, via IMD's public website.

This is the PRIMARY satellite source for PRISM-TC (replacing an earlier
attempt at Himawari-8/9, kept in himawari_nict.py as an unverified secondary
option - see satellite/README.md "Why IMD/INSAT instead of Himawari" for the
full story of why this source was switched to mid-build).

What it actually is: the India Meteorological Department publishes a
continuously-refreshed Thermal Infrared-1 (10.8 micron) quicklook image from
ISRO's INSAT-3D/3DS satellite, covering an "Asia sector" that comfortably
includes the whole North Indian Ocean - the Arabian Sea AND the Bay of
Bengal in one frame (unlike Himawari at 140.7E, which cannot see the western
Arabian Sea at all - a real geometric limitation documented in region.py).
This is also simply a better fit for an India-focused project: it is India's
own operational weather satellite, not a foreign one.

URL (confirmed live, manually, on 2026-09-23 - see README for how this was
found): a single static image, no tiles, no query parameters, no login:
    https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg

Important honesty notes (see satellite/README.md for the full versions):
  - This image is in a Mercator projection with an "Asia sector" framing
    chosen by IMD, not a box we control. We do NOT attempt to crop it to a
    custom lat/lon box the way region.py does for Himawari's geostationary
    disk - doing that correctly would require calibrating exact pixel
    positions for the Mercator grid from a real downloaded file, which
    hasn't been done (region_crop_supported = False; the full frame is used
    as-is). Its approximate visible extent, read directly off the labelled
    lat/lon gridlines drawn on the image, is documented in
    region_description below.
  - The exact scan time is printed inside the image itself (a header banner
    with GMT/IST). This code does NOT parse that (no OCR dependency); it
    uses the HTTP response's Last-Modified header if the server sends one,
    otherwise it falls back to "the time we fetched it", which is clearly
    a received/fetch time, not a guaranteed scan time - see the
    `extra["timestamp_source"]` field on the returned metadata.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from .. import config
from .base import (InvalidResponseError, NetworkError, ObservationMetadata,
                    SatelliteSource)


class IMDInsatSource(SatelliteSource):
    name = "imd_insat"

    region_crop_supported = False
    region_description = (
        "IMD's own 'Asia sector' frame (approximate extent read off the "
        "image's own labelled gridlines: roughly 45-105 deg E, 0-45 deg N) "
        "- covers the whole Arabian Sea and Bay of Bengal in one image, not "
        "cropped to a custom box."
    )

    def __init__(self):
        self.url = config.IMD_INSAT_IR1_URL
        self.timeout = config.REQUEST_TIMEOUT_SECONDS
        self.retries = config.DOWNLOAD_RETRIES

    def is_configured(self) -> bool:
        return True  # no credentials needed - public static file

    def _request(self, method: str, **kwargs):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.request(method, self.url, timeout=self.timeout, **kwargs)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (404, 403) and attempt == self.retries:
                    raise InvalidResponseError(f"{self.url} returned HTTP {resp.status_code}.")
            except requests.RequestException as exc:
                last_exc = exc
            time.sleep(min(2 ** attempt, 8))
        raise NetworkError(f"Could not reach {self.url} after {self.retries} attempts: {last_exc}")

    @staticmethod
    def _parse_last_modified(resp) -> tuple[str, str]:
        """Returns (observation_time_utc_iso, timestamp_source)."""
        header = resp.headers.get("Last-Modified")
        if header:
            try:
                dt = parsedate_to_datetime(header)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "http_last_modified_header"
            except (TypeError, ValueError):
                pass
        # No usable Last-Modified header - fall back to fetch time. This is
        # honestly labelled via timestamp_source, not silently presented as
        # the true scan time.
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "fetch_time_fallback"

    def get_latest_metadata(self) -> ObservationMetadata:
        # A HEAD request is enough to read Last-Modified without downloading
        # the (roughly 1-2 MB) image body - keeps polling cheap. Some static
        # file servers reject HEAD; fall back to a streamed GET that reads
        # only headers, never the body, if HEAD fails outright.
        try:
            resp = self._request("HEAD")
        except NetworkError:
            resp = self._request("GET", stream=True)
            resp.close()

        obs_time, ts_source = self._parse_last_modified(resp)
        return ObservationMetadata(
            source=self.name,
            satellite="INSAT-3D/3DS (ISRO, via IMD public mirror)",
            product="Asia sector, Level-1C Mercator quicklook",
            channel="IR1 (thermal infrared, 10.8 micron)",
            observation_time_utc=obs_time,
            fetch_url=self.url,
            extra={"timestamp_source": ts_source},
        )

    def download(self, metadata: ObservationMetadata):
        resp = self._request("GET")
        content = resp.content
        if not content:
            raise InvalidResponseError(f"{self.url} returned an empty response.")
        try:
            image = Image.open(BytesIO(content))
            image.load()  # forces full decode now; a truncated file raises here
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidResponseError(f"{self.url} was not a valid image: {exc}")
        return image.convert("RGB"), content
