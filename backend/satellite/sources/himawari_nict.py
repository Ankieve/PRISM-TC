"""Himawari-8/9 real-time full-disk quicklook, via NICT's public mirror.

This is the PRIMARY satellite source for PRISM-TC because it is the only
one of the sources we evaluated that is:
  - completely free
  - requires no account, no API key, no credit card
  - has a stable, widely-used URL scheme (the same one used by several
    public "Himawari live wallpaper" open-source tools)

What it actually is: NICT (Japan's National Institute of Information and
Communications Technology) re-publishes JMA's Himawari-8/9 full-disk
imagery as pre-rendered PNG tiles for public / educational viewing, updated
on the same 10-minute cadence as the satellite's full-disk scan. It is a
quicklook (visualisation) product, not calibrated scientific brightness
temperature data - see satellite/README.md for what that means for our
model.

Endpoints used (undocumented officially by NICT, but stable and widely
relied upon - verify with test_connection.py from a machine with real
internet, since this sandboxed build environment cannot reach the public
internet to confirm it live):
  latest.json : {BASE}/latest.json
      -> {"date": "YYYY-MM-DD HH:MM:SS", "file": "<level>d/<px>/.../HHMMSS_0_0.png"}
  tile        : {BASE}/{level}/{px}/{yyyy}/{mm}/{dd}/{HHMMSS}_{X}_{Y}.png
      X, Y in [0, N-1] where N is the numeric part of the level (e.g. "4d" -> 4x4 tiles)
"""
from __future__ import annotations

import io
import re
import time
from datetime import datetime, timezone

import requests
from PIL import Image, UnidentifiedImageError

from .. import config
from .base import (InvalidResponseError, NetworkError, ObservationMetadata,
                    SatelliteSource)

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$")


class HimawariNictSource(SatelliteSource):
    name = "himawari_nict"

    def __init__(self):
        self.base_url = config.HIMAWARI_BASE_URL.rstrip("/")
        self.level = config.HIMAWARI_LEVEL
        self.tile_px = config.HIMAWARI_TILE_PX
        self.grid_n = self._grid_size(self.level)
        self.timeout = config.REQUEST_TIMEOUT_SECONDS
        self.retries = config.DOWNLOAD_RETRIES

    @staticmethod
    def _grid_size(level: str) -> int:
        match = re.match(r"^(\d+)d$", level)
        if not match:
            raise InvalidResponseError(f"Unrecognised HIMAWARI_LEVEL '{level}' (expected e.g. '4d').")
        return int(match.group(1))

    def is_configured(self) -> bool:
        return True  # no credentials needed

    def _get(self, url: str):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (404, 403) and attempt == self.retries:
                    raise InvalidResponseError(
                        f"{url} returned HTTP {resp.status_code}.")
            except requests.RequestException as exc:
                last_exc = exc
            time.sleep(min(2 ** attempt, 8))
        raise NetworkError(f"Could not reach {url} after {self.retries} attempts: {last_exc}")

    def get_latest_metadata(self) -> ObservationMetadata:
        url = f"{self.base_url}/latest.json"
        resp = self._get(url)
        try:
            payload = resp.json()
            date_str = payload["date"]
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidResponseError(f"latest.json had an unexpected shape: {exc}")

        m = _DATE_RE.match(date_str.strip())
        if not m:
            raise InvalidResponseError(f"latest.json date '{date_str}' did not match the expected format.")
        yyyy, mm, dd, HH, MM, SS = m.groups()
        obs_dt = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS), tzinfo=timezone.utc)

        return ObservationMetadata(
            source=self.name,
            satellite="Himawari-8/9 (JMA, via NICT public mirror)",
            product=f"Full-disk quicklook, {self.grid_n}x{self.grid_n} tile grid ({self.grid_n * self.tile_px}px)",
            channel="composite (as rendered by NICT, not a single calibrated band)",
            observation_time_utc=obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            fetch_url=url,
            extra={"yyyy": yyyy, "mm": mm, "dd": dd, "hhmmss": f"{HH}{MM}{SS}"},
        )

    def _tile_url(self, meta: ObservationMetadata, x: int, y: int) -> str:
        e = meta.extra
        return (f"{self.base_url}/{self.level}/{self.tile_px}/"
                f"{e['yyyy']}/{e['mm']}/{e['dd']}/{e['hhmmss']}_{x}_{y}.png")

    def _validate_tile(self, url: str, content: bytes) -> Image.Image:
        """Decode+validate one tile's raw bytes. Raises InvalidResponseError
        on genuine corruption/truncation.

        NOTE: this deliberately does NOT reject on small byte size. A
        full-disk image is square but Earth's disk is round, so corner
        tiles (far from the visible disk) are almost entirely a single flat
        color and can legitimately PNG-compress to just a few hundred
        bytes - that is not truncation. An earlier version of this code
        used a `len(content) < 512` heuristic that produced exactly this
        false positive on a real corner tile (observed: a 444-byte, fully
        valid tile rejected as "truncated"). The real signal for
        corruption/truncation is whether the bytes decode as a complete
        image of the expected size, not how many bytes they are.
        """
        if not content:
            raise InvalidResponseError(f"Tile {url} was empty (0 bytes).")
        try:
            tile = Image.open(io.BytesIO(content))
            tile.load()  # forces full decode now; a truncated PNG raises here
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidResponseError(f"Tile {url} was not a valid image: {exc}")
        if tile.size != (self.tile_px, self.tile_px):
            raise InvalidResponseError(
                f"Tile {url} decoded to size {tile.size}, expected "
                f"({self.tile_px}, {self.tile_px}) - likely truncated or wrong URL scheme."
            )
        return tile

    def download(self, metadata: ObservationMetadata):
        """Fetch every tile in the grid and stitch them into one full-disk
        image. Raises NetworkError/InvalidResponseError on any tile
        failure - a partially-stitched disk is worse than a clear error."""
        n = self.grid_n
        canvas = Image.new("RGB", (n * self.tile_px, n * self.tile_px))
        raw_bytes_total = bytearray()
        for y in range(n):
            for x in range(n):
                url = self._tile_url(metadata, x, y)
                resp = self._get(url)
                content = resp.content
                tile = self._validate_tile(url, content)
                canvas.paste(tile.convert("RGB"), (x * self.tile_px, y * self.tile_px))
                raw_bytes_total.extend(content)
        return canvas, bytes(raw_bytes_total)
