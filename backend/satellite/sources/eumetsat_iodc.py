"""EUMETSAT Meteosat-8 "Indian Ocean Data Coverage" (IODC) - OPTIONAL
fallback source, implemented but UNVERIFIED from this build environment.

Why this one is technically the best-positioned source geographically:
Meteosat-8 sits at 41.5 degrees East specifically to cover the Indian
Ocean, so both the Arabian Sea and the Bay of Bengal are near the centre
of its view (unlike Himawari at 140.7E, which sees the North Indian Ocean
at a shallow, oblique viewing angle near the edge of its disk). EUMETSAT
also publishes real calibrated Level 1.5 SEVIRI data with documented
geolocation, which is scientifically a better match for anything more
serious than a quicklook picture.

Why it is not the PRIMARY source: it requires a EUMETSAT Earth Observation
Portal account (free registration, no card - https://eoportal.eumetsat.int/)
and OAuth2 API credentials, and this sandboxed build environment has no
route to the public internet (confirmed: even pypi.org is blocked here by
network policy), so the exact Data Store collection id and the token/
search/download endpoint shapes below could NOT be exercised against the
live API before shipping this. They are written to EUMETSAT's publicly
documented Data Store API pattern (OAuth2 client-credentials token, then a
collection search, then a per-product download), but you must verify them
with real credentials before relying on this path - see
satellite/test_connection.py, which will refuse to silently report success
if this source is selected but any of these assumptions fail.

Required environment variables (see .env.example):
  EUMETSAT_CONSUMER_KEY
  EUMETSAT_CONSUMER_SECRET
  EUMETSAT_COLLECTION_ID   - confirm the exact id for "Meteosat-8/IODC,
                             Level 1.5 image data" on data.eumetsat.int
                             yourself; we deliberately do not guess one.
"""
from __future__ import annotations

import time

import requests

from .. import config
from .base import (AuthError, InvalidResponseError, NetworkError,
                    NotConfiguredError, ObservationMetadata, SatelliteSource)

_TOKEN_URL = "https://api.eumetsat.int/token"
_SEARCH_URL = "https://api.eumetsat.int/data/search-products/os"
_DOWNLOAD_URL_TEMPLATE = "https://api.eumetsat.int/data/download/products/{product_id}"


class EumetsatIodcSource(SatelliteSource):
    name = "eumetsat_iodc"

    def __init__(self):
        self.key = config.EUMETSAT_CONSUMER_KEY
        self.secret = config.EUMETSAT_CONSUMER_SECRET
        self.collection_id = config.EUMETSAT_COLLECTION_ID
        self.timeout = config.REQUEST_TIMEOUT_SECONDS
        self._token = None
        self._token_expiry = 0.0

    def is_configured(self) -> bool:
        return bool(self.key and self.secret and self.collection_id)

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        if not self.is_configured():
            raise NotConfiguredError(
                "EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET / "
                "EUMETSAT_COLLECTION_ID are not all set in .env.")
        try:
            resp = requests.post(
                _TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self.key, self.secret),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise NetworkError(f"Could not reach EUMETSAT token endpoint: {exc}")
        if resp.status_code == 401:
            raise AuthError("EUMETSAT rejected the consumer key/secret (401).")
        if resp.status_code != 200:
            raise InvalidResponseError(f"EUMETSAT token endpoint returned HTTP {resp.status_code}.")
        try:
            payload = resp.json()
            self._token = payload["access_token"]
            self._token_expiry = time.time() + float(payload.get("expires_in", 3600))
        except (ValueError, KeyError) as exc:
            raise InvalidResponseError(f"Unexpected token response shape: {exc}")
        return self._token

    def get_latest_metadata(self) -> ObservationMetadata:
        token = self._get_token()
        try:
            resp = requests.get(
                _SEARCH_URL,
                params={
                    "format": "json",
                    "pi": self.collection_id,
                    "si": 0,
                    "c": 1,
                    "sort": "start,desc",
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise NetworkError(f"Could not reach EUMETSAT search endpoint: {exc}")
        if resp.status_code != 200:
            raise InvalidResponseError(f"EUMETSAT search returned HTTP {resp.status_code}.")
        try:
            payload = resp.json()
            item = payload["features"][0]
            product_id = item["id"]
            observation_time = item["properties"]["date"].split("/")[0]
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            raise InvalidResponseError(
                "EUMETSAT search response did not have the expected shape "
                f"(verify _SEARCH_URL/collection id against the live API): {exc}")

        return ObservationMetadata(
            source=self.name,
            satellite="Meteosat-8 (IODC)",
            product="SEVIRI Level 1.5 image data (IODC)",
            channel="IR/WV (see product metadata once verified)",
            observation_time_utc=observation_time,
            fetch_url=_DOWNLOAD_URL_TEMPLATE.format(product_id=product_id),
            extra={"product_id": product_id, "token": token},
        )

    def download(self, metadata: ObservationMetadata):
        raise NotImplementedError(
            "EUMETSAT product download/decoding (native format -> image) is "
            "scaffolded but not implemented: SEVIRI Level 1.5 products are "
            "distributed as compressed native/HRIT format, not a plain PNG, "
            "and decoding them needs a library (e.g. satpy) that could not "
            "be installed or verified in this build environment. Finish "
            "this by installing satpy on a machine with real internet, "
            "confirming EUMETSAT_COLLECTION_ID on data.eumetsat.int, and "
            "implementing the native-format read here.")
