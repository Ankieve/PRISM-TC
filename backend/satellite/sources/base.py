"""Common interface every satellite source implements, and the exceptions
pipeline.py knows how to handle without crashing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class SourceError(Exception):
    """Base class for a source failing in an understood way (network,
    auth, no-new-data, bad response, ...). pipeline.py catches this and
    records it in state.json instead of letting it crash the server."""
    stage = "source"


class NetworkError(SourceError):
    stage = "network"


class AuthError(SourceError):
    stage = "auth"


class NoNewDataError(SourceError):
    """Raised when the source is reachable and answered normally, but has
    nothing newer than what we already processed. Not a failure."""
    stage = "no_new_data"


class InvalidResponseError(SourceError):
    stage = "invalid_response"


class NotConfiguredError(SourceError):
    """Raised when a source is selected but its required credentials/config
    are missing (e.g. EUMETSAT keys not set)."""
    stage = "not_configured"


class NotImplementedSourceError(SourceError):
    """Raised by sources that are deliberately scaffolded but not finished
    (see sources/mosdac.py) rather than pretending to work."""
    stage = "not_implemented"


@dataclass
class ObservationMetadata:
    source: str                  # short id, e.g. "himawari_nict"
    satellite: str                # e.g. "Himawari-9"
    product: str                  # e.g. "Full disk true-colour/IR quicklook"
    channel: str                  # e.g. "composite" or a specific band name
    observation_time_utc: str     # ISO 8601, when the satellite captured it
    fetch_url: Optional[str] = None
    extra: Optional[dict] = None


class SatelliteSource:
    name = "base"

    # Whether pipeline.py should run region.crop_north_indian_ocean() (the
    # geostationary-projection crop math) on this source's image. That math
    # is specific to a geostationary satellite's viewing geometry (needs a
    # sub-satellite longitude) and would silently produce a WRONG crop if
    # applied to an image in a different projection (e.g. a Mercator
    # quicklook that already frames a fixed region, like IMD's). Sources
    # that are not geostationary-disk quicklooks must set this to False and
    # set region_description instead - pipeline.py then uses the full
    # downloaded image as-is rather than guessing a crop.
    region_crop_supported: bool = True
    # Human-readable description of what the image actually covers, shown
    # in the frontend instead of a computed coverage percentage. Only used
    # when region_crop_supported is False.
    region_description: Optional[str] = None

    def is_configured(self) -> bool:
        """Return False if required credentials/config are missing."""
        return True

    def get_latest_metadata(self) -> ObservationMetadata:
        """Return metadata for the newest available observation. Raise a
        SourceError subclass on any failure. Should be cheap enough to poll
        often (e.g. a HEAD request rather than downloading the full image)."""
        raise NotImplementedError

    def download(self, metadata: ObservationMetadata):
        """Download and return (PIL.Image, raw_bytes) for the observation
        described by metadata. Raise a SourceError subclass on failure."""
        raise NotImplementedError
