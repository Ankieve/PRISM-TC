"""MOSDAC / ISRO (INSAT-3D / 3DR / 3DS) - NOT IMPLEMENTED.

This is deliberately a stub, not a fake implementation. Here is exactly
why, so a teammate with a MOSDAC account can pick this up:

  - MOSDAC (https://www.mosdac.gov.in/) requires an account, and almost
    all INSAT-3D/3DR/3DS products are served through the browser-based
    "Open Data" / order portal behind a logged-in session (cookies from a
    web login), not a documented, stable, anonymous REST API suitable for
    unattended polling.
  - This sandboxed build environment has no route to the public internet
    at all (verified: even pypi.org is blocked here by network policy), so
    there was no way to inspect MOSDAC's actual current endpoints, session
    mechanics, or terms of use for automated access from here.
  - Automating a browser login is a materially different (and riskier -
    ToS, credential handling, session/cookie storage) engineering task
    than calling a public REST API, and should not be built and shipped
    without someone who can actually test it against the live site with a
    real account and read MOSDAC's terms of use for automated access first.

What IS in place for whoever finishes this:
  - config.py already reads MOSDAC_USERNAME / MOSDAC_PASSWORD from the
    environment (never hard-code them).
  - The SatelliteSource interface in base.py is exactly what a working
    MosdacSource would need to implement (get_latest_metadata, download).
  - pipeline.py and the /api/satellite/* endpoints do not care which
    source is active - selecting SATELLITE_SOURCE=mosdac in .env once this
    class is implemented is the only wiring needed.

Do not remove this docstring when implementing it - keep the record of why
it was skipped for the next person who wonders.
"""
from __future__ import annotations

from .base import NotImplementedSourceError, ObservationMetadata, SatelliteSource


class MosdacSource(SatelliteSource):
    name = "mosdac"

    def is_configured(self) -> bool:
        return False

    def get_latest_metadata(self) -> ObservationMetadata:
        raise NotImplementedSourceError(
            "MOSDAC/INSAT is not implemented (requires a logged-in MOSDAC "
            "session and could not be verified from this build environment "
            "- see the module docstring in backend/satellite/sources/mosdac.py). "
            "Use SATELLITE_SOURCE=himawari_nict instead.")

    def download(self, metadata: ObservationMetadata):
        raise NotImplementedSourceError("MOSDAC/INSAT download is not implemented.")
