# PRISM-TC near-real-time satellite module

This document is the full, honest account of the satellite upgrade: what
was built, how it was tested, what it actually proves, and what it does
not. Read this before demoing the feature or answering judge questions
about it.

**This file was rewritten mid-build after switching the primary satellite
source.** Section 0 explains why, before the rest of the document (which
now describes the current, IMD/INSAT-based design).

## 0. Why IMD/INSAT instead of Himawari

The first version of this module used Himawari-8/9 (Japan's geostationary
satellite) via NICT's public mirror, because it was free and needed no
account. Two real problems came up once a teammate actually tested it with
real internet (this project was built in a sandbox with no internet
access at all - see section 10):

1. **The tile-file validation had a real bug.** A byte-size heuristic
   (`< 512 bytes = truncated`) incorrectly rejected a genuine, valid,
   small PNG tile - full-disk images are square but Earth's disk is
   round, so a corner tile far from the visible disk can legitimately
   compress to a few hundred bytes. This was found, root-caused, and
   fixed (validation now decodes the image and checks its dimensions
   instead of counting bytes - see `sources/himawari_nict.py` and the
   regression test in `test_offline.py`).
2. **NICT's real, current tile-file naming could not be confirmed.**
   `latest.json` returns a `file` field (e.g.
   `PI_H09_20260922_0910_TRC_FLDK_R10_PGPFD.png`) that does not match the
   tile-grid URL pattern (`{HHMMSS}_{x}_{y}.png`) this module assumed,
   and manually constructed tile URLs did not resolve either. Reverse-
   engineering NICT's actual current API through a live browser session
   turned out to be slow and inconclusive.

Separately, and more importantly: **Himawari (140.7 deg E) cannot
geometrically see the whole North Indian Ocean** - the western Arabian
Sea is beyond its horizon (this was correctly measured and reported by
the original code, not hidden - see the coverage-fraction bug write-up
that follows below in this history). For a project specifically about
Indian cyclones, that is a real limitation, not just an inconvenience.

**The fix: switch to India's own satellite.** IMD (India Meteorological
Department) publishes a continuously-refreshed quicklook image from
ISRO's INSAT-3D/3DS satellite at a single, static, unauthenticated URL:

```
https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg
```

This was found and confirmed by hand (a teammate opened the URL directly
in a browser and sent a screenshot back, since this build environment has
no internet - see section 10) on 2026-09-23. It is dramatically simpler
to consume than Himawari's tile system (one static file, no grid, no
per-tile stitching), and its "Asia sector" framing already covers the
*entire* North Indian Ocean - the Arabian Sea and the Bay of Bengal both
- in one image, which Himawari could not do. It is also simply a better
fit for an India-focused hackathon project: India's own operational
weather satellite, not a foreign one.

Himawari's code is kept in the repository (`sources/himawari_nict.py`,
with its real bug now fixed) as a documented-but-unverified secondary
option for future work, but it is no longer the default primary or
fallback source - see section 9 and `.env.example`.

## 1. What changed, file by file

Package `backend/satellite/` (nothing outside it was rewritten):

- `config.py` - all settings, read from `.env` / environment variables.
  No hard-coded credentials anywhere. Default `SATELLITE_SOURCE` is now
  `imd_insat`.
- `storage.py` - JSON-file state (`state.json`), history (`history.json`),
  atomic writes, duplicate-observation detection.
- `region.py` - the geostationary-projection crop math, still used by the
  (now secondary/unverified) Himawari source. Not used by the default
  IMD source - see section 3.
- `sources/base.py` - the source interface, the exceptions the pipeline
  knows how to handle without crashing, and the `region_crop_supported` /
  `region_description` flags a source uses to say whether the
  geostationary crop math applies to it.
- `sources/imd_insat.py` - the PRIMARY, working, free, no-account source
  (section 9). A single static-image fetch, no tile grid.
- `sources/himawari_nict.py` - kept for reference/future work; a real bug
  was found and fixed in it (section 0), but its current tile URL scheme
  is unverified - not used by default.
- `sources/eumetsat_iodc.py` - an OPTIONAL fallback, implemented against
  EUMETSAT's documented API shape but **unverified** (see section 10).
- `sources/mosdac.py` - a documented **stub**, not implemented, with the
  concrete reasons why (see section 10).
- `inference_bridge.py` - the (disabled-by-default) bridge from a live
  image to the existing model, with the compatibility reasoning in full
  (section 8).
- `pipeline.py` - orchestrates one check/download/validate/(optional
  crop)/(optional inference)/store cycle. Never raises; always ends in a
  clean state. Now branches on `region_crop_supported` so the
  geostationary crop math is never silently misapplied to a source in a
  different projection (a real design bug caught and fixed before it
  shipped - see `test_offline.py`'s
  `test_pipeline_skips_crop_for_a_source_that_opts_out`).
- `scheduler.py` - a background thread calling the pipeline on an
  interval matched to the real product cadence.
- `test_connection.py` - **run this yourself, with real internet** (see
  section 6). Rewritten for the single-fetch IMD flow.
- `test_offline.py` - the tests that could be, and were, run in this
  build's sandboxed environment (section 12 has the actual output).

Changed files:
- `backend/server.py` - added the satellite import (wrapped in try/except
  so a failure here can never take down the existing app), 6 new routes,
  and starting the scheduler in `__main__`. Nothing about the existing
  `/api/health`, `/api/samples`, `/api/demo`, `/api/predict` routes was
  touched, and `backend/tests/smoke_test.py` (the pre-existing 33-check
  suite) still passes unchanged.
- `index.html`, `app.js`, `style.css` - added one new card ("Live /
  Near-Real-Time Satellite") using the same visual language as the rest
  of the dashboard (`.card`, `.badge`, `.link-btn`). `app.js` was updated
  again when the source switched, so the coverage line shows either a
  computed percentage (Himawari-style sources) or a plain description
  (IMD-style sources) depending on what the backend reports - never a
  fabricated number.
- `backend/requirements.txt` - added `requests`.
- `backend/tests/smoke_test_satellite.py` - new, self-contained smoke
  test for the new routes (no internet needed).
- `.env.example`, `.gitignore` - new.
- `CLAUDE.md` - a short pointer section added at the end.

## 2. What was added (by category)

- **Satellite pipeline**: fetch -> validate -> (crop, only if the source
  supports it) -> (optional) infer -> store -> log, adapted to the
  existing stdlib-server architecture (no FastAPI introduced).
- **Backend**: 5 new GET routes + 1 new POST route, all defensive (a
  satellite failure returns a clear error/status, never a 500 crash,
  never a fabricated success).
- **Frontend**: one new dashboard card with a LIVE/STALE/UPDATING/ERROR/NO
  DATA badge, image, metadata grid, and an always-visible disclaimer.
- **Storage**: `backend/data/satellite/{raw,processed,latest,metadata}/`,
  plain files and JSON, no database.
- **Configuration**: `.env` / `.env.example`, nothing hard-coded.

## 3. How the data flows (actual, implemented architecture)

```
INSAT-3D/3DS (ISRO)
      -> IMD public mirror (free, no login, single static file)
      -> sources/imd_insat.py            (HTTP GET, retries, timeout,
                                           Last-Modified header if present)
      -> pipeline.py: validate image decodes and is a plausible size
      -> (no crop - region_crop_supported=False; IMD's own "Asia sector"
             framing already covers the whole North Indian Ocean)
      -> storage.py: save raw + "processed" (=same image) + latest PNG,
             update state.json (atomic)
      -> [ONLY IF ENABLE_LIVE_INFERENCE=true] inference_bridge.py -> predict.py
      -> server.py: GET /api/satellite/status | image | metadata | history
      -> app.js: polls /api/satellite/status every 60s, renders the card
```

The existing, separate, already-working flow (upload an image -> `/api/predict`
-> `predict.py` -> class + confidence) is completely untouched and is still
the only source of the validated ~60%-accuracy prediction.

## 4. How to run it

```bash
cd PRISM-TC-Frontend
pip install -r backend/requirements.txt
cp .env.example .env            # edit if you want to change anything
python backend/server.py
```
Open `http://localhost:8000`. Within a few seconds the new card should show
`UPDATING`, then either `LIVE` (internet worked) or `ERROR` (see the
message in the card). Run `python backend/satellite/test_connection.py`
first if you want to verify the fetch on its own before starting the server
(section 6).

To turn the satellite feature off entirely (e.g. for a demo where you don't
want any extra network calls): set `SATELLITE_ENABLED=false` in `.env`.

## 5. Environment variables

All documented with defaults and reasoning in `.env.example`. The ones you
are most likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `SATELLITE_ENABLED` | `true` | master on/off switch |
| `SATELLITE_SOURCE` | `imd_insat` | which source to use |
| `IMD_INSAT_IR1_URL` | (IMD's URL) | override only if IMD moves/renames the file |
| `UPDATE_INTERVAL` | `300` (5 min) | polling interval, seconds |
| `ENABLE_LIVE_INFERENCE` | `false` | run the model on the live image (experimental) |
| `SATELLITE_LAT_MIN/MAX`, `SATELLITE_LON_MIN/MAX` | -5/30/40/100 | only used by crop-capable sources (not the default) |
| `EUMETSAT_CONSUMER_KEY/SECRET/COLLECTION_ID` | empty | optional fallback, needs a free account |

## 6. How to verify live satellite data (concrete tests)

This project was built in a sandboxed environment with **no route to the
public internet at all** - not even to `pypi.org` (verified directly; see
section 10). So the live-network parts were written and reasoned through
carefully, and validated as far as possible offline (storage, error
handling, the pipeline's crop-skip branching - section 12), plus one round
of manual verification through a teammate's browser (which is how the
IMD URL and the real image shown in section 0 were confirmed). You should
still run this yourself before a live demo:

```bash
python backend/satellite/test_connection.py
```
This performs 5 real checks (reachability, freshness, download,
validation, full pipeline run), saves `test_output_imd.jpg`, and tells you
to open it and confirm it actually shows India / the Arabian Sea / the Bay
of Bengal with a real IR cloud picture.

Second concrete test, once the server is running:
```bash
curl http://localhost:8000/api/satellite/status
```
Expect `"status": "LIVE"` with a recent `observation_time_utc`. `"status":
"ERROR"` with a message is an honest failure report, not a crash - read
the `error` field.

## 7. What happens when new data arrives (the automated flow)

Every `UPDATE_INTERVAL` seconds (default 300 = 5 minutes): the scheduler
calls `pipeline.run_once()`, which does a cheap `HEAD` request (falling
back to a streamed `GET` that reads no body if `HEAD` isn't supported) to
read the file's `Last-Modified` header, compares it against the last one
we successfully processed (`storage.already_have_observation`), and does
nothing further if it's the same one (duplicate detection - we don't
re-download or reprocess the same file). If the server doesn't send a
`Last-Modified` header at all, the code honestly falls back to "the time
we fetched it" rather than pretending to know the true scan time - this
is recorded in `extra["timestamp_source"]` and printed by
`test_connection.py`. If it's new, it downloads the file, checks it
decodes as a real image of a plausible size, saves raw + processed +
"latest" copies, optionally runs inference, and atomically rewrites
`state.json` plus appends one line to `history.json`. The frontend picks
this up on its next 60-second poll. Any failure at any stage is caught,
logged with the `[Satellite]` prefix and the failing stage, and turned
into a clean `ERROR` state - the scheduler thread itself never dies, so it
keeps retrying on schedule.

## 8. Model limitations - does the model genuinely use live data?

**No, not by default, and here is exactly why** (this is the honest answer
the brief asked for, not a hedge):

The model (EfficientNet-B0, `backend/predict.py` + `backend/model.pth`) was
fine-tuned only on TCIR: single-channel, brightness-temperature-like
grayscale chips, each one **cropped and centred on one named storm** by
TCIR's own extraction pipeline, ~201x201px, resized to 224x224, no
ImageNet normalisation. The live IMD/INSAT image is a different kind of
image on several counts: it is a wide regional (Asia-sector) view rather
than a tight storm-centred chip; it is a rendered quicklook (with a
labelled colour/greyscale count bar, coastlines, place-name labels and a
title banner drawn on top) rather than a clean calibrated array; and it
has never been evaluated by anyone, on anything, for this model - the
~60% test accuracy and the confusion matrices in the model card are
TCIR-only numbers. Feeding it straight into `predict()` would produce a
number that looks exactly like a real prediction while having no measured
relationship to reality.

So: `inference_bridge.py` exists and is wired end-to-end (per the brief's
"if feasible implement a separate compatible inference pipeline"), but it
is **off by default** (`ENABLE_LIVE_INFERENCE=false`), and even when turned
on it returns a differently-shaped, clearly `"experimental": true` result
with an explicit caveat string, never mixed into the same fields as the
validated TCIR-based prediction. Doing this properly (locating a storm in
the live frame and cropping tightly to it, at comparable resolution to
TCIR, then re-validating accuracy on that pipeline) is real future work,
not something to fake for a demo.

## 9. Data source

```
Satellite:            INSAT-3D / INSAT-3DS (ISRO)
Product:               Asia sector, Thermal Infrared-1 (10.8 micron),
                        Level-1C Mercator quicklook
Provider:              India Meteorological Department (IMD) - public
                        real-time website
Update frequency:       Not officially documented for this specific
                        quicklook; INSAT-3D/3DS's normal full-disk scan
                        cadence is roughly every 30 minutes
Approximate latency:    Not confirmed - verify with test_connection.py
Access method:          Plain HTTPS GET, single static JPEG file, no
                        query parameters
Cost:                   Free
Authentication:         None required
Geographic coverage:    IMD's own "Asia sector" framing - approximately
                        45-105 deg E, 0-45 deg N (read directly off the
                        image's own labelled gridlines). Covers the whole
                        Arabian Sea AND the whole Bay of Bengal in a
                        single frame - a real advantage over Himawari
                        (see section 0)
```

Secondary/unverified (kept in the codebase, not used by default - see
section 0):
```
Satellite:            Himawari-8/9 (Japan Meteorological Agency)
Product:               Full-disk quicklook imagery, tile-based
Provider:              NICT (Japan) - public real-time mirror
Cost:                  Free
Authentication:        None required
Status:                A real validation bug was found and fixed in this
                       build (section 0), but the current tile-file
                       naming scheme could not be confirmed live from a
                       real browser - do not rely on this source without
                       re-verifying it yourself first.
Geographic coverage:   Full Earth disk as seen from 140.7 deg E - cannot
                       see the western Arabian Sea at all (beyond its
                       horizon), which is the main reason this project
                       switched to IMD/INSAT instead.
```

Optional fallback (scaffolded, unverified):
```
Satellite:            Meteosat-8 (EUMETSAT), positioned for Indian Ocean coverage
Product:               SEVIRI Level 1.5 image data (IODC)
Provider:              EUMETSAT Data Store
Cost:                  Free
Authentication:        Free account + API key required (eoportal.eumetsat.int)
Geographic coverage:   Centred on the Indian Ocean (41.5 deg E)
```

## 10. Known limitations (brutally honest)

1. **The live-network path was verified only manually, once, through a
   teammate's browser - not by an automated test with real internet.**
   This sandboxed build environment has no route to the public internet
   at all (confirmed directly: `curl` to `pypi.org`, NICT, MOSDAC,
   EUMETSAT, IMD, NOAA and AWS S3 all returned a network-policy 403
   before even reaching those servers). The IMD URL in `config.py` was
   confirmed live by a teammate opening it directly in Chrome and sending
   a screenshot back (see section 0) - a real, encouraging result - but
   you must still run `test_connection.py` yourself on a normal machine
   before a live demo, since that is the first time this code's actual
   HTTP logic (retries, header parsing, decoding) touches the real
   internet end to end.
2. **No programmatic geographic crop is applied to the IMD image** - it
   is a Mercator projection with a fixed "Asia sector" framing, and doing
   a precise custom crop correctly would need pixel-exact calibration
   from a real downloaded file (not done - see `sources/imd_insat.py`).
   The approximate extent (45-105 deg E, 0-45 deg N) is read directly off
   the image's own labelled gridlines and reported honestly as a text
   description, not a computed coverage percentage - `region_crop_supported
   = False` makes this an explicit, deliberate choice rather than a silent
   gap.
3. **The exact satellite scan time is not parsed from the image.** The
   image has a GMT/IST banner burned into it, but this code does not OCR
   that (no OCR dependency added). It uses the HTTP `Last-Modified`
   header if the server sends one; otherwise it honestly falls back to
   "the time we fetched it" (`extra["timestamp_source"] =
   "fetch_time_fallback"`), which `test_connection.py` will tell you
   about explicitly if it happens.
4. **The model does not validly consume this live imagery**, for the
   reasons in section 8. Live inference is implemented but disabled by
   default, and always labelled experimental when enabled.
5. **Himawari's real bug was fixed, but its current API is unverified**
   - see section 0 for the full story. It remains in the codebase for
   future work, not as a working fallback today.
6. **EUMETSAT IODC is scaffolded, not verified.** Its exact Data Store
   collection id and the precise shape of its search/download responses
   could not be confirmed against the live API from this environment.
   Native SEVIRI product decoding (needing a library like `satpy`) is
   also not implemented.
7. **MOSDAC/INSAT direct access is not implemented at all**, on purpose
   rather than faked - see `sources/mosdac.py`'s docstring for the
   specific reasons (session-based login, no documented public API
   found, terms-of-use for automation unclear).
8. **This module intentionally cannot break the existing, working app.**
   Every integration point (`server.py`'s import, every route, the
   scheduler) is wrapped so a satellite failure degrades to a clear error
   state rather than taking anything else down - verified by an actual
   test run (section 12), not just claimed.

## 11. If you only remember one sentence

IMD's public INSAT-3D/3DS Asia-sector infrared quicklook is a real, free,
no-login source - India's own satellite - now wired end-to-end into
PRISM-TC's dashboard with honest status reporting and no fabricated crop
or timestamp precision; it feeds a "near-real-time picture" panel, not the
AI prediction, because the model was never trained or validated on this
kind of image - and the code says so, out loud, in the UI.

## 12. Actual test output from this build (not hypothetical)

`python backend/satellite/test_offline.py` (no internet - run in the build
environment, after switching to IMD/INSAT):
```
[PASS] Mumbai projects to visible scan angles (not None)
[PASS] Full-disk limb angle is a small positive number of radians
[PASS] North Indian Ocean box projects to a valid pixel box on a 2200x2200 disk
[PASS] Pixel box is within image bounds
[PASS] Coverage correctly reports the box as only PARTIALLY visible from Himawari (the western Arabian Sea, lon<~60E, is beyond its horizon)
[PASS] A point on the far side of the Earth is correctly reported as not visible
[PASS] crop_north_indian_ocean returns an image for a synthetic full disk
[PASS] Cropped image has positive dimensions
[PASS] crop_north_indian_ocean also returns a CoverageResult
[PASS] write_state returns the merged state
[PASS] read_state reflects what was written
[PASS] already_have_observation is True for the same timestamp
[PASS] already_have_observation is False for a different timestamp
[PASS] append_history / read_history round-trips one entry
[PASS] A corrupted state.json falls back to defaults instead of raising
[PASS] run_once() with an unimplemented source returns ERROR, not a crash
[PASS] Error message names the failing source
[PASS] a small-but-valid, correctly-sized tile is accepted, not rejected as truncated
[PASS] an actually-truncated tile (half the bytes cut off) is still correctly rejected
[PASS] a source with region_crop_supported=False reaches LIVE, not ERROR
[PASS] no crop is applied: the saved image keeps the source's original size
[PASS] region.coverage_fraction is honestly None (not a fabricated number)
[PASS] region.description carries the source's own honest description
23/23 checks passed.
```

`python backend/tests/smoke_test.py` (the pre-existing 33-check suite):
still `ALL PASSED`, unchanged, after this feature was added and after the
source switch.

`python backend/tests/smoke_test_satellite.py` (new, no internet needed),
re-run after switching the default source to `imd_insat`:
```
13 passed, 0 failed
```
This includes driving a real running server through a POST
`/api/satellite/update` with no internet available (this build
environment's actual condition) against the new default source and
confirming it returns a clean `ERROR` status with a real error message -
not a fabricated success, and not a crash - while `/api/health`,
`/api/samples` and `/api/demo` kept working throughout.

Manual verification (not part of the automated suite, done through a
teammate's browser since this build environment has no internet):
`https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg` was confirmed on
2026-09-23 to load as a real 2002x2242 INSAT-3DS infrared image, correctly
showing the Indian subcontinent, Pakistan, Iran, the Arabian Sea, the Bay
of Bengal, and what appears to be an actual active cyclonic cloud system
near the Odisha/Andhra Pradesh coast, with a real GMT/IST timestamp banner
and labelled Mercator lat/lon gridlines burned into the image.
