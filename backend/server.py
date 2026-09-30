"""PRISM-TC backend (SIH26070) - standard-library web server, no Flask needed.

Run:   python backend/server.py          (from the PRISM-TC-Frontend folder)
Open:  http://localhost:8000

Serves the frontend (index.html, style.css, app.js) AND the API:
    GET  /api/health    server + model status
    GET  /api/samples   list of built-in sample images
    GET  /api/demo      run the model on the next built-in sample image
    POST /api/predict   JSON body, see README/CLAUDE.md for the fields

Near-real-time satellite (see backend/satellite/README.md for the full
picture, including honest limitations):
    GET  /api/satellite/status     current status (LIVE/STALE/UPDATING/ERROR/NO_DATA) + metadata
    GET  /api/satellite/latest     alias of /status, kept for readability on the frontend
    GET  /api/satellite/metadata   just the source/product/timestamps, no status logic
    GET  /api/satellite/image      the latest cropped North Indian Ocean PNG
    GET  /api/satellite/history    recent update history
    POST /api/satellite/update     trigger one manual check-and-update cycle now

Real ERA5 reanalysis fetch (see backend/era5.py for full honesty notes - this
is a real Copernicus dataset, not synthetic, but it is NOT real-time):
    GET  /api/era5/status          whether cdsapi/xarray are installed and a
                                    CDS API key was found (no network call)
    GET  /api/era5/fetch?lat=&lon= fetch SST/wind-shear/humidity/vorticity for
                                    one point from ERA5 (needs the operator's
                                    own free CDS account - see era5.py)

Environment variables: HOST (default 127.0.0.1), PORT (default 8000).
Satellite- and ERA5-specific ones are documented in .env.example.
Without backend/model.pth the server still starts in DEMO MODE and every
response is clearly marked as placeholder. If the satellite module fails to
import or misbehaves for any reason, the rest of this server (health,
samples, demo, predict) is completely unaffected - see the try/except
around its import below.
"""

import base64
import binascii
import hashlib
import io
import json
import math
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import logic  # noqa: E402  (needs the sys.path line above)
import model_card  # noqa: E402
import ood_guard  # noqa: E402
import backtest  # noqa: E402

# The satellite feature is entirely optional and must never take the rest
# of the app down. If anything about it fails to import (a typo, a missing
# optional dependency, whatever) we log it once and every /api/satellite/*
# route below returns a clear 503 instead of the server crashing.
SATELLITE_IMPORT_ERROR = None
try:
    from satellite import config as sat_config  # noqa: E402
    from satellite import pipeline as sat_pipeline  # noqa: E402
    from satellite import scheduler as sat_scheduler  # noqa: E402
    from satellite import storage as sat_storage  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    SATELLITE_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# Grad-CAM is also entirely optional (needs torch + the real model.pth) and
# must never take the rest of the app down - same defensive pattern as above.
GRADCAM_IMPORT_ERROR = None
try:
    import gradcam as gradcam_module  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    GRADCAM_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# ERA5 (real Copernicus reanalysis fetch) is also entirely optional - it needs
# cdsapi + xarray installed AND the operator's own free CDS API key in
# ~/.cdsapirc. era5.py itself never crashes (see its module docstring); this
# import can only fail if era5.py's own file has a bug, not from missing
# cdsapi/xarray (those are handled inside era5.py so /api/era5/status can
# still report exactly what's missing).
ERA5_IMPORT_ERROR = None
try:
    import era5  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    ERA5_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", BACKEND_DIR.parent)).resolve()
SAMPLES_DIR = BACKEND_DIR / "samples"
TRACK_CSV = BACKEND_DIR / "data" / "track_data.csv"

MAX_BODY_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
STATIC_EXT = {".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".map", ".json"}
# fixed MIME types: the Windows registry can report wrong ones (e.g. .css as text/plain)
MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".json": "application/json",
        ".map": "application/json"}


# ------------------------------------------------------------- predictor ----
class Predictor:
    """Wraps predict.predict(). Falls back to clearly-labelled placeholders."""

    def __init__(self):
        self.fn = None
        self.error = None
        self.lock = threading.Lock()
        try:
            from predict import predict  # loads model.pth when imported
            self.fn = predict
        except Exception as exc:  # model.pth missing, torch not installed, ...
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def loaded(self):
        return self.fn is not None

    def classify(self, image):
        """Returns (class_index, category, confidence, probs, source)."""
        image = image.convert("RGB")
        if self.loaded:
            with self.lock:
                category, confidence, probs = self.fn(image)
            probs = {c: float(probs[c]) for c in logic.CLASS_ORDER}
            return logic.CLASS_ORDER.index(category), category, float(confidence), probs, "model"
        seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
        p = np.random.default_rng(seed).dirichlet(np.ones(5) * 0.6)
        idx = int(p.argmax())
        probs = {c: round(float(p[i]) * 100, 2) for i, c in enumerate(logic.CLASS_ORDER)}
        return idx, logic.CLASS_ORDER[idx], float(p[idx]) * 100, probs, "placeholder"


PREDICTOR = None
TRACKS = None
SAMPLES = None
_demo_counter = 0
_demo_lock = threading.Lock()


def init_state():
    global PREDICTOR, TRACKS, SAMPLES
    PREDICTOR = Predictor()
    TRACKS = logic.TrackDB(TRACK_CSV)
    SAMPLES = pd.read_csv(SAMPLES_DIR / "samples.csv")
    SAMPLES["dt"] = pd.to_datetime(SAMPLES["time"].astype(str), format="%Y%m%d%H")
    try:
        ensure_dirs = getattr(__import__("era5"), "ensure_dirs", None)
        if callable(ensure_dirs):
            ensure_dirs()
    except Exception:
        pass
    for _mp in (BACKEND_DIR / "model_v3b.onnx", BACKEND_DIR / "model_v4_seed1.onnx"):
        try:
            if _mp.is_file():
                print(f"Model {_mp.name} SHA256: {hashlib.sha256(_mp.read_bytes()).hexdigest()}")
            else:
                print(f"Model {_mp.name} MISSING")
        except Exception as _e:
            print(f"Model {_mp.name} hash failed: {_e}")


# ---------------------------------------------------------------- helpers ---
class BadRequest(Exception):
    pass


def number(body, key, default, lo, hi):
    """Read a number from the request with range checking."""
    raw = body.get(key, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number.")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}.")
    return value


def optional_number(body, key, lo, hi):
    """Like number(), but returns None (not a forced default) when the field
    is absent/empty - used for sst/wind_shear, which are genuinely optional."""
    raw = body.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number.")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}.")
    return value


def decode_image(data_url):
    """data URL / base64 string -> PIL image, with size checks."""
    if not isinstance(data_url, str) or not data_url:
        raise BadRequest("'image' must be a base64 data URL.")
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        raise BadRequest("The image data is not valid base64.")
    if len(raw) > 10 * 1024 * 1024:
        raise BadRequest("Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
            raise BadRequest("Image dimensions are too large (max 4096x4096).")
        img.load()
    except BadRequest:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise BadRequest("Could not read the file as an image (use PNG or JPG).")
    return img


def sample_row(filename):
    rows = SAMPLES[SAMPLES["filename"] == filename]
    if rows.empty:
        raise BadRequest("Unknown sample image.")
    return rows.iloc[0]


def run_prediction(body):
    lat = number(body, "latitude", 15.5, -90, 90)
    lon = number(body, "longitude", 85.0, -180, 360)
    wind_in = number(body, "wind", 75, 0, 250)
    pressure_in = number(body, "pressure", 980, 850, 1050)
    # sst/wind_shear/humidity feed the rule-based environment heuristic (see
    # logic.environment_favors_intensification); vorticity is informational only.
    # All four can come from the operator's own typed values, or (see
    # /api/era5/fetch) from a real ERA5 reanalysis pull - environment_source
    # records which, purely for honest labelling in the response.
    sst = optional_number(body, "sst", -5, 45)
    wind_shear = optional_number(body, "wind_shear", 0, 250)
    humidity = optional_number(body, "humidity", 0, 100)
    vorticity = optional_number(body, "vorticity", -1, 1)
    sst_land = body.get("sst_land") is True
    environment_source = str(body.get("environment_source") or "input")
    if environment_source not in ("input", "era5"):
        environment_source = "input"
    warnings = []

    image, sample_name, exclude, past = None, None, [], []
    image_source, source_storm_id = None, None
    if body.get("sample"):
        row = sample_row(str(body["sample"]))
        image = Image.open(SAMPLES_DIR / row["filename"])
        image.load()
        sample_name = row["filename"]
        exclude = [row["storm_id"]]
        past = TRACKS.past_track(row["storm_id"], row["dt"])
        image_source = "built-in TCIR sample"
        source_storm_id = str(row["storm_id"])
    elif body.get("image"):
        image = decode_image(body["image"])
        sample_name = "uploaded image"
        image_source = "user upload"

    # Doc item 6 ("Data provenance"): itemized fields the frontend renders as
    # a small "Image source / Storm / Model / Prediction source" panel, in
    # addition to (not replacing) the existing prose provenance line.
    extra = {
        "image": sample_name,
        "model_loaded": PREDICTOR.loaded,
        "image_source": image_source,
        "source_storm_id": source_storm_id,
        "model_name": "PRISM-TC EfficientNet-B0",
    }
    if image is not None:
        try:
            ood = ood_guard.assess(image)
        except Exception as exc:  # noqa: BLE001 - the guard must never break a prediction
            ood = None
            warnings.append(f"OOD guard could not run: {exc}")

        if ood is not None and ood["level"] == "likely_ood":
            # Refuse to guess. predict.py (and the DEMO MODE placeholder) always
            # returns SOME confident-looking category no matter what bytes it is
            # given - a bill, a screenshot, a random photo - because softmax
            # always sums to 100%. Silently classifying an input this unlike a
            # TCIR IR chip would present a meaningless label as if it were a
            # real result, so this stops before the classifier ever runs. See
            # ood_guard.py's docstring for exactly what triggered this.
            return {
                "rejected": True,
                "reason": "Invalid input: this does not look like a TCIR-style tropical "
                          "cyclone infrared image, so no cyclone category is shown rather "
                          "than guessing one.",
                "ood": ood,
                "meta": {
                    "mode": "image",
                    "image": sample_name,
                    "model_loaded": PREDICTOR.loaded,
                    "image_source": image_source,
                    "source_storm_id": source_storm_id,
                    "model_name": "PRISM-TC EfficientNet-B0",
                    "position": [lat, lon],
                    "warnings": ["Rejected by the out-of-distribution guard before reaching "
                                 "the classifier - see 'ood' for exactly why."],
                },
            }

        if ood is not None:
            extra["ood"] = ood
            if ood["level"] == "warning":
                warnings.append(
                    "Input image is a bit outside the usual range for a TCIR-style IR "
                    "chip (see 'ood' in meta).")

        idx, cat, conf, probs, source = PREDICTOR.classify(image)
        if source == "placeholder":
            warnings.append("DEMO MODE: model.pth not found - category and confidence are placeholders.")
        if conf < 50:
            warnings.append("Low confidence: the model is unsure between classes.")

        if body.get("gradcam"):
            if source != "model":
                warnings.append("Grad-CAM was requested but is only available for a real "
                                 "model prediction (not DEMO MODE placeholders).")
            elif GRADCAM_IMPORT_ERROR:
                warnings.append(f"Grad-CAM was requested but is unavailable on this server: "
                                 f"{GRADCAM_IMPORT_ERROR}")
            else:
                try:
                    extra["gradcam_image"] = gradcam_module.explain(image, idx)
                except Exception as exc:  # noqa: BLE001 - Grad-CAM must never break a prediction
                    warnings.append(f"Grad-CAM overlay failed: {exc}")

        return logic.build_response(
            class_index=idx, category=cat, confidence=conf, probs=probs,
            lat=lat, lon=lon, wind_input=wind_in, pressure_input=pressure_in,
            mode="image", source=source, tracks=TRACKS, exclude_storms=exclude,
            past_track=past, warnings=warnings, extra_meta=extra,
            sst=sst, wind_shear=wind_shear, humidity=humidity, vorticity=vorticity,
            environment_source=environment_source, sst_land=sst_land)

    # no image: classify from the wind the user typed (rule, not AI)
    idx = logic.class_from_wind(wind_in)
    warnings.append("No image provided: category comes from the entered wind speed "
                    "(IMD rule), not from the AI model. Add an image for an AI prediction.")
    return logic.build_response(
        class_index=idx, category=logic.CLASS_ORDER[idx], confidence=None, probs=None,
        lat=lat, lon=lon, wind_input=wind_in, pressure_input=pressure_in,
        mode="rule-based", source="rule", tracks=TRACKS, warnings=warnings,
        extra_meta=extra, sst=sst, wind_shear=wind_shear, humidity=humidity,
        vorticity=vorticity, environment_source=environment_source, sst_land=sst_land)


def run_demo(query):
    global _demo_counter
    with _demo_lock:
        try:
            i = _demo_counter if "i" not in query else int(query["i"][0])
        except ValueError:
            raise BadRequest("'i' must be a whole number.")
        _demo_counter += 1
    row = SAMPLES.iloc[i % len(SAMPLES)]
    resp = run_prediction({
        "sample": row["filename"],
        "latitude": float(row["lat"]),
        "longitude": float(row["lon"]),
    })
    resp["meta"]["true_class"] = row["true_class"]
    resp["meta"]["demo_index"] = int(i % len(SAMPLES))
    return resp


# ---------------------------------------------------------------- handler ---
class Handler(BaseHTTPRequestHandler):
    server_version = "PRISM-TC/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # -- output helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        try:
            data = path.read_bytes()
        except OSError:
            return self._json(404, {"error": "Not found"})
        ctype = MIME.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _static(self, url_path):
        rel = unquote(url_path).lstrip("/") or "index.html"
        if rel.startswith("samples/"):
            base, rel = SAMPLES_DIR, rel[len("samples/"):]
        else:
            base = FRONTEND_DIR
        target = (base / rel).resolve()
        inside = base in target.parents or target == base
        hidden = any(part.startswith(".") for part in target.relative_to(base).parts) if inside else True
        in_backend = BACKEND_DIR in target.parents and base != SAMPLES_DIR
        if (not inside or hidden or in_backend or not target.is_file()
                or target.suffix.lower() not in STATIC_EXT):
            return self._json(404, {"error": "Not found"})
        return self._file(target)

    # -- verbs
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        print("DEBUG: do_GET called:", self.path, flush=True)
        url = urlparse(self.path)
        try:
            if url.path == "/api/health":
                return self._json(200, {
                    "status": "ONLINE" if PREDICTOR.loaded else "DEMO MODE",
                    "model_loaded": PREDICTOR.loaded,
                    "model": "EfficientNet-B0 wind-regression ensemble (v3b + v4_seed1)",
                    "classes": logic.CLASS_ORDER,
                    "test_accuracy_note": "53.7% exact-match / 93.1% within-one-class on a 12-storm, 520-image storm-wise held-out test",
                    "detail": None if PREDICTOR.loaded else PREDICTOR.error,
                    "gradcam_available": GRADCAM_IMPORT_ERROR is None and PREDICTOR.loaded,
                    "era5_available": ERA5_IMPORT_ERROR is None and era5.is_available(),
                    "model_sha256": _model_hashes(),
                })
            if url.path == "/api/samples":
                return self._json(200, [
                    {"filename": r["filename"], "true_class": r["true_class"],
                     "storm_id": r["storm_id"], "lat": float(r["lat"]),
                     "lon": float(r["lon"]), "url": f"/samples/{r['filename']}"}
                    for _, r in SAMPLES.iterrows()])
            if url.path == "/api/demo":
                return self._json(200, run_demo(parse_qs(url.query)))
            if url.path == "/api/model/transparency":
                return self._json(200, model_card.get_model_card())
            if url.path == "/api/backtest/storms":
                return self._json(200, backtest.list_backtestable_storms(TRACKS))
            if url.path in ("/api/satellite/status", "/api/satellite/latest"):
                return self._satellite_status()
            if url.path == "/api/satellite/metadata":
                return self._satellite_metadata()
            if url.path == "/api/satellite/image":
                return self._satellite_image()
            if url.path == "/api/satellite/history":
                return self._satellite_history()
            if url.path == "/api/era5/status":
                return self._era5_status()
            if url.path == "/api/era5/fetch":
                return self._era5_fetch(parse_qs(url.query))
            if url.path == "/api/era5/job":
                return self._era5_job(parse_qs(url.query))
            if url.path.startswith("/api/"):
                return self._json(404, {"error": "Unknown API route"})
            return self._static(url.path)
        except BadRequest as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # never crash the server
            self.log_message("ERROR %s: %s", type(exc).__name__, exc)
            return self._json(500, {"error": "Internal server error while processing the request."})

    # -- satellite routes (kept separate; each one is defensive on its own
    #    so a bug here can't take down /api/predict or /api/demo above)
    def _satellite_unavailable(self):
        return self._json(503, {
            "error": "Satellite module is not available on this server.",
            "detail": SATELLITE_IMPORT_ERROR,
        })

    def _satellite_status(self):
        if SATELLITE_IMPORT_ERROR:
            return self._satellite_unavailable()
        try:
            return self._json(200, sat_storage.read_state())
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"Could not read satellite status: {exc}"})

    def _satellite_metadata(self):
        if SATELLITE_IMPORT_ERROR:
            return self._satellite_unavailable()
        state = sat_storage.read_state()
        return self._json(200, {
            "source": state.get("source"),
            "satellite": state.get("satellite"),
            "product": state.get("product"),
            "channel": state.get("channel"),
            "observation_time_utc": state.get("observation_time_utc"),
            "received_time_utc": state.get("received_time_utc"),
            "processed_time_utc": state.get("processed_time_utc"),
            "region": state.get("region"),
        })

    def _satellite_image(self):
        if SATELLITE_IMPORT_ERROR:
            return self._satellite_unavailable()
        path = sat_config.LATEST_DIR / "latest.png"
        if not path.is_file():
            return self._json(404, {"error": "No satellite image has been captured yet."})
        return self._file(path)

    def _satellite_history(self):
        if SATELLITE_IMPORT_ERROR:
            return self._satellite_unavailable()
        try:
            return self._json(200, sat_storage.read_history())
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"Could not read satellite history: {exc}"})

    # -- ERA5 routes (kept separate and fully defensive - a failure here can
    #    never affect /api/predict, which still works with typed sst/shear values)
    def _era5_status(self):
        if ERA5_IMPORT_ERROR:
            return self._json(503, {"ready": False, "error": ERA5_IMPORT_ERROR})
        try:
            return self._json(200, era5.status())
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"ready": False, "error": f"Could not read ERA5 status: {exc}"})

    def _era5_fetch(self, query):
        if ERA5_IMPORT_ERROR:
            return self._json(503, {"error": f"ERA5 module unavailable: {ERA5_IMPORT_ERROR}"})
        try:
            if not query.get("lat") or not query.get("lon"):
                raise BadRequest("'lat' and 'lon' query parameters are required.")
            lat = float(query["lat"][0])
            lon = float(query["lon"][0])
            if not (math.isfinite(lat) and -90 <= lat <= 90):
                raise BadRequest("'lat' must be a number between -90 and 90.")
            if not (math.isfinite(lon) and -180 <= lon <= 360):
                raise BadRequest("'lon' must be a number between -180 and 360.")
        except (TypeError, ValueError):
            return self._json(400, {"error": "'lat'/'lon' must be numbers."})
        except BadRequest as exc:
            return self._json(400, {"error": str(exc)})
        try:
            job = era5.submit_fetch(lat, lon)
            return self._era5_job_payload(job)
        except Exception as exc:  # noqa: BLE001 - never let this crash the server
            self.log_message("ERROR era5 fetch: %s", exc)
            return self._json(500, {"error": f"Unexpected ERA5 error: {exc}"})

    def _era5_job_payload(self, job):
        status = job["status"]
        if status == "done":
            return self._json(200, job)
        if status in ("queued", "running"):
            return self._json(202, job)
        return self._json(503 if not job.get("configured", True) else 502, job)

    def _era5_job(self, query):
        if ERA5_IMPORT_ERROR:
            return self._json(503, {"error": f"ERA5 module unavailable: {ERA5_IMPORT_ERROR}"})
        job_id = (query.get("id") or [""])[0]
        if not job_id:
            return self._json(400, {"error": "'id' query parameter is required."})
        try:
            job = era5.get_job(job_id)
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"Could not read ERA5 job: {exc}"})
        if job is None:
            return self._json(404, {"error": "Unknown or expired job id - submit again."})
        return self._era5_job_payload(job)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/backtest/run":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                if not isinstance(body, dict) or not body.get("storm_id"):
                    raise BadRequest("'storm_id' is required.")
                result = backtest.run_backtest(TRACKS, body["storm_id"])
                return self._json(400 if "error" in result else 200, result)
            except BadRequest as exc:
                return self._json(400, {"error": str(exc)})
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": "Request body must be valid JSON."})
            except Exception as exc:  # noqa: BLE001
                self.log_message("ERROR backtest: %s", exc)
                return self._json(500, {"error": f"Backtest failed: {exc}"})
        if url.path == "/api/satellite/update":
            if SATELLITE_IMPORT_ERROR:
                return self._satellite_unavailable()
            try:
                return self._json(200, sat_pipeline.run_once())
            except Exception as exc:  # noqa: BLE001 - manual trigger must never 500-crash the process
                self.log_message("ERROR satellite manual update: %s", exc)
                return self._json(500, {"error": f"Manual satellite update failed: {exc}"})
        if url.path != "/api/predict":
            return self._json(404, {"error": "Unknown API route"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise BadRequest("Empty request body.")
            if length > MAX_BODY_BYTES:
                return self._json(413, {"error": "Request too large (max 15 MB)."})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise BadRequest("Request body must be valid JSON.")
            if not isinstance(body, dict):
                raise BadRequest("Request body must be a JSON object.")
            return self._json(200, run_prediction(body))
        except BadRequest as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            self.log_message("ERROR %s: %s", type(exc).__name__, exc)
            return self._json(500, {"error": "Internal server error while processing the request."})


def _model_hashes():
    out = {}
    for name in ("model_v3b.onnx", "model_v4_seed1.onnx"):
        p = BACKEND_DIR / name
        try:
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
        except Exception as exc:
            out[name] = f"ERROR: {exc}"
    return out


def make_server(host="127.0.0.1", port=8000):
    init_state()
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    httpd = make_server(host, port)
    print(f"PRISM-TC running at http://{'localhost' if host == '127.0.0.1' else host}:{port}")
    print("Model:", "LOADED" if PREDICTOR.loaded else f"NOT LOADED (demo mode) - {PREDICTOR.error}")
    print("Serving frontend from:", FRONTEND_DIR)
    if SATELLITE_IMPORT_ERROR:
        print("Satellite: DISABLED (import failed) -", SATELLITE_IMPORT_ERROR)
    else:
        try:
            sat_scheduler.start()
        except Exception as exc:  # noqa: BLE001 - a scheduler failure must never stop the server
            print("Satellite: scheduler failed to start -", exc)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
