# CLAUDE.md - PRISM-TC (SIH26070) frontend + backend

Read this fully before changing anything. You are helping **Member 4** (frontend/integration)
finish a Smart India Hackathon prototype. The hackathon is soon; prefer small, safe changes.

## What the project is
A web dashboard that takes an infrared satellite image of a tropical cyclone and shows:
- the predicted intensity class (5 IMD classes) + confidence + class probabilities  -> **REAL** (AI model)
- a 24 h track outlook, wind outlook, pressure, risk index, rapid-intensification flag -> **SIMULATED**
  (historical-analog method and simple rules; NOT machine-learning forecasts)

Classes (index -> name): 0 Depression, 1 Cyclonic Storm, 2 Severe Cyclonic Storm,
3 Very Severe Cyclonic Storm, 4 Extremely Severe/Super Cyclone.

## Honesty rules (do not break these)
- Keep the MODEL / SIMULATED / EST. / PLACEHOLDER tags. Never present simulated values as AI output.
- Do NOT claim CNN+GRU or ERA5-fused AI predictions. Only TCIR infrared images + IBTrACS train/drive
  the classifier and analog outlook. ERA5 (see `backend/era5.py`) is real but LIMITED SCOPE: it can
  fetch real SST/wind-shear/humidity/vorticity for the existing rule-based environment check only -
  it never touches the AI classifier. (The original frontend text claimed CNN+GRU/ERA5 fusion; it
  was corrected on purpose - do not reintroduce that claim.)
- Model accuracy: about 60% on 372 unseen test images (macro-F1 0.49, average error 0.51 severity
  levels). Test set is 50% "Depression" (always guessing it scores 50%) and has only 10 Very Severe
  and 6 Extremely Severe images. The confidence score is overconfident (ECE ~27%): it is NOT the
  true probability of being right.
- If model.pth is missing the backend runs in DEMO MODE and every response is marked "placeholder".

## Files
```
index.html, style.css, app.js     dashboard (Leaflet map, Chart.js chart, no build step)
backend/server.py                 stdlib HTTP server (no Flask): serves the page AND /api/*
backend/logic.py                  class-from-wind rule, pressure estimate, risk index, analog track
backend/predict.py                loads model.pth, predict(PIL image) -> (category, confidence, probs)
backend/model.pth                 NOT IN REPO YET - copy from Drive scripts/ (final model)
backend/data/track_data.csv       historical positions/winds/classes (IBTrACS-derived)
backend/samples/                  10 test images + samples.csv (true class, storm id, lat, lon)
backend/tests/smoke_test.py       HTTP tests (uses a fake model unless --real)
backend/era5.py                   optional real ERA5 reanalysis fetch (needs cdsapi+xarray+CDS key)
backend/requirements-era5.txt     optional extra deps for era5.py (NOT in the base requirements.txt)
backend/tests/test_era5.py        offline tests for era5.py + the humidity rule (no cdsapi needed)
_original_frontend_backup/        Member 4's original files (reference only)
ROADMAP.md                        phased vision beyond this prototype (Phase 1) - honest, not a claim
```

## Run
```
pip install -r backend/requirements.txt
# copy the final model.pth into backend/
python backend/server.py          # http://localhost:8000
python backend/tests/smoke_test.py            # backend checks with fake model
python backend/tests/smoke_test.py --real     # with the real model.pth
```
Env vars: HOST (default 127.0.0.1), PORT (default 8000). If the page is opened with VS Code Live
Server (port 5500) or from a file, app.js talks to http://localhost:8000/api (override: `?api=URL`).

## API contract (frontend <-> backend)
- `GET /api/health` -> `{status: "ONLINE"|"DEMO MODE", model_loaded, model, classes, detail}`
- `GET /api/samples` -> `[{filename, true_class, storm_id, lat, lon, url}]`
- `GET /api/demo[?i=N]` -> same shape as predict, using the next built-in sample (`meta.true_class` included)
- `POST /api/predict` JSON body:
  `latitude, longitude, wind, pressure` (numbers, validated), optional `sst, wind_shear, humidity`
  (all three now feed `logic.environment_favors_intensification` - humidity only downgrades an
  otherwise-"favorable" verdict when dry, never upgrades one), optional `vorticity` (informational
  only, shown but never used in the verdict), optional `environment_source` (`"input"` default or
  `"era5"` - purely a label for where the four numbers above came from, echoed back in
  `meta.environment_input_source`), and one of `sample` (filename from /api/samples) or
  `image` (base64 data URL, PNG/JPG, max 10 MB). No image => category is derived from `wind` by the
  IMD rule (`meta.mode = "rule-based"`, confidence null).
- Response: `{prediction:{category,class_index,confidence,wind,wind_display,pressure,risk_index,
  rapid_intensification,environment_favorability}, probabilities:{class:pct}, track:[[lat,lon]...],
  past_track:[[lat,lon]...], forecast:[{time:"T+3h",wind}], sources:{category,confidence,wind,
  pressure,risk,track,forecast,rapid,environment_favorability}, meta:{mode,position,analog_storm,
  warnings,environment_input_source,image,model_loaded}}`
- Errors: `{error: "message"}` with HTTP 400 (bad input) / 404 / 413 / 500. The UI shows them in a toast.
- `sources.*` values drive the tags in the UI: model | placeholder | rule | estimate | input | simulated.
- `GET /api/era5/status` -> `{enabled,packages_installed,credentials_found,ready,import_error,
  lag_days,note}` - never makes a network call, safe to poll.
- `GET /api/era5/fetch?lat=..&lon=..` -> real ERA5 values for that point:
  `{sst_c,wind_shear_kt,humidity_pct,vorticity_850_s1,valid_time_utc,requested_time_utc,lag_days,
  lat,lon,cached,source}`. 503 with `{error,configured:false}` if cdsapi/xarray aren't installed or
  no `~/.cdsapirc` was found; 502 with `{error,configured:true}` if a real attempt failed (network,
  CDS queue, bad response); 400 if lat/lon are missing/invalid. See `backend/era5.py` for the full
  honesty notes (it is a reanalysis, not real-time - always several days behind "now").

## Rules for the AI model
- `backend/predict.py` must NOT apply ImageNet normalization. The shipped checkpoint was trained
  without it; adding `transforms.Normalize` drops accuracy from ~60% to ~30%.
- Preprocessing = convert to RGB, Resize((224,224)), ToTensor. Do not change without retraining.
- The model file must be called `model.pth` and sit next to `predict.py`.

## State of the work (what is done)
- Frontend restyled/extended: image upload + sample picker, probability bars, REAL/SIMULATED tags,
  honest data-source cards, error toasts, graceful offline fallbacks (SVG track + canvas chart when
  Leaflet/Chart.js cannot load).
- Backend written and tested with a fake model (33 HTTP checks pass): validation, error handling,
  path-traversal protection, CORS.
- Verified in headless Chromium **without internet**: layout, demo, upload, rule-based mode, error
  toast, mobile layout. NOT verified: the Leaflet map and Chart.js paths (CDNs were blocked in the
  test environment) and the real model (torch/model.pth were not available there).

## What Member 4 still has to do (in this order)
1. **Copy the final model.pth into backend/** (from the team Drive `scripts/`). Install requirements,
   run `python backend/server.py`. The top-left status must say "ONLINE - model loaded". Run
   `python backend/tests/smoke_test.py --real`. Fix any error (send it to Member 2 if it is about the model).
2. **Test with internet** (Leaflet + Chart.js + fonts load from CDNs): the map must show tiles, the dashed
   orange simulated track and the red current-position dot; the wind chart must draw. Fix any console error.
   Then test at least 5 different sample images and 2 uploaded images: no crash, correct image preview,
   class + confidence + probability bars, reasonable speed (first call is slow: model loads), track + chart.
   Also press "Generate AI Prediction" with no image (rule-based) and with a bad latitude (toast).
3. **Make the demo work offline** (venue Wi-Fi may fail): download leaflet.js, leaflet.css (+ its images/),
   and chart.umd.js once, put them in a `vendor/` folder and point the `<link>/<script>` tags in index.html
   at them (keep the CDN as a fallback if you like). Map tiles still need internet; without them the
   track is drawn on a dark background, which is acceptable.
4. **Integrate Member 5's trajectory work** (if it exists): replace `TrackDB.analog_outlook` in
   `backend/logic.py` (or add a second function) so `track` / `forecast` come from Member 5's output.
   Keep the response schema above and keep those values tagged SIMULATED unless they are truly
   model-based. If Member 5 delivers nothing, keep the analog outlook.
5. **Deploy or run locally**: Streamlit Cloud cannot host this. Simplest and safest = run locally on the
   demo laptop. Optional public hosting: Render "Web Service", build `pip install -r backend/requirements.txt`,
   start `python backend/server.py`, env `HOST=0.0.0.0` (Render sets PORT). model.pth (16 MB) must be in the repo.
6. **Record a backup demo video** (Win+G Game Bar): Run Demo, pick a sample, upload an image, show the
   REAL vs SIMULATED tags, show the notice bar. 1-2 minutes. Do this before the presentation.
7. Optional polish only after 1-6: favicon, print/screenshots for the pitch deck.

## Pitch numbers (use these)
~60% accuracy on 372 unseen test images, macro-F1 0.49, average error about half a severity level,
big improvement on Severe Cyclonic Storm (27/64 correct vs 8/64 for the first model); limits: one IR
channel, small dataset, TCIR only (not live INSAT/MOSDAC), overconfident scores, track/risk are simulated.

## Do not
- Do not delete `_original_frontend_backup/`. - Do not add normalization to predict.py.
- Do not rename model.pth. - Do not remove the REAL/SIMULATED tags or the notice bar.
- Do not commit secrets. - Do not change the API field names without updating app.js and the tests.

## Near-real-time satellite (added later - read backend/satellite/README.md for the full story)
A self-contained `backend/satellite/` package adds a "Live / Near-Real-Time Satellite" card to the
dashboard, fed by IMD's public INSAT-3D/3DS Asia-sector infrared quicklook (free, no account, India's
own satellite - see README section 0 for why this replaced an earlier Himawari-based attempt). It is
intentionally isolated: if it fails or is disabled, `/api/predict`, `/api/demo` and the rest of the app
are completely unaffected (see the try/except around its import in `server.py`).

Honest limitations you should know before demoing this:
- It's a quicklook picture, not calibrated data, and the model (`predict.py`) is NOT run on it by
  default (`ENABLE_LIVE_INFERENCE=false`) because the model was only trained/validated on tightly
  cropped, storm-centred TCIR chips - not wide regional live imagery. Turning this on shows an
  EXPERIMENTAL, unvalidated result, clearly separated from the real TCIR-based prediction.
- No custom geographic crop is applied - IMD's own "Asia sector" framing already covers the whole
  Arabian Sea and Bay of Bengal in one image, so the full downloaded frame is used as-is. `GET
  /api/satellite/status` reports this via `region.description` (a text description, not a fabricated
  coverage percentage) rather than pretending to have computed a precise crop.
- The exact satellite scan time is not OCR'd from the image's own timestamp banner; the HTTP
  `Last-Modified` header is used if the server sends one, otherwise the fetch time is used and
  honestly labelled as such (`region`/`metadata` -> check `extra.timestamp_source`).
- The live-network path has only been verified manually once (a teammate opened the IMD URL directly
  in a browser). Run `python backend/satellite/test_connection.py` on a machine with real internet
  before trusting it in a demo, and read the printed instructions about eyeballing
  `test_output_imd.jpg` against the real coastline.
- New endpoints: `GET /api/satellite/{status,latest,metadata,image,history}`, `POST /api/satellite/update`.
- New env vars: see `.env.example`. New test: `python backend/tests/smoke_test_satellite.py` (self-contained,
  no internet needed - it deliberately checks that a failed fetch reports ERROR rather than faking success).

## UX/credibility features (added later - read this before demoing any of them)
Built from a third-party UX review of this project against the SIH problem statement. All of
this is additive - nothing above changed behavior. Fully testable in a no-torch, no-internet
sandbox: `backend/model_card.py`, `backend/ood_guard.py`, `backend/backtest.py`, the SST/wind-shear
rule in `backend/logic.py`, and the torch-free half of `backend/calibration.py`. Needs the real
model.pth + torch on your own machine, never run in the sandbox that wrote this code:
`backend/fit_calibration.py` and `backend/gradcam.py`/`gradcam_demo.py`.

- **Model Transparency panel** - `GET /api/model/transparency` (`backend/model_card.py`) reports
  the same accuracy/macro-F1/ECE numbers as this file's "Pitch numbers" section, plus known
  failure modes, as structured JSON. New dashboard card renders it. If you retrain the model,
  update the numbers in `model_card.py` AND in this file's "Pitch numbers" section together.
- **Out-of-distribution (OOD) input guard** - `backend/ood_guard.py`. A heuristic (brightness,
  contrast, "colorfulness"), calibrated from `backend/samples/*.png`, that flags an uploaded
  image that looks nothing like a TCIR IR chip (a photo, a screenshot, a solid color, a document).
  NOT a trained OOD detector - it will miss some odd inputs and can flag a legitimate unusual one.
  Two severities: `"warning"` (a bit outside the usual range) still gets classified normally, with
  a caution banner under the image preview and `meta.ood` on `/api/predict`. `"likely_ood"` (very
  unlike a TCIR chip - a photo, a document, a screenshot) is a **hard block**: `run_prediction()`
  in `server.py` returns early with `{"rejected": true, "reason": ..., "ood": {...}}` and never
  calls the classifier at all. This exists because the classifier (and the DEMO MODE placeholder)
  always returns SOME confident-looking category no matter what bytes it's given - softmax always
  sums to 100% - so a real user got a "Severe Cyclonic Storm" reading from a photo of an electric
  bill before this was added. The frontend (`app.js`'s `renderRejection()`) shows a plain "Not
  classifiable" / "REJECTED" state instead of any numbers when `data.rejected` is true. This does
  NOT improve the classifier's accuracy on real-but-different cyclone photos (a depression image
  misread as super cyclone, say) - it only stops the app from confidently labeling inputs that
  aren't TCIR-format infrared chips at all. Improving accuracy on legitimate but off-distribution
  real cyclone photos would need retraining on a broader image set, which is a separate, larger
  effort from this guard.
  **Threshold history**: brightness `_STRONG_PAD` was originally 55.0, then lowered to 35.0 after
  a real user upload (a bright, near-grayscale document/bill image: brightness 244.1 vs a
  reference max of 199.9, diff 44.2) landed in the softer "warning" band and still returned a
  full confident classification. A compound-signal rule was added at the same time: two or more
  independently-borderline "warning" signals (e.g. brightness AND contrast each a bit off, but
  neither extreme alone) now escalate together to `likely_ood`, since that combination is itself
  evidence against a real TCIR chip even when no single measurement crosses the strong threshold.
  All 10 bundled TCIR samples were re-verified against both changes with zero false rejections.
  This is still a 3-feature heuristic, not a trained detector - it will keep missing OOD inputs
  that happen to match TCIR's brightness/contrast/color profile by coincidence.
- **SST / wind-shear / humidity rule-based environment check** - `logic.environment_favors_intensification()`.
  SST and wind-shear inputs drive a small, documented meteorological rule of thumb (SST >= 26.5C +
  shear <= 10kt = "favorable", etc.) reported as `prediction.environment_favorability` /
  `sources.environment_favorability = "rule-based"`. Humidity now also feeds this rule (dry
  700hPa air, <40%, downgrades an otherwise-"favorable" verdict to "neutral" - one-directional,
  never upgrades an unfavorable one; see the function's docstring for the citation). Vorticity is
  informational only and is never used in the verdict. This whole check is separate from and does
  NOT change the existing analog-based `rapid_intensification` flag - the two are never conflated.
- **Real ERA5 reanalysis fetch** - `backend/era5.py`, wired as `GET /api/era5/{status,fetch}` and a
  "Fetch from ERA5" button next to the SST/shear/humidity/vorticity fields. This is REAL Copernicus
  data (not synthetic), but it is a reanalysis product with a several-day publication lag, not a
  live observation - the response's `valid_time_utc`/`lag_days` say exactly how old it is, and the
  frontend badge shows that timestamp rather than implying "now". Optional: needs
  `pip install -r backend/requirements-era5.txt` (cdsapi, xarray, netCDF4 - NOT in the base
  requirements.txt) plus the operator's own free Copernicus CDS account/API key in `~/.cdsapirc`.
  Without either, `/api/era5/status` reports `ready: false` and the fetch button shows a clear
  "not configured" message - it never breaks `/api/predict`, which keeps working with typed values.
  IMPORTANT: this code could not be exercised against a real CDS server from the sandbox it was
  written in (no internet, no CDS account there) - read the "Honesty notes" in era5.py's docstring
  before relying on it, especially the flag about the CDS request parameter name possibly having
  changed since this was written (same "unverified, confirm before trusting" spirit as the
  Himawari satellite source).
- **Backtest mode** - `GET /api/backtest/storms`, `POST /api/backtest/run {storm_id}`
  (`backend/backtest.py`). Tests `logic.TrackDB.analog_outlook()` (NOT the AI classifier) against
  a real historical storm from `backend/data/track_data.csv`, always excluding that storm from
  its own candidate pool. Reports track error (km, haversine) and wind error (kt) per 3h step.
  New dashboard card with a storm picker.
- **Confidence calibration (temperature scaling)** - `backend/calibration.py` implements the math
  (`apply_temperature`, `compute_ece`) with no torch dependency; `predict.py` applies it
  automatically (T=1.0 = no-op if never fitted, so nothing changes until you act). To actually
  fit a temperature: copy the real `model.pth` into `backend/`, then run
  `python backend/fit_calibration.py` (ideally with `--data`/`--images-dir` pointing at a larger
  held-out labelled set than the 10-image `samples/` - fitting on the same images used to report
  accuracy would be circular). It writes `backend/temperature.json`; `GET
  /api/model/transparency`'s `calibration` field then reports `fitted: true`.
- **Grad-CAM overlay** - `backend/gradcam.py`, wired as an opt-in `gradcam: true` field on
  `POST /api/predict` (only works with a real model prediction, degrades to a warning otherwise -
  see the defensive `try/except` in `server.py`, same pattern as the satellite import guard).
  Frontend shows a "Show Grad-CAM overlay" checkbox, but only once `GET /api/health` reports
  `gradcam_available: true` (real model.pth + torch + gradcam.py all working). Run
  `python backend/gradcam_demo.py` on your own machine to generate and eyeball a real example
  before showing it to judges - it is a coarse debugging heatmap, not validated attribution.
- **Positive Image Compatibility indicator** - `#oodWarning` no longer just hides on a clean pass.
  When `ood.level === "none"` (an image was provided and looked fine), it now shows a green
  "Input validation: TCIR-compatible ✓ PASSED" banner (`renderOodWarning()` in `app.js`,
  `.ood-warning.ood-ok` in `style.css`, `#icon-check-circle` symbol in `index.html`) instead of
  silence - so a clean pass is as visible as a warning or rejection, not just the absence of one.
- **Data provenance panel** - `#dataProvenance` in the AI Prediction card (`renderDataProvenance()`
  in `app.js`), itemizing exactly where every field came from: Image source (built-in TCIR sample
  vs. user upload), Storm (the sample's IBTrACS storm_id, hidden for uploads), Position (lat/lon),
  Model (`PRISM-TC EfficientNet-B0`, flagged when not loaded), and Prediction source (MODEL /
  PLACEHOLDER / RULE-BASED / REJECTED). Backend fields feeding it: `meta.image_source`,
  `meta.source_storm_id`, `meta.model_name`, `meta.position` - added in `server.py`'s
  `run_prediction()` alongside the existing `extra` dict, present on both normal and rejected
  responses. This is additive: the original prose `#provenance` line is unchanged.
- **AI vs Simulation architecture diagram** - `.arch-diagram` (inline SVG) inside the existing
  "Multi-Source Data" card, right above the source list. Shows the real split: AI Classifier (TCIR
  image in, REAL intensity class out) vs. Analytical Engine (historical analogs in, SIMULATED
  track/wind/risk out), both feeding one Dashboard box, with ERA5 and INSAT/MOSDAC drawn as
  separate dashboard-only inputs that never reach the classifier. Purely illustrative - it does
  not change any behavior, only makes the existing separation (already true in the code and
  already described in prose elsewhere on this page) visible at a glance.
- **Judge Demo Mode** - `#judgeDemo` card near the top of the dashboard, 3 one-click, guaranteed-
  to-work scenarios for a 60-90s demo that never depends on internet or having a good example
  image on hand: Scenario 1 runs `/api/predict` on the first bundled TCIR sample (real classify);
  Scenario 2 generates a synthetic bright/near-grayscale "document" image on a `<canvas>` client-
  side (`makeSyntheticDocumentDataUrl()` in `app.js`, no file needed) and confirms it gets
  REJECTED; Scenario 3 scrolls to and refreshes the existing Live Satellite card and reports
  whatever its real, honestly-reported status currently is (never faked for the demo). Each
  scenario fills a small result readout (`#judgeDemoResult`) with: accepted/rejected, whether the
  classifier was called, predicted class, confidence, OOD result, reason, and whether that
  behavior was expected - Scenario 2 explicitly flags itself with a `✗` if it is ever
  ACCEPTED instead of REJECTED, which would mean the OOD guard has regressed.
- **Export Report (print/PDF)** - `#exportReportBtn` in the AI Prediction card calls the browser's
  own `window.print()` (no new dependency, no server round-trip). `@media print` rules in
  `style.css` switch the dark dashboard theme to a light, ink-friendly one and hide everything
  that isn't part of a result: the sidebar, top bar, status strip, the whole `#environment`
  input card (upload controls, lat/lon/wind/pressure/sst/shear/humidity/vorticity fields, the
  ERA5 fetch button, Grad-CAM checkbox - anything with class `no-print`), and every section other
  than `#dashboard` (satellite/transparency/backtest/references/system, plus `#analytics` on the
  live device build). `#printReportHeader` (invisible on screen, `class="print-only"`) is filled
  by `app.js`'s `updatePrintReportHeader()` right before printing - both from the button and from
  a real `beforeprint` listener, so Ctrl+P / the browser's own Print menu gets the same header,
  not just the button - with a fresh timestamp, the current category/confidence/position, and a
  REAL-vs-SIMULATED honesty line repeated for anyone reading a printout with no access to the
  live app. Known trade-off: since the whole `#environment` card is hidden for a clean layout,
  a Grad-CAM overlay result or an OOD warning (both rendered inside that card) will not appear in
  the printed report even if one was generated on screen - acceptable for a v1, revisit if judges
  want to see those in a printout too.
- **References & Glossary panel** (`#references`) - static, hand-written citations for the actual
  papers/datasets this prototype builds on (TCIR, IBTrACS, EfficientNet, Guo et al. 2017,
  Selvaraju et al. 2017, IMD bands) plus a plain-language glossary (ECE, macro-F1, analog method,
  Grad-CAM, OOD, temperature scaling, rule-based vs. ML) - aimed at a teacher/researcher audience
  who wants to check the actual sources, not just trust a claim.
- **API Reference section** (`#apiReference`) - a static, hand-written page inside the dashboard
  documenting every HTTP route in `backend/server.py` (method, path, one-line description),
  organized into Prediction / Model evaluation & transparency / Near-real-time satellite / ERA5
  reanalysis groups, with a pointer to this file's own "API contract" section for full
  request/response field names. Keep both in sync when you add or change a route: update the API
  contract section above, then add/edit the matching `.api-endpoint` block in `index.html`.
- New tests: `backend/tests/smoke_test_features.py` (transparency/OOD/environment-rule/backtest/
  Grad-CAM-degradation, all torch-free), `backend/tests/test_calibration.py` (calibration math),
  `backend/tests/test_era5.py` (humidity-rule logic + era5.py's not-configured path + the two new
  HTTP routes - all torch-free and network-free, since cdsapi/xarray are optional add-ons).
- **Inline SVG icon sprite** (UI/UX pass) - every emoji used as a UI icon (nav items, brand mark,
  buttons, warning/status icons, the "Multi-Source Data" and "System Information" diagrams) was
  replaced with a hand-authored `<svg><use href="#icon-NAME"></use></svg>` reference. The 18
  `<symbol>` definitions live inline as the first child of `<body>` in `index.html` (a
  `<svg style="display:none">` sprite) - no icon font, no external CDN, so it still works with
  zero internet access. Styling is one shared `.icon` class in `style.css` (16x16,
  `stroke: currentColor`, `fill: none`) plus size/color overrides per container
  (`.brand-icon .icon`, `.ai-icon .icon`, `.source-icon .icon`, `.architecture .icon`). Two icons
  (`icon-play`, `icon-sparkle`) are solid glyphs and set `fill="currentColor" stroke="none"`
  directly on their own `<path>`, which correctly overrides the inherited `.icon { fill: none }`
  rule (presentation attributes beat inherited CSS). `app.js` has two spots that rewrite a
  button's `innerHTML` at runtime (`fetchFromEra5()`'s `finally` block, and `setLoading()` for the
  predict button) - both now emit the same `<svg class="icon">...` markup instead of an emoji
  literal, and the loading state uses `icon-gear` with a `.icon-spin` CSS keyframe animation
  instead of an hourglass emoji. If you add a new UI icon: draw a new `<symbol>` in the sprite,
  reference it with `<svg class="icon"><use href="#icon-NAME"></use></svg>`, and never reintroduce
  an emoji as a UI icon (plain body text can still use one, e.g. inside a written explanation).
