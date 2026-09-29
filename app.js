/* =========================================================
   PRISM-TC FRONTEND
   SIH26070

   Talks to the backend in backend/server.py:
     GET  /api/health    GET /api/samples    GET /api/demo
     POST /api/predict
   REAL      : category, confidence, class probabilities (AI model)
   SIMULATED : track, wind outlook, pressure, risk, rapid flag
   ========================================================= */

/* When the page is opened with VS Code Live Server (port 5500) or as a
   file, talk to the backend on port 8000. Override with ?api=URL */
const DEFAULT_API =
    (location.protocol === "file:" || location.port === "5500")
        ? "http://localhost:8000/api"
        : "/api";

const API_BASE =
    new URLSearchParams(location.search).get("api") || DEFAULT_API;

const API_TIMEOUT_MS = 60000;   /* first model call can be slow */

/* No build step in this project, so these are maintained by hand - bump
   both whenever a notable set of changes ships. Shown in the sidebar
   footer (#versionFooter) as a simple, honest "what am I looking at"
   stamp for anyone returning to the dashboard later. */
const APP_VERSION = "1.0.0";
const APP_BUILD_DATE = "2026-09-25";

const CLASS_ORDER = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe/Super Cyclone"
];

const CLASS_COLORS = [
    "#4C9F70", "#E0B040", "#E8873A", "#D6493B", "#8B1E3F"
];

let map;
let hasLeaflet = false;
let trackLayers = [];
let windChart;
let samples = [];

/* the image chosen by the user: null | {kind, name, sample?, dataUrl?, trueClass?} */
let selectedImage = null;
let toastTimer;


/* =========================================================
   SMALL HELPERS
   ========================================================= */

function el(id) {
    return document.getElementById(id);
}

function showError(message) {
    showToast(message, "error");
}

function showToast(message, kind) {
    const box = el("toast");
    box.textContent = message;
    box.className = "toast " + (kind || "info");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        box.classList.add("hidden");
    }, 7000);
}

function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "—";
    }
    return number.toFixed(1);
}

function setTag(id, text, kind) {
    const tag = el(id);
    if (!text) {
        tag.classList.add("hidden");
        return;
    }
    tag.textContent = text;
    tag.className = "tag " + kind;
}


/* =========================================================
   INITIALIZE MAP
   ========================================================= */

function initializeMap() {

    hasLeaflet = typeof L !== "undefined";

    if (!hasLeaflet) {
        /* offline / CDN blocked: we draw a simple SVG plot instead */
        el("map").innerHTML =
            '<div class="map-fallback">Map library not loaded ' +
            "(no internet?). A simple track plot will be shown.</div>";
        return;
    }

    map = L.map("map").setView(
        [15.5, 85],
        5
    );

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution:
                "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);

}


/* =========================================================
   READ FORM VALUES
   ========================================================= */

function getInputs() {

    return {

        latitude:
            Number(el("latitude").value),

        longitude:
            Number(el("longitude").value),

        sst:
            Number(el("sst").value),

        wind_shear:
            Number(el("shear").value),

        pressure:
            Number(el("pressureInput").value),

        wind:
            Number(el("windInput").value),

        humidity:
            Number(el("humidity").value),

        vorticity:
            Number(el("vorticity").value),

        environment_source:
            era5Active ? "era5" : "input"
    };

}


/* =========================================================
   ERA5 REAL REANALYSIS FETCH
   Populates SST / wind shear / humidity / vorticity from the real
   Copernicus ERA5 dataset via GET /api/era5/fetch (see backend/era5.py).
   Purely additive: if it fails (not configured on this server, no
   internet, CDS queue slow), the four fields keep whatever the operator
   typed and nothing else in the app is affected.
   ========================================================= */

let era5Active = false;   /* true only right after a successful ERA5 fetch */

function clearEra5Badge() {
    era5Active = false;
    const badge = el("era5Badge");
    if (badge) {
        badge.classList.add("hidden");
        badge.textContent = "";
    }
}

async function fetchFromEra5() {

    const btn = el("era5FetchBtn");
    const statusEl = el("era5Status");
    const lat = Number(el("latitude").value);
    const lon = Number(el("longitude").value);

    btn.disabled = true;
    btn.textContent = "Fetching from ERA5...";
    statusEl.textContent = "Contacting the Copernicus Climate Data Store - this can take up to a minute or more if the request is queued.";
    statusEl.classList.remove("error");
    clearEra5Badge();

    try {

        const result = await apiRequest(
            `/era5/fetch?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`
        );

        el("sst").value = result.sst_c;
        el("shear").value = result.wind_shear_kt;
        el("humidity").value = result.humidity_pct;
        el("vorticity").value = result.vorticity_850_s1;

        era5Active = true;
        const badge = el("era5Badge");
        badge.textContent = `ERA5 REAL DATA · valid ${result.valid_time_utc || result.requested_time_utc}`;
        badge.classList.remove("hidden");

        statusEl.textContent =
            `Loaded from ERA5 (Copernicus reanalysis, ${result.lag_days}-day typical lag - ` +
            `not a live observation). ${result.cached ? "Served from local cache." : ""}`;

    } catch (error) {

        clearEra5Badge();
        statusEl.textContent = error.message || "ERA5 fetch failed.";
        statusEl.classList.add("error");

    } finally {

        btn.disabled = false;
        btn.innerHTML = '<svg class="icon"><use href="#icon-satellite"></use></svg> Fetch from ERA5 <small>real reanalysis</small>';

    }

}


/* =========================================================
   API HELPER
   ========================================================= */

async function apiRequest(endpoint, options = {}) {

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    let response;

    try {

        response = await fetch(
            `${API_BASE}${endpoint}`,
            {
                headers: {
                    "Content-Type": "application/json"
                },
                signal: controller.signal,
                ...options
            }
        );

    } catch (error) {

        if (error.name === "AbortError") {
            throw new Error("The server took too long to answer.");
        }
        throw new Error(
            "Cannot reach the backend. Start it with: python backend/server.py"
        );

    } finally {

        clearTimeout(timer);

    }

    let payload = null;

    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }

    if (!response.ok) {
        throw new Error(
            (payload && payload.error) ||
            `API error: ${response.status}`
        );
    }

    return payload;

}


/* =========================================================
   CHECK HEALTH
   ========================================================= */

async function checkHealth() {

    const status = el("aiStatus");
    const dot = el("statusDot");

    try {

        const data = await apiRequest("/health");

        el("gradcamOption").classList.toggle("hidden", !data.gradcam_available);

        const era5Badge = el("era5SourceBadge");
        if (era5Badge) {
            if (data.era5_available) {
                era5Badge.textContent = "ACTIVE";
                era5Badge.classList.remove("plan");
            } else {
                era5Badge.textContent = "OPTIONAL";
                era5Badge.classList.add("plan");
            }
        }

        if (data.model_loaded) {

            status.textContent = "ONLINE · model loaded";
            status.style.color = "#36d399";
            el("sysTitle").textContent = "System Online";
            el("sysSub").textContent = "AI model ready";
            dot.className = "status-dot";

        } else {

            status.textContent = "DEMO MODE · no model file";
            status.style.color = "#fbbf24";
            el("sysTitle").textContent = "Demo Mode";
            el("sysSub").textContent = "Placeholder predictions";
            dot.className = "status-dot warn";

        }

    } catch (error) {

        status.textContent = "OFFLINE";
        status.style.color = "#fb7185";
        el("sysTitle").textContent = "Backend Offline";
        el("sysSub").textContent = "Start backend/server.py";
        dot.className = "status-dot bad";

        console.error(error);

    }

}


/* =========================================================
   SAMPLE IMAGES + IMAGE SELECTION
   ========================================================= */

async function loadSamples() {

    try {

        samples = await apiRequest("/samples");

        const select = el("sampleSelect");

        samples.forEach((sample, index) => {

            const option = document.createElement("option");
            option.value = sample.filename;
            option.textContent =
                `Sample ${index + 1} · storm ${sample.storm_id}`;
            select.appendChild(option);

        });

    } catch (error) {

        console.error(error);

    }

}

function showPreview(src, name, info) {

    el("preview").src = src;
    el("previewName").textContent = name;
    el("previewInfo").textContent = info || "";
    el("previewBox").classList.remove("hidden");

}

function clearImage() {

    selectedImage = null;
    el("previewBox").classList.add("hidden");
    el("preview").removeAttribute("src");
    el("sampleSelect").value = "";
    el("imageInput").value = "";

}

function chooseSample(filename) {

    if (!filename) {
        clearImage();
        return;
    }

    const sample = samples.find(s => s.filename === filename);

    if (!sample) {
        return;
    }

    selectedImage = {
        kind: "sample",
        name: sample.filename,
        sample: sample.filename,
        trueClass: sample.true_class
    };

    el("imageInput").value = "";
    el("latitude").value = sample.lat;
    el("longitude").value = sample.lon;

    showPreview(
        sample.url,
        sample.filename,
        `Sample from the test set · storm ${sample.storm_id}`
    );

}

function onFileChosen(event) {

    const file = event.target.files && event.target.files[0];

    if (!file) {
        return;
    }

    if (!["image/png", "image/jpeg"].includes(file.type)) {
        showError("Please choose a PNG or JPG image.");
        event.target.value = "";
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image is larger than 10 MB. Please choose a smaller one.");
        event.target.value = "";
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {

        selectedImage = {
            kind: "upload",
            name: file.name,
            dataUrl: reader.result
        };

        el("sampleSelect").value = "";

        showPreview(
            reader.result,
            file.name,
            `${(file.size / 1024).toFixed(0)} KB · uploaded`
        );

    };

    reader.onerror = () => {
        showError("Could not read that file.");
    };

    reader.readAsDataURL(file);

}


/* =========================================================
   RUN DEMO
   ========================================================= */

async function runDemo() {

    setLoading(true);

    try {

        const data = await apiRequest("/demo");

        console.log(
            "Demo response:",
            data
        );

        /* show which sample the backend used */
        if (data.meta && data.meta.image) {
            el("sampleSelect").value = data.meta.image;
            chooseSample(data.meta.image);
            if (data.meta.position) {
                el("latitude").value = data.meta.position[0];
                el("longitude").value = data.meta.position[1];
            }
        }

        updateDashboard(data);

        el("modeStatus").textContent = "Demo";

    } catch (error) {

        showError(error.message);

        console.error(error);

    } finally {

        setLoading(false);

    }

}


/* =========================================================
   GENERATE PREDICTION
   ========================================================= */

async function generatePrediction() {

    const inputs = getInputs();

    const body = { ...inputs };

    if (selectedImage && selectedImage.kind === "sample") {
        body.sample = selectedImage.sample;
    } else if (selectedImage && selectedImage.kind === "upload") {
        body.image = selectedImage.dataUrl;
    }

    if (el("gradcamCheckbox") && el("gradcamCheckbox").checked) {
        body.gradcam = true;
    }

    el("gradcamBox").classList.add("hidden");

    setLoading(true);

    try {

        const data = await apiRequest(
            "/predict",
            {
                method: "POST",
                body: JSON.stringify(body)
            }
        );

        console.log(
            "Prediction:",
            data
        );

        if (data.rejected) {

            renderRejection(data);

            el("modeStatus").textContent = "Rejected (not a valid cyclone image)";

        } else {

            updateDashboard(data);

            el("modeStatus").textContent =
                selectedImage ? "Live image" : "Rule-based (no image)";

        }

    } catch (error) {

        showError(error.message);

        console.error(error);

    } finally {

        setLoading(false);

    }

}


/* =========================================================
   UPDATE DASHBOARD
   ========================================================= */

function updateDashboard(data) {

    console.log(
        "Dashboard data:",
        data
    );

    const prediction = data.prediction || data;
    const sources = data.sources || {};
    const meta = data.meta || {};


    /* CATEGORY */

    const category =
        prediction.category ||
        prediction.class_name ||
        prediction.cyclone_category ||
        "Unknown";

    el("category").textContent = category;

    const catSource = sources.category;

    if (catSource === "model") {
        setTag("tagCategory", "MODEL", "real");
    } else if (catSource === "placeholder") {
        setTag("tagCategory", "PLACEHOLDER", "sim");
    } else if (catSource === "rule") {
        setTag("tagCategory", "RULE", "sim");
    } else {
        setTag("tagCategory", "", "");
    }


    /* WIND: class range when it came from an image */

    const wind = prediction.wind ?? prediction.predicted_wind ??
        prediction.wind_speed ?? data.wind;

    el("wind").textContent =
        prediction.wind_display || formatNumber(wind);

    el("windUnit").textContent =
        meta.mode === "image" ? "knots (class range)" : "knots";

    setTag(
        "tagWind",
        sources.wind === "estimate" ? "EST." :
        sources.wind === "input" ? "INPUT" : "",
        sources.wind === "input" ? "real" : "plan"
    );


    /* PRESSURE */

    const pressure = prediction.pressure ?? data.pressure;

    el("pressure").textContent = formatNumber(pressure);

    setTag(
        "tagPressure",
        sources.pressure === "simulated" ? "SIM" :
        sources.pressure === "input" ? "INPUT" : "",
        sources.pressure === "input" ? "real" : "sim"
    );


    /* RISK */

    const risk =
        prediction.risk_index ??
        prediction.risk ??
        data.risk_index ??
        0;

    el("risk").textContent = formatNumber(risk);

    setTag("tagRisk", sources.risk ? "SIM" : "", "sim");


    /* CONFIDENCE (null when no AI prediction) */

    const confidence = prediction.confidence ?? data.confidence;

    el("confidence").textContent = formatNumber(confidence);

    setTag(
        "tagConf",
        confidence === null || confidence === undefined ? "" :
        (sources.confidence === "placeholder" ? "PLACEHOLDER" : "MODEL"),
        sources.confidence === "placeholder" ? "sim" : "real"
    );


    /* PROBABILITIES */

    renderProbabilities(data.probabilities, category);


    /* PROVENANCE LINE */

    renderProvenance(category, meta, sources);
    renderDataProvenance(meta, sources);


    /* OUT-OF-DISTRIBUTION INPUT GUARD */

    renderOodWarning(meta.ood);


    /* GRAD-CAM OVERLAY (only present when requested and available) */

    if (meta.gradcam_image) {
        el("gradcamImage").src = meta.gradcam_image;
        el("gradcamBox").classList.remove("hidden");
    } else {
        el("gradcamBox").classList.add("hidden");
    }


    /* RAPID INTENSIFICATION */

    const rapid =
        prediction.rapid_intensification ??
        data.rapid_intensification ??
        false;

    el("rapidText").textContent =
        rapid
            ? "Simulated outlook shows a possible wind rise of 30 kt or more within 24 h."
            : "No rapid intensification in the simulated 24 h outlook.";

    renderEnvironmentHint(prediction.environment_favorability);


    /* RISK BADGE */

    updateRiskBadge(Number(risk));


    /* TRACK */

    const track =
        data.track ||
        data.forecast_track ||
        prediction.track ||
        [];

    drawTrack(data.past_track || [], track);


    /* CHART */

    const forecast =
        data.forecast ||
        data.wind_forecast ||
        prediction.forecast ||
        [];

    if (
        Array.isArray(forecast) &&
        forecast.length
    ) {

        updateChart(
            forecast
        );

    }


    /* tell the Analytics section (analytics/analytics.js) about this result */

    document.dispatchEvent(
        new CustomEvent("vayuvega:prediction", { detail: data })
    );

}


/* Shown instead of updateDashboard() when the out-of-distribution guard
   flags the input as very unlike a TCIR IR chip (see backend/ood_guard.py
   and backend/server.py's likely_ood early-return). Deliberately does NOT
   populate category/wind/pressure/risk/confidence/track/chart with numbers
   from a class the server never actually predicted - guessing would be
   worse than refusing. */
function renderRejection(data) {

    el("category").textContent = "Not classifiable";
    setTag("tagCategory", "REJECTED", "danger");

    el("wind").textContent = "—";
    setTag("tagWind", "", "");

    el("pressure").textContent = "—";
    setTag("tagPressure", "", "");

    el("risk").textContent = "—";
    setTag("tagRisk", "", "");

    el("confidence").textContent = "—";
    setTag("tagConf", "", "");

    renderProbabilities(
        null, null,
        "No AI probabilities - this input was rejected before it reached the classifier."
    );

    const line = el("provenance");
    line.className = "provenance warn";
    line.textContent = data.reason ||
        "This input does not look like a real satellite cyclone image.";

    renderOodWarning(data.ood);
    renderDataProvenance(data.meta || {}, { category: "rejected" });

    showToast(
        data.reason || "This doesn't look like a real satellite cyclone image - "
        + "no category was guessed.",
        "error"
    );

}


/* =========================================================
   EXPORT REPORT (print / save as PDF)
   No new dependencies: uses the browser's own window.print(), styled by
   the @media print rules in style.css. #printReportHeader is invisible on
   screen and only shown in the print stylesheet; we fill it with a fresh
   timestamp + the current result summary right before printing, both from
   the button click and from a real "beforeprint" event (so Ctrl+P / the
   browser's own Print menu gets the same header, not just our button).
   ========================================================= */

function updatePrintReportHeader() {

    const header = el("printReportHeader");
    if (!header) {
        return;
    }

    const brandEl = document.querySelector(".sidebar .brand h1");
    const brand = brandEl ? brandEl.textContent.trim() : "PRISM-TC";

    const category = el("category") ? el("category").textContent.trim() : "—";
    const confidence = el("confidence") ? el("confidence").textContent.trim() : "—";
    const lat = el("latitude") ? el("latitude").value : "—";
    const lon = el("longitude") ? el("longitude").value : "—";
    const generated = new Date().toLocaleString();

    header.innerHTML =
        `<h1>${brand} — Cyclone Intensity Report</h1>` +
        `<p>Generated ${generated} &middot; Category: ${category}` +
        (confidence !== "—" && confidence !== "" ? ` (${confidence}% model confidence)` : "") +
        ` &middot; Position: ${lat}, ${lon}</p>` +
        `<div class="print-honesty">` +
        `<strong>REAL (AI model):</strong> predicted category, confidence and class probabilities ` +
        `(53.7% exact-match / 93.1% within-one-class on a storm-wise held-out test - see the Model Transparency panel in the ` +
        `live dashboard for full numbers). <strong>SIMULATED (not machine-learned):</strong> track, ` +
        `wind outlook, pressure, risk index and rapid-intensification flag (historical-analog method ` +
        `and documented meteorological rules of thumb).` +
        `</div>`;

}

function exportReport() {
    updatePrintReportHeader();
    window.print();
}


/* Sidebar version/build footer - see APP_VERSION/APP_BUILD_DATE above. */
function renderVersionFooter() {

    const footer = el("versionFooter");
    if (!footer) {
        return;
    }

    const parsed = new Date(`${APP_BUILD_DATE}T00:00:00`);
    const formatted = isNaN(parsed.getTime())
        ? APP_BUILD_DATE
        : parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

    footer.textContent = `SIH26070 • VAYUVEGA · v${APP_VERSION} · Updated ${formatted}`;
    footer.title = `Build ${APP_BUILD_DATE}`;

}


function renderProvenance(category, meta, sources) {

    const line = el("provenance");
    line.className = "provenance";
    const parts = [];

    if (sources.category === "model") {

        parts.push(
            `AI prediction from ${meta.image || "the image"}.`
        );

        if (selectedImage && selectedImage.trueClass) {
            const match = selectedImage.trueClass === category;
            parts.push(
                `Dataset label: ${selectedImage.trueClass} — ` +
                (match ? "match." : "not a match.")
            );
        }

    } else if (sources.category === "placeholder") {

        parts.push(
            "DEMO MODE: model file not found, so this result is a placeholder."
        );
        line.classList.add("warn");

    } else if (sources.category === "rule") {

        parts.push(
            "Category derived from the entered wind speed (IMD rule), not from the AI model."
        );

    }

    if (meta.analog_storm) {
        parts.push(
            `Simulated outlook copies storm ${meta.analog_storm}.`
        );
    }

    /* only extra warnings (e.g. low confidence); the two cases above already explain themselves */
    (meta.warnings || []).forEach(w => {
        if (!w.startsWith("No image provided") && !w.startsWith("DEMO MODE") &&
            !w.startsWith("Input image") && !w.startsWith("Rule-based environment check") &&
            !w.startsWith("850hPa relative vorticity")) {
            parts.push(w);
        }
    });

    line.textContent = parts.join(" ");

}


/* Doc item 6 ("Data provenance"): itemized fields, separate from the prose
   provenance sentence above, so a judge can see at a glance exactly where
   every number came from - image source, storm, position, model, and
   whether the category came from the real model, the DEMO MODE placeholder,
   a rule, or was rejected outright. */
function renderDataProvenance(meta, sources) {

    const box = el("dataProvenance");
    if (!box) return;

    const predictionSourceLabels = {
        model: "MODEL (real AI prediction)",
        placeholder: "PLACEHOLDER (DEMO MODE - model.pth not loaded)",
        rule: "RULE-BASED (from typed wind speed, not AI)",
        rejected: "REJECTED (OOD guard - classifier never ran)",
    };

    el("dpImageSource").textContent =
        meta.image_source
            ? (meta.image_source === "built-in TCIR sample"
                ? `Built-in TCIR sample (${meta.image || "—"})`
                : "User upload")
            : "n/a (rule-based, no image)";

    const stormRow = el("dpStormRow");
    if (meta.source_storm_id) {
        el("dpStorm").textContent = meta.source_storm_id;
        stormRow.classList.remove("hidden");
    } else {
        stormRow.classList.add("hidden");
    }

    const pos = meta.position;
    el("dpPosition").textContent =
        Array.isArray(pos) ? `${pos[0]}, ${pos[1]}` : "—";

    el("dpModel").textContent =
        meta.model_name
            ? `${meta.model_name}${meta.model_loaded ? "" : " (not loaded - DEMO MODE)"}`
            : "n/a (rule-based)";

    el("dpPredSource").textContent =
        predictionSourceLabels[sources && sources.category] || sources.category || "—";

    box.classList.remove("hidden");

}


/* =========================================================
   JUDGE DEMO MODE (doc item 8) - 3 prepared, guaranteed-to-work scenarios
   for a 60-90s demo that never depends on internet or finding a good
   example image at the venue. Deliberately independent of whatever the
   user currently has selected in the normal upload/sample UI.
   ========================================================= */

function makeSyntheticDocumentDataUrl() {
    // Mirrors the Python fixture in backend/tests/smoke_test_features.py:
    // bright, near-white, low-color, moderate-contrast - looks like a
    // scanned document/bill, not a TCIR infrared chip.
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 500;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgb(250,250,250)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgb(60,60,65)";
    for (let y = 0; y < canvas.height; y += 25) {
        ctx.fillRect(30, y, 340, 2);
    }
    return canvas.toDataURL("image/png");
}

function renderJudgeDemoResult(fields) {
    const box = el("judgeDemoResult");
    if (!box) return;
    box.classList.remove("hidden");
    el("judgeDemoResultTitle").textContent = fields.title || "Result";
    el("jdAccepted").textContent = fields.accepted;
    el("jdClassifierCalled").textContent = fields.classifierCalled;
    el("jdPredictedClass").textContent = fields.predictedClass;
    el("jdConfidence").textContent = fields.confidence;
    el("jdOodResult").textContent = fields.oodResult;
    el("jdReason").textContent = fields.reason;
    el("jdExpected").textContent = fields.expected;
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runJudgeScenario1() {
    // Scenario 1: a real bundled TCIR sample -> should classify normally.
    if (!samples.length) {
        await loadSamples();
    }
    if (!samples.length) {
        showToast("No bundled samples available.", "error");
        return;
    }
    const sample = samples[0];
    try {
        const data = await apiRequest("/predict", {
            method: "POST",
            body: JSON.stringify({
                sample: sample.filename, latitude: sample.lat, longitude: sample.lon
            })
        });
        updateDashboard(data);
        el("modeStatus").textContent = "Judge demo: Scenario 1";
        const ood = data.meta && data.meta.ood;
        renderJudgeDemoResult({
            title: "Scenario 1 result - valid TCIR sample",
            accepted: "ACCEPTED",
            classifierCalled: "Yes",
            predictedClass: data.prediction.category,
            confidence: data.prediction.confidence != null ? `${data.prediction.confidence}%` : "n/a",
            oodResult: (ood && ood.level) || "none",
            reason: "Image matches the reference brightness/contrast/color profile measured " +
                    "from this project's own TCIR samples.",
            expected: "✓ Yes - a real sample should classify normally.",
        });
    } catch (error) {
        showToast(error.message, "error");
    }
}

async function runJudgeScenario2() {
    // Scenario 2: a synthetic, clearly non-cyclone "document" image,
    // generated on the fly (no file needed) -> should be rejected outright.
    const dataUrl = makeSyntheticDocumentDataUrl();
    try {
        const data = await apiRequest("/predict", {
            method: "POST",
            body: JSON.stringify({ image: dataUrl, latitude: 15.5, longitude: 85.0 })
        });
        el("modeStatus").textContent = "Judge demo: Scenario 2";
        if (data.rejected) {
            renderRejection(data);
        } else {
            updateDashboard(data);
        }
        const oodLevel = (data.ood && data.ood.level) ||
            (data.meta && data.meta.ood && data.meta.ood.level) || "—";
        renderJudgeDemoResult({
            title: "Scenario 2 result - invalid document",
            accepted: data.rejected ? "REJECTED" : "ACCEPTED (unexpected)",
            classifierCalled: data.rejected ? "No" : "Yes",
            predictedClass: data.rejected ? "none shown" : (data.prediction && data.prediction.category) || "—",
            confidence: data.rejected ? "n/a" : `${data.prediction && data.prediction.confidence}%`,
            oodResult: oodLevel,
            reason: data.reason || "—",
            expected: data.rejected
                ? "✓ Yes - a non-cyclone document should be rejected outright."
                : "✗ No - this should have been rejected. Please report this.",
        });
    } catch (error) {
        showToast(error.message, "error");
    }
}

function runJudgeScenario3() {
    // Scenario 3: live satellite - display only, surfaces whatever the
    // existing card's honest status currently is. Never faked for the demo.
    const section = el("liveSatellite");
    if (section) {
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (typeof refreshSatellite === "function") {
        refreshSatellite();
    }
    const badge = el("satBadge");
    const status = badge ? badge.textContent.trim() : "UNKNOWN";
    el("modeStatus").textContent = "Judge demo: Scenario 3";
    renderJudgeDemoResult({
        title: "Scenario 3 result - live satellite",
        accepted: "N/A (display only, not a classification)",
        classifierCalled: "No - live imagery is never fed to the classifier by design",
        predictedClass: "—",
        confidence: "—",
        oodResult: "—",
        reason: `Satellite card currently reports: ${status}. This is the real, honestly-` +
                "reported status - not simulated for the demo.",
        expected: "✓ Yes - shows whatever the real feed currently reports (LIVE or ERROR).",
    });
}


/* Out-of-distribution input guard - see backend/ood_guard.py. A heuristic
   check, not a learned model: shows why an input looked unusual so the user
   doesn't take a confident-looking classification at face value. */
function renderOodWarning(ood) {

    const box = el("oodWarning");

    if (!ood) {
        // No image was provided at all (rule-based mode) - nothing to say.
        box.classList.add("hidden");
        return;
    }

    const stats = ood.stats || {};
    const statLine = `(brightness ${stats.mean_brightness ?? "—"}, contrast ${stats.brightness_std ?? "—"}, colorfulness ${stats.colorfulness ?? "—"})`;

    box.classList.remove("hidden");
    box.classList.toggle("ood-strong", ood.level === "likely_ood");
    box.classList.toggle("ood-ok", ood.level === "none");

    const icon = el("oodWarningIcon");
    if (icon) {
        icon.querySelector("use").setAttribute(
            "href", ood.level === "none" ? "#icon-check-circle" : "#icon-alert");
    }

    if (ood.level === "none") {
        // Doc item 1: explicit positive confirmation, not just silence, so a
        // clean pass is as visible as a rejection - the "Image Compatibility"
        // gate should show its YES branch, not only its NO branch.
        el("oodWarningTitle").textContent = "Input validation: TCIR-compatible";
        el("oodWarningTag").textContent = "✓ PASSED";
        el("oodWarningTag").className = "tag real";
        el("oodWarningText").textContent =
            "This image's brightness, contrast and color profile fall within the range " +
            "measured from this project's own TCIR sample images. " + statLine;
        return;
    }

    el("oodWarningTitle").textContent = "Input looks unusual";
    el("oodWarningTag").textContent = "RULE-BASED CHECK";
    el("oodWarningTag").className = "tag sim";
    el("oodWarningText").textContent =
        (ood.level === "likely_ood"
            ? "This image looks unlike the TCIR infrared chips the model was trained on. "
            : "This image is a bit outside the usual range for a TCIR infrared chip. ") +
        (ood.reasons || []).join(" ") + " " + statLine;

}


/* SST / wind-shear rule-based environment check - see
   logic.environment_favors_intensification(). Independent of the analog-based
   rapid_intensification flag above; shown as its own line so the two are
   never conflated. */
function renderEnvironmentHint(verdict) {

    const el2 = el("rapidEnvText");

    if (!verdict) {
        el2.classList.add("hidden");
        el2.textContent = "";
        return;
    }

    const sst = el("sst") ? el("sst").value : "—";
    const shear = el("shear") ? el("shear").value : "—";
    const humidity = el("humidity") ? el("humidity").value : null;
    const vorticity = el("vorticity") ? el("vorticity").value : null;

    const labels = {
        favorable: "look favorable for intensification (warm SST, low wind shear)",
        unfavorable: "look unfavorable for intensification (cool SST and/or high wind shear)",
        neutral: "are mixed/neutral for intensification (or SST/shear look favorable but " +
                 "700hPa humidity is too dry, which inhibits intensification)",
    };

    const humidityPart = humidity != null ? `, humidity ${humidity}%` : "";
    const vorticityPart = vorticity != null
        ? ` 850hPa vorticity ${Number(vorticity).toExponential(2)} s⁻¹ is shown for context only and does not affect this verdict.`
        : "";

    el2.textContent =
        `Environment check (rule-based, not the AI model): SST ${sst}°C, wind shear ${shear} kt` +
        `${humidityPart} → conditions ${labels[verdict] || verdict}.${vorticityPart}`;
    el2.classList.remove("hidden");

}


function renderProbabilities(probs, predicted, emptyMessage) {

    const box = el("probBars");
    box.innerHTML = "";

    if (!probs) {
        const p = document.createElement("p");
        p.className = "hint";
        p.textContent = emptyMessage || "No AI probabilities for a rule-based result.";
        box.appendChild(p);
        return;
    }

    CLASS_ORDER.forEach((name, i) => {

        const value = Number(probs[name] ?? 0);

        const row = document.createElement("div");
        row.className = "prob-row" + (name === predicted ? " top" : "");

        const label = document.createElement("span");
        label.className = "prob-name";
        label.textContent = name;

        const track = document.createElement("div");
        track.className = "prob-track";

        const fill = document.createElement("div");
        fill.className = "prob-fill";
        fill.style.width = Math.max(0, Math.min(100, value)) + "%";
        fill.style.background = CLASS_COLORS[i];
        track.appendChild(fill);

        const num = document.createElement("b");
        num.textContent = value.toFixed(1) + "%";

        row.append(label, track, num);
        box.appendChild(row);

    });

}


/* =========================================================
   RISK BADGE
   ========================================================= */

function updateRiskBadge(risk) {

    const badge = el("riskBadge");

    badge.className = "badge";

    if (risk >= 75) {

        badge.classList.add("danger");
        badge.textContent = "HIGH RISK";

    } else if (risk >= 45) {

        badge.classList.add("warning");
        badge.textContent = "MODERATE RISK";

    } else {

        badge.classList.add("safe");
        badge.textContent = "LOW RISK";

    }

}


/* =========================================================
   DRAW CYCLONE TRACK
   ========================================================= */

function toCoordinates(track) {

    return track
        .map(point => {

            if (Array.isArray(point)) {
                return [Number(point[0]), Number(point[1])];
            }

            return [
                Number(point.latitude ?? point.lat),
                Number(point.longitude ?? point.lon ?? point.lng)
            ];

        })
        .filter(
            point =>
                Number.isFinite(point[0]) &&
                Number.isFinite(point[1])
        );

}

function drawTrack(pastTrack, futureTrack) {

    const past = toCoordinates(Array.isArray(pastTrack) ? pastTrack : []);
    const future = toCoordinates(Array.isArray(futureTrack) ? futureTrack : []);

    if (!past.length && !future.length) {
        return;
    }

    if (!hasLeaflet) {
        drawTrackFallback(past, future);
        return;
    }

    trackLayers.forEach(layer => map.removeLayer(layer));
    trackLayers = [];

    const bounds = [];

    if (past.length > 1) {
        trackLayers.push(
            L.polyline(past, { color: "#19b5fe", weight: 4 })
                .addTo(map)
                .bindTooltip("Observed track (IBTrACS)")
        );
    }

    if (future.length > 1) {
        trackLayers.push(
            L.polyline(
                future,
                { color: "#ff9f43", weight: 4, dashArray: "8 8" }
            )
            .addTo(map)
            .bindTooltip("Simulated 24 h outlook")
        );
    }

    const now = future[0] || past[past.length - 1];

    trackLayers.push(
        L.circleMarker(now, {
            radius: 9, color: "#ffffff", weight: 2,
            fillColor: "#fb7185", fillOpacity: 1
        })
        .addTo(map)
        .bindPopup("<strong>Current position</strong>")
        .openPopup()
    );

    if (future.length > 1) {
        trackLayers.push(
            L.circleMarker(future[future.length - 1], {
                radius: 6, color: "#ffffff", weight: 2,
                fillColor: "#ff9f43", fillOpacity: 1
            })
            .addTo(map)
            .bindTooltip("Simulated position in 24 h")
        );
    }

    past.concat(future).forEach(p => bounds.push(p));

    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 7 });

}

/* Simple SVG track plot used when Leaflet cannot load */
function drawTrackFallback(past, future) {

    const all = past.concat(future);
    const lats = all.map(p => p[0]);
    const lons = all.map(p => p[1]);
    const minLat = Math.min(...lats) - 1;
    const maxLat = Math.max(...lats) + 1;
    const minLon = Math.min(...lons) - 1;
    const maxLon = Math.max(...lons) + 1;

    const W = 700, H = 460, pad = 30;
    const sx = v => pad + (v - minLon) / (maxLon - minLon || 1) * (W - 2 * pad);
    const sy = v => H - pad - (v - minLat) / (maxLat - minLat || 1) * (H - 2 * pad);
    const pts = list => list.map(p => `${sx(p[1]).toFixed(1)},${sy(p[0]).toFixed(1)}`).join(" ");

    const now = future[0] || past[past.length - 1];
    const end = future[future.length - 1];

    el("map").innerHTML =
        `<svg viewBox="0 0 ${W} ${H}" class="track-svg" role="img" aria-label="Cyclone track">` +
        `<text x="${pad}" y="20" class="svg-note">Simple track plot (map tiles unavailable)</text>` +
        (past.length > 1 ? `<polyline points="${pts(past)}" fill="none" stroke="#19b5fe" stroke-width="4"/>` : "") +
        (future.length > 1 ? `<polyline points="${pts(future)}" fill="none" stroke="#ff9f43" stroke-width="4" stroke-dasharray="8 8"/>` : "") +
        `<circle cx="${sx(now[1])}" cy="${sy(now[0])}" r="9" fill="#fb7185" stroke="#fff" stroke-width="2"/>` +
        (future.length > 1 ? `<circle cx="${sx(end[1])}" cy="${sy(end[0])}" r="6" fill="#ff9f43" stroke="#fff" stroke-width="2"/>` : "") +
        `<text x="${W - pad}" y="${H - 8}" text-anchor="end" class="svg-note">lat ${minLat.toFixed(0)}–${maxLat.toFixed(0)}, lon ${minLon.toFixed(0)}–${maxLon.toFixed(0)}</text>` +
        `</svg>`;

}


/* =========================================================
   WIND CHART
   ========================================================= */

function updateChart(forecast) {

    const labels = [];
    const values = [];

    forecast.forEach(
        (item, index) => {

            if (typeof item === "number") {
                labels.push(`T+${index * 6}h`);
                values.push(item);
                return;
            }

            labels.push(
                item.time ||
                item.hour ||
                `T+${index * 6}h`
            );

            values.push(
                Number(
                    item.wind ??
                    item.predicted_wind ??
                    item.value ??
                    0
                )
            );

        }
    );

    renderChart(labels, values);

}

function renderChart(labels, values) {

    const canvas = el("windChart");

    if (typeof Chart === "undefined") {
        drawChartFallback(canvas, labels, values);
        return;
    }

    const ctx = canvas.getContext("2d");

    if (windChart) {
        windChart.destroy();
    }

    windChart = new Chart(
        ctx,
        {

            type: "line",

            data: {

                labels,

                datasets: [

                    {

                        label: "Wind (knots) - simulated",

                        data: values,

                        borderColor: "#19b5fe",

                        backgroundColor: "rgba(25,181,254,0.12)",

                        borderWidth: 3,

                        tension: 0.35,

                        pointRadius: 4,

                        fill: true

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        labels: { color: "#8fa5bf" }
                    }

                },

                scales: {

                    x: {
                        ticks: { color: "#8fa5bf" },
                        grid: { color: "rgba(255,255,255,0.05)" }
                    },

                    y: {
                        ticks: { color: "#8fa5bf" },
                        grid: { color: "rgba(255,255,255,0.05)" }
                    }

                }

            }

        }
    );

}

/* Plain-canvas line chart used when Chart.js cannot load */
function drawChartFallback(canvas, labels, values) {

    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    const h = canvas.clientHeight || 260;

    canvas.width = w * ratio;
    canvas.height = h * ratio;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const padL = 44, padR = 14, padT = 16, padB = 30;
    const lo = Math.floor(Math.min(...values) / 10) * 10 - 10;
    const hi = Math.ceil(Math.max(...values) / 10) * 10 + 10;
    const x = i => padL + i / Math.max(values.length - 1, 1) * (w - padL - padR);
    const y = v => padT + (1 - (v - lo) / (hi - lo || 1)) * (h - padT - padB);

    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#8fa5bf";
    ctx.strokeStyle = "rgba(255,255,255,0.06)";

    for (let g = 0; g <= 4; g++) {
        const v = lo + (hi - lo) * g / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y(v));
        ctx.lineTo(w - padR, y(v));
        ctx.stroke();
        ctx.fillText(v.toFixed(0), 8, y(v) + 4);
    }

    labels.forEach((label, i) => {
        if (i % 2 === 0) {
            ctx.fillText(label, x(i) - 12, h - 10);
        }
    });

    ctx.beginPath();
    values.forEach((v, i) => {
        if (i === 0) { ctx.moveTo(x(i), y(v)); } else { ctx.lineTo(x(i), y(v)); }
    });
    ctx.strokeStyle = "#19b5fe";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = "#19b5fe";
    values.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(x(i), y(v), 4, 0, Math.PI * 2);
        ctx.fill();
    });

}


/* =========================================================
   LOADING STATE
   ========================================================= */

function setLoading(loading) {

    const predictButton = el("predictBtn");
    const demoButton = el("demoBtn");

    predictButton.disabled = loading;
    demoButton.disabled = loading;

    predictButton.innerHTML =
        loading
            ? '<svg class="icon icon-spin"><use href="#icon-gear"></use></svg> Processing...'
            : '<svg class="icon"><use href="#icon-sparkle"></use></svg> Generate AI Prediction';

}


/* =========================================================
   LIVE / NEAR-REAL-TIME SATELLITE

   Polls the backend's /api/satellite/* routes (see backend/satellite/).
   Everything here is defensive: if the satellite feature is unavailable
   (503), disabled, or the fetch itself times out, this panel shows a
   clear status instead of breaking the rest of the page. It never talks
   to a satellite provider directly - only to our own backend.
   ========================================================= */

const SAT_POLL_MS = 60000;         /* refresh the panel every 60s */
const SAT_BADGE_KIND = {
    LIVE: "safe",
    STALE: "warning",
    UPDATING: "neutral",
    ERROR: "danger",
    NO_DATA: "neutral"
};
let satPollTimer = null;

function satFormatAge(isoTime) {
    if (!isoTime) return "—";
    const then = new Date(isoTime).getTime();
    if (Number.isNaN(then)) return "—";
    const minutes = Math.max(0, Math.round((Date.now() - then) / 60000));
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours} h ${minutes % 60} min ago`;
}

function satRenderState(state) {

    const status = state.status || "NO_DATA";
    const badge = el("satBadge");
    badge.textContent = status.replace("_", " ");
    badge.className = "badge " + (SAT_BADGE_KIND[status] || "neutral");

    el("satSource").textContent = state.source || "—";
    el("satSatellite").textContent = state.satellite || "—";
    el("satProduct").textContent = state.product || "—";
    el("satObsTime").textContent = state.observation_time_utc || "—";
    el("satAge").textContent = satFormatAge(state.observation_time_utc);
    el("satLastCheck").textContent = satFormatAge(state.last_check_utc) === "—"
        ? "—" : `${satFormatAge(state.last_check_utc)} (${state.last_check_utc || ""})`;
    el("satLastSuccess").textContent = state.last_success_utc
        ? satFormatAge(state.last_success_utc) : "—";

    if (state.region && state.region.coverage_fraction != null) {
        const pct = Math.round(state.region.coverage_fraction * 100);
        el("satCoverage").textContent = state.region.fully_visible
            ? "Full requested region visible"
            : `~${pct}% of the North Indian Ocean box visible from this satellite`;
    } else if (state.region && state.region.description) {
        el("satCoverage").textContent = state.region.description;
    } else {
        el("satCoverage").textContent = "—";
    }

    const img = el("satImage");
    const placeholder = el("satPlaceholder");
    if (status === "LIVE" || status === "STALE") {
        img.src = `${API_BASE}/satellite/image?ts=${Date.now()}`;
        img.classList.remove("hidden");
        placeholder.classList.add("hidden");
    } else {
        img.classList.add("hidden");
        placeholder.classList.remove("hidden");
        placeholder.textContent = status === "UPDATING"
            ? "Checking for a new satellite image..."
            : "No satellite image captured yet.";
    }

    const errorBox = el("satErrorBox");
    if (state.error) {
        errorBox.textContent = `${state.error_stage ? `[${state.error_stage}] ` : ""}${state.error}`;
        errorBox.classList.remove("hidden");
    } else {
        errorBox.classList.add("hidden");
    }
}

async function refreshSatellite() {
    try {
        const state = await apiRequest("/satellite/status");
        satRenderState(state);
    } catch (error) {
        // The satellite feature failing must never look like the whole
        // app is broken - show a scoped message in this panel only.
        el("satBadge").textContent = "UNAVAILABLE";
        el("satBadge").className = "badge neutral";
        el("satErrorBox").textContent =
            "Could not reach the satellite status endpoint: " + error.message;
        el("satErrorBox").classList.remove("hidden");
    }
}

function startSatellitePolling() {
    refreshSatellite();
    if (satPollTimer) clearInterval(satPollTimer);
    satPollTimer = setInterval(refreshSatellite, SAT_POLL_MS);
}


/* =========================================================
   MODEL TRANSPARENCY PANEL

   Fetches the static model card from GET /api/model/transparency once at
   load. See backend/model_card.py - every number here is copied from
   CLAUDE.md, nothing is computed live.
   ========================================================= */

async function loadModelTransparency() {
    try {
        const card = await apiRequest("/model/transparency");
        const m = card.metrics || {};

        el("mtAccuracy").textContent = m.accuracy_pct != null ? `${m.accuracy_pct}%` : "—";
        el("mtMacroF1").textContent = m.macro_f1 != null ? m.macro_f1 : "—";
        el("mtAvgError").textContent = m.average_error_severity_levels != null
            ? `${m.average_error_severity_levels} levels` : "—";
        el("mtEce").textContent = m.expected_calibration_error_pct != null
            ? `~${m.expected_calibration_error_pct}%` : "—";

        const cal = card.calibration || {};
        el("mtCalibration").textContent = cal.fitted
            ? `Calibrated (T=${cal.temperature})`
            : "Not applicable - no softmax logits to calibrate (see Known failure modes)";

        const notes = (card.test_set_class_notes || [])
            .map(n => `${n.class}: ${n.share_pct != null ? n.share_pct + "% of test set" : n.count + " test images"} - ${n.note}`);
        el("mtTestSetNote").textContent =
            `Test set: ${m.test_set_size || "?"} held-out images. ` + notes.join(" ");

        const failuresBox = el("mtFailureModes");
        failuresBox.innerHTML = "";
        (card.known_failure_modes || []).forEach(text => {
            const li = document.createElement("li");
            li.textContent = text;
            failuresBox.appendChild(li);
        });

        const honestyBox = el("mtHonestyNotes");
        honestyBox.innerHTML = "";
        (card.honesty_notes || []).forEach(text => {
            const li = document.createElement("li");
            li.textContent = text;
            honestyBox.appendChild(li);
        });

    } catch (error) {
        el("mtTestSetNote").textContent =
            "Could not load the model transparency data: " + error.message;
        console.error(error);
    }
}


/* =========================================================
   BACKTEST MODE (analog track/wind outlook)

   See backend/backtest.py. This tests logic.TrackDB.analog_outlook() against
   real historical storms already in backend/data/track_data.csv - it does
   NOT test the AI image classifier.
   ========================================================= */

async function loadBacktestStorms() {
    try {
        const storms = await apiRequest("/backtest/storms");
        const select = el("backtestSelect");
        select.innerHTML = "<option value=\"\">choose a historical storm...</option>";
        storms.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s.storm_id;
            opt.textContent = `${s.storm_id} - ${s.category} (${s.start_time})`;
            select.appendChild(opt);
        });
    } catch (error) {
        console.error("Could not load backtestable storms:", error);
    }
}

async function runBacktest() {
    const storm_id = el("backtestSelect").value;
    if (!storm_id) {
        showError("Choose a storm to backtest first.");
        return;
    }

    const btn = el("backtestBtn");
    btn.disabled = true;
    btn.textContent = "Running...";

    try {
        const result = await apiRequest("/backtest/run", {
            method: "POST",
            body: JSON.stringify({ storm_id })
        });

        el("btStorm").textContent = `${result.storm_id} (${result.true_category})`;
        el("btAnalog").textContent = result.analog_storm_used || "—";
        el("btTrackErr").textContent = result.track_error_km && result.track_error_km.mean != null
            ? `${result.track_error_km.mean} km` : "—";
        el("btWindErr").textContent = result.wind_error_kt && result.wind_error_kt.mean != null
            ? `${result.wind_error_kt.mean} kt` : "—";

        const body = el("backtestTableBody");
        body.innerHTML = "";
        const steps = result.predicted_forecast || [];
        steps.forEach((step, i) => {
            const actualWind = result.actual_forecast && result.actual_forecast[i]
                ? result.actual_forecast[i].wind : "—";
            const trackErr = result.track_error_km && result.track_error_km.by_step
                ? result.track_error_km.by_step[i] : "—";
            const row = document.createElement("tr");
            row.innerHTML =
                `<td>${step.time}</td><td>${step.wind}</td><td>${actualWind}</td><td>${trackErr}</td>`;
            body.appendChild(row);
        });

        el("backtestResults").classList.remove("hidden");
        el("backtestPlaceholder").classList.add("hidden");

    } catch (error) {
        showError(error.message);
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run backtest";
    }
}


/* =========================================================
   SIDEBAR SCROLL-SPY

   Highlights the sidebar nav item for whichever major section is currently
   in view, so the sidebar stays a useful "you are here" map on a long page
   instead of only ever showing "Dashboard" as active. Only spies on the
   real top-level <section> landmarks (not the smaller Forecast/Input jump
   targets nested inside the dashboard grid) so it stays simple and robust.
   ========================================================= */

function setupScrollSpy() {

    const spySectionIds = [
        "dashboard", "analytics", "liveSatellite",
        "modelTransparency", "backtest", "references", "apiReference", "system"
    ];

    const navLinks = Array.from(document.querySelectorAll(".sidebar nav .nav-item"));
    const sections = spySectionIds.map(id => el(id)).filter(Boolean);

    if (!sections.length) {
        return;
    }

    const setActive = (id) => {
        navLinks.forEach(link => {
            link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
        });
    };

    /* Classic scroll-spy: the "current" section is the last one whose top
       has scrolled up to (or past) a fixed line near the top of the
       viewport. Simpler and more predictable across very different section
       heights than trying to compare IntersectionObserver ratios. */
    const TOP_LINE_PX = 140;

    const updateActive = () => {

        /* Near the very bottom of a short page, the last section(s) may
           never be able to scroll their top up to TOP_LINE_PX (there simply
           isn't enough page left below them) - so once scrolled to the
           bottom, just activate the last section directly. */
        const atBottom = window.innerHeight + window.scrollY
            >= document.documentElement.scrollHeight - 4;
        if (atBottom) {
            setActive(sections[sections.length - 1].id);
            return;
        }

        let current = sections[0];
        for (const section of sections) {
            if (section.getBoundingClientRect().top <= TOP_LINE_PX) {
                current = section;
            }
        }
        setActive(current.id);
    };

    let ticking = false;
    window.addEventListener("scroll", () => {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(() => {
            updateActive();
            ticking = false;
        });
    });

    updateActive();

}


/* =========================================================
   EVENT LISTENERS
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeMap();

        checkHealth();
        loadSamples();
        startSatellitePolling();
        loadModelTransparency();
        loadBacktestStorms();
        setupScrollSpy();
        renderVersionFooter();

        el("demoBtn").addEventListener("click", runDemo);
        el("predictBtn").addEventListener("click", generatePrediction);
        el("healthBtn").addEventListener("click", checkHealth);
        el("satRefreshBtn").addEventListener("click", refreshSatellite);
        el("imageInput").addEventListener("change", onFileChosen);
        el("clearImage").addEventListener("click", clearImage);
        el("backtestBtn").addEventListener("click", runBacktest);
        el("era5FetchBtn").addEventListener("click", fetchFromEra5);
        el("exportReportBtn").addEventListener("click", exportReport);
        el("judgeDemo1").addEventListener("click", runJudgeScenario1);
        el("judgeDemo2").addEventListener("click", runJudgeScenario2);
        el("judgeDemo3").addEventListener("click", runJudgeScenario3);
        window.addEventListener("beforeprint", updatePrintReportHeader);
        ["sst", "shear", "humidity", "vorticity"].forEach(id => {
            const input = el(id);
            if (input) {
                input.addEventListener("input", clearEra5Badge);
            }
        });
        el("sampleSelect").addEventListener(
            "change",
            event => chooseSample(event.target.value)
        );

        /* INITIAL MAP MARKER */

        if (hasLeaflet) {

            const lat = Number(el("latitude").value);
            const lon = Number(el("longitude").value);

            trackLayers.push(
                L.circleMarker([lat, lon], {
                    radius: 8, color: "#ffffff", weight: 2,
                    fillColor: "#19b5fe", fillOpacity: 1
                })
                .addTo(map)
                .bindPopup("Initial cyclone position")
            );

        }

    }
);
