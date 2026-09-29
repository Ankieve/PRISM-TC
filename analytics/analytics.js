/* =========================================================
   VAYUVEGA - DATA & ANALYTICS SECTION  (Member 5)

   Draws every chart from window.VAYUVEGA_ANALYTICS, which is written by
   analytics/build_analytics.py from the project's real files.
   No CDN needed: charts are plain SVG, so the section also works offline.

   Live link with Member 4's dashboard: app.js dispatches the browser event
   "vayuvega:prediction" (detail = the /api response) after every prediction.
   ========================================================= */
(function () {
    "use strict";

    const root = document.getElementById("analytics");
    if (!root) { return; }

    const D = window.VAYUVEGA_ANALYTICS;
    if (!D) {
        root.innerHTML =
            '<div class="an-empty"><strong>Analytics data not found.</strong> ' +
            "Run <code>python analytics/build_analytics.py</code> and reload.</div>";
        return;
    }

    /* same colours as the class bars of the main dashboard */
    const COLORS = ["#4C9F70", "#E0B040", "#E8873A", "#D6493B", "#8B1E3F"];
    const SHORT = ["Depression", "Cyclonic", "Severe", "Very Severe", "Extreme"];
    const CLASSES = D.classes;
    const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const MON_LONG = MONTHS;

    /* ---------------------------------------------------------- utils -- */
    const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const num = (n, d = 0) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    const pctOrNa = v => (v === null || v === undefined) ? "n/a" : num(v, 1) + "%";
    const stormById = id => D.storms.find(s => s.id === id);
    const sampleByFile = f => D.samples.find(s => s.filename === f);

    function parseT(t) {              /* 2010102218 -> Date (UTC) */
        const s = String(t);
        return new Date(Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8), +s.slice(8, 10)));
    }
    function fmtT(t) {
        const d = parseT(t);
        return d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear() +
            ", " + String(d.getUTCHours()).padStart(2, "0") + ":00 UTC";
    }
    function niceStep(range, count) {
        const raw = range / Math.max(count, 1);
        const mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const f = raw / mag;
        return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
    }
    function niceMax(v) {
        if (v <= 0) { return 1; }
        const step = niceStep(v, 4);
        return Math.ceil(v / step) * step;
    }
    function ticks(min, max, count) {
        const step = niceStep(max - min, count);
        const out = [];
        for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) { out.push(+v.toFixed(6)); }
        return out;
    }

    function card(id, title, sub, tags, body, cls) {
        return '<div class="card an-card ' + (cls || "") + '" id="' + id + '">' +
            '<div class="card-header"><div><h3>' + title + "</h3><p>" + sub + "</p></div>" +
            '<div class="an-tags">' + (tags || "") + "</div></div>" +
            '<div class="an-body">' + body + "</div></div>";
    }
    const tag = (text, kind) => '<em class="tag ' + kind + '">' + text + "</em>";
    const T_DATA = tag("IBTrACS DATA", "real");
    const T_COMP = tag("COMPUTED", "real");
    const T_REP = tag("REPORTED", "plan");

    /* ----------------------------------------------------- SVG charts -- */
    function columns(o) {
        const W = o.W || 560, H = o.H || 250, L = 44, R = 10, T = 12, B = o.B || 30;
        const pw = W - L - R, ph = H - T - B, n = o.labels.length;
        const totals = o.labels.map((_, i) => o.series.reduce((a, s) => a + s.values[i], 0));
        const ymax = niceMax(Math.max(...totals, 1));
        const sy = v => T + (1 - v / ymax) * ph;
        const bw = pw / n;
        let g = "";
        ticks(0, ymax, 4).forEach(v => {
            g += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(v) + '" y2="' + sy(v) + '"/>' +
                '<text x="' + (L - 6) + '" y="' + (sy(v) + 3) + '" text-anchor="end">' + num(v) + "</text>";
        });
        o.labels.forEach((lab, i) => {
            let acc = 0;
            o.series.forEach((s, k) => {
                const v = s.values[i];
                if (!v) { return; }
                const color = s.colorFn ? s.colorFn(i) : s.color;
                g += '<rect x="' + (L + i * bw + bw * 0.11).toFixed(1) + '" y="' + sy(acc + v).toFixed(1) +
                    '" width="' + (bw * 0.78).toFixed(1) + '" height="' + (sy(acc) - sy(acc + v)).toFixed(1) +
                    '" fill="' + color + '"><title>' + esc(o.tipPrefix ? o.tipPrefix(i) : lab) + " - " +
                    esc(s.name) + ": " + num(v) + "</title></rect>";
                acc += v;
            });
            if (!o.xEvery || i % o.xEvery === 0) {
                g += '<text x="' + (L + i * bw + bw / 2).toFixed(1) + '" y="' + (H - B + 14) +
                    '" text-anchor="middle">' + esc(lab) + "</text>";
            }
        });
        (o.vlines || []).forEach(v => {
            const x = L + v.at * bw;
            g += '<line class="vline" x1="' + x + '" x2="' + x + '" y1="' + T + '" y2="' + (H - B) + '"/>' +
                '<text class="vlab" x="' + (x + 3) + '" y="' + (T + 9) + '">' + esc(v.label) + "</text>";
        });
        g += '<line class="axis" x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - B) + '" y2="' + (H - B) + '"/>';
        if (o.xTitle) {
            g += '<text x="' + (L + pw / 2) + '" y="' + (H - 3) + '" text-anchor="middle">' + esc(o.xTitle) + "</text>";
        }
        return '<svg class="an-svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.aria) + '">' + g + "</svg>";
    }

    function lines(o) {
        const W = o.W || 560, H = o.H || 250, L = 46, R = o.R || 14, T = 12, B = 34;
        const pw = W - L - R, ph = H - T - B;
        const sx = x => L + (x - o.xMin) / (o.xMax - o.xMin || 1) * pw;
        const sy = y => T + (1 - (y - o.yMin) / (o.yMax - o.yMin || 1)) * ph;
        let g = "";
        ticks(o.yMin, o.yMax, 4).forEach(v => {
            g += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(v).toFixed(1) + '" y2="' + sy(v).toFixed(1) + '"/>' +
                '<text x="' + (L - 6) + '" y="' + (sy(v) + 3).toFixed(1) + '" text-anchor="end">' + num(v) + "</text>";
        });
        (o.xTicks || []).forEach(t => {
            g += '<text x="' + sx(t.x).toFixed(1) + '" y="' + (H - B + 14) + '" text-anchor="middle">' + esc(t.label) + "</text>" +
                '<line class="tick" x1="' + sx(t.x).toFixed(1) + '" x2="' + sx(t.x).toFixed(1) + '" y1="' + (H - B) + '" y2="' + (H - B + 4) + '"/>';
        });
        (o.hlines || []).forEach(h => {
            g += '<line class="hline" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(h.y).toFixed(1) + '" y2="' + sy(h.y).toFixed(1) +
                '" stroke="' + h.color + '"/><text class="hlab" x="' + (W - R - 3) + '" y="' + (sy(h.y) - 3).toFixed(1) +
                '" text-anchor="end">' + esc(h.label) + "</text>";
        });
        o.series.forEach(s => {
            const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" ");
            g += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="' + (s.width || 2.4) +
                '"' + (s.dash ? ' stroke-dasharray="' + s.dash + '"' : "") + ' stroke-linejoin="round"/>';
            if (s.dots) {
                s.points.forEach(p => {
                    g += '<circle cx="' + sx(p[0]).toFixed(1) + '" cy="' + sy(p[1]).toFixed(1) + '" r="3.2" fill="' + s.color +
                        '"><title>' + esc(s.name) + ": " + esc(s.tip ? s.tip(p) : p[1]) + "</title></circle>";
                });
            }
        });
        (o.markers || []).forEach(m => {
            g += '<circle cx="' + sx(m.x).toFixed(1) + '" cy="' + sy(m.y).toFixed(1) + '" r="6" fill="' + m.color +
                '" stroke="#fff" stroke-width="2"><title>' + esc(m.tip || "") + "</title></circle>";
        });
        g += '<line class="axis" x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - B) + '" y2="' + (H - B) + '"/>';
        if (o.xTitle) { g += '<text x="' + (L + pw / 2) + '" y="' + (H - 3) + '" text-anchor="middle">' + esc(o.xTitle) + "</text>"; }
        if (o.yTitle) { g += '<text transform="translate(11 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.yTitle) + "</text>"; }
        return '<svg class="an-svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.aria) + '">' + g + "</svg>";
    }

    /* matrix heat-map: cell colour = share of the ROW (rows sum to 100 %) */
    function heat(o) {
        const n = SHORT.length, cell = 62, L = 92, T = 34, RT = 52;
        const W = L + n * cell + RT, H = T + n * cell + 28;
        let g = '<text x="' + (L + n * cell / 2) + '" y="12" text-anchor="middle" class="axt">' + esc(o.colTitle) + "</text>";
        g += '<text x="' + (L + n * cell + RT / 2) + '" y="' + (T - 6) + '" text-anchor="middle">total</text>';
        for (let j = 0; j < n; j++) {
            g += '<text x="' + (L + j * cell + cell / 2) + '" y="' + (T - 6) + '" text-anchor="middle">' + SHORT[j] + "</text>";
        }
        o.matrix.forEach((row, i) => {
            const tot = row.reduce((a, b) => a + b, 0);
            g += '<text x="' + (L - 8) + '" y="' + (T + i * cell + cell / 2 + 4) + '" text-anchor="end">' + SHORT[i] + "</text>";
            row.forEach((v, j) => {
                const p = tot ? v / tot : 0;
                g += '<rect x="' + (L + j * cell + 1) + '" y="' + (T + i * cell + 1) + '" width="' + (cell - 2) + '" height="' + (cell - 2) +
                    '" rx="5" fill="rgba(25,181,254,' + (0.06 + 0.86 * p).toFixed(3) + ')"' +
                    (i === j ? ' stroke="rgba(255,255,255,.55)"' : "") + "><title>" +
                    esc(CLASSES[i] + " -> " + CLASSES[j] + ": " + num(v) + " (" + num(100 * p, 1) + "% of row)") + "</title></rect>";
                if (v) {
                    g += '<text class="cellv" x="' + (L + j * cell + cell / 2) + '" y="' + (T + i * cell + cell / 2 - 1) +
                        '" text-anchor="middle">' + num(100 * p, p >= 0.995 ? 0 : 1) + "%</text>" +
                        '<text class="cellc" x="' + (L + j * cell + cell / 2) + '" y="' + (T + i * cell + cell / 2 + 12) +
                        '" text-anchor="middle">' + num(v) + "</text>";
                }
            });
            g += '<text x="' + (L + n * cell + RT / 2) + '" y="' + (T + i * cell + cell / 2 + 4) + '" text-anchor="middle">' + num(tot) + "</text>";
        });
        g += '<text transform="translate(11 ' + (T + n * cell / 2) + ') rotate(-90)" text-anchor="middle" class="axt">' + esc(o.rowTitle) + "</text>";
        return '<svg class="an-svg heat" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.aria) + '">' + g + "</svg>";
    }

    function hbars(rows, fixedMax) {
        const max = fixedMax || Math.max(...rows.map(r => r.value), 1e-9);
        return '<div class="an-bars">' + rows.map(r =>
            '<div class="an-bar-row"><span class="an-bar-name">' + esc(r.label) + "</span>" +
            '<div class="an-bar-track"><div class="an-bar-fill" style="width:' + (100 * r.value / max).toFixed(1) +
            "%;background:" + r.color + '"></div></div><b>' + esc(r.text) + "</b></div>").join("") + "</div>";
    }

    const legend = () => '<div class="an-legend">' + CLASSES.map((c, i) =>
        '<span><i style="background:' + COLORS[i] + '"></i>' + esc(c) + "</span>").join("") + "</div>";

    /* ------------------------------------------------- static sections -- */
    const O = D.overview, CD = D.class_distribution, IC = D.intensity_change, BT = D.backtest, EV = D.evaluation;

    function header() {
        const inp = D.meta.inputs;
        return '<div class="an-head"><div><h2>&#128200; Data &amp; Analytics</h2>' +
            "<p>Every number and chart below is computed from the project's own files (see the tag on each card). " +
            "Nothing is filled in by hand.</p></div>" +
            '<div class="an-prov"><span>Built ' + esc(D.meta.generated_utc) + " UTC</span>" +
            "<span>track_data.csv #" + esc(inp["backend/data/track_data.csv"]) + "</span>" +
            "<span>test-set predictions: " + esc(inp["test_predictions.csv"]) + "</span></div></div>";
    }

    function kpis() {
        const riPct = IC.ri_rate_pct;
        const items = [
            [num(O.records), "storm observations", "every " + O.step_hours + " h, " + O.first_obs.slice(0, 4) + "-" + O.last_obs.slice(0, 4)],
            [num(O.storms), "distinct storms", "North Indian Ocean, seasons " + O.season_years[0] + "-" + O.season_years[1]],
            [O.wind_kt_range[0] + "-" + O.wind_kt_range[1], "kt max wind range", "mean " + O.wind_kt_mean + " kt"],
            [num(CD.majority_baseline_pct, 1) + "%", "are Depression", "always guessing it = " + num(CD.majority_baseline_pct, 0) + "% \"accuracy\""],
            [num(CD.imbalance_ratio_max_to_min, 1) + "x", "class imbalance", "Depression vs Extremely Severe"],
            [num(riPct, 1) + "%", "24 h windows with rapid intensification", "wind up 30 kt or more"]
        ];
        return '<div class="an-kpis">' + items.map(i =>
            '<div class="an-kpi"><strong>' + esc(i[0]) + "</strong><span>" + esc(i[1]) + "</span><small>" + esc(i[2]) + "</small></div>").join("") + "</div>";
    }

    function cardClassDistribution() {
        const rows = CD.rows.map(r => ({
            label: r.class, value: r.records, color: COLORS[r.index],
            text: num(r.records) + "  (" + num(r.pct, 1) + "%)"
        }));
        const peak = CD.rows.map(r => r.storms_reaching_as_peak);
        const body = hbars(rows) +
            '<p class="an-note">Peak class reached by each of the ' + O.storms + " storms: " +
            CD.rows.map((r, i) => "<b>" + peak[i] + "</b> " + SHORT[i]).join(" &middot; ") + ".</p>" +
            '<p class="an-note">Why it matters: the strongest classes are rare, so a model can look good on average ' +
            "while failing on exactly the storms that matter most. Use macro-F1, not accuracy alone.</p>";
        return card("anClass", "Class distribution", "Observations per IMD intensity class", T_DATA, body);
    }

    function cardWindHist() {
        const wh = D.wind_hist;
        const labels = wh.bin_edges_kt.slice(0, -1).map(e => String(e));
        const series = CLASSES.map((c, i) => ({ name: c, color: COLORS[i], values: wh.counts_by_class[c] }));
        const vlines = wh.class_start_kt.slice(1).map((kt, i) => ({ at: (kt - wh.bin_edges_kt[0]) / 10, label: SHORT[i + 1] }));
        const body = columns({
            labels, series, vlines, xEvery: 1, xTitle: "Maximum sustained wind (kt), 10-kt bins", W: 560, H: 250,
            tipPrefix: i => wh.bin_edges_kt[i] + "-" + wh.bin_edges_kt[i + 1] + " kt", aria: "Histogram of wind speed by class"
        }) + legend() +
            '<p class="an-note">Dashed lines are the IMD class boundaries (34, 48, 64 and 90 kt). Colours show the class label stored in the data.</p>';
        return card("anWind", "Wind-speed distribution", "Maximum sustained wind of all observations", T_DATA, body);
    }

    /* ----------------------------------------------------------- map -- */
    const mapState = { visible: new Set([0, 1, 2, 3, 4]), storm: null, sample: null };

    function basinSvg() {
        const bm = D.basemap, S = bm.scale;
        const W = (bm.lon1 - bm.lon0) * S, H = (bm.lat1 - bm.lat0) * S;
        const X = lon => ((lon - bm.lon0) * S).toFixed(1), Y = lat => ((bm.lat1 - lat) * S).toFixed(1);
        let g = '<rect width="' + W + '" height="' + H + '" fill="#081321"/>';
        for (let lon = 50; lon <= 100; lon += 10) {
            g += '<line class="grat" x1="' + X(lon) + '" x2="' + X(lon) + '" y1="0" y2="' + H + '"/>' +
                '<text class="gl" x="' + (+X(lon) + 3) + '" y="' + (H - 4) + '">' + lon + "&deg;E</text>";
        }
        for (let lat = 10; lat <= 30; lat += 10) {
            g += '<line class="grat" y1="' + Y(lat) + '" y2="' + Y(lat) + '" x1="0" x2="' + W + '"/>' +
                '<text class="gl" x="4" y="' + (+Y(lat) - 3) + '">' + lat + "&deg;N</text>";
        }
        g += '<path d="' + bm.path + '" fill="#1e3452" stroke="#3b5c86" stroke-width=".8"/>';
        const sx = X(D.sectors.split_lon);
        g += '<line class="split" x1="' + sx + '" x2="' + sx + '" y1="0" y2="' + H + '"/>' +
            '<text class="gl" x="' + (+sx + 4) + '" y="12">sector split 77.5&deg;E</text>';
        /* tracks */
        D.storms.forEach(s => {
            const d = s.lat.map((la, i) => (i ? "L" : "M") + X(s.lon[i]) + "," + Y(la)).join("");
            g += '<path d="' + d + '" fill="none" stroke="rgba(190,210,235,.22)" stroke-width=".9"/>';
        });
        /* observations, weakest first so strong storms stay visible */
        for (let c = 0; c < 5; c++) {
            if (!mapState.visible.has(c)) { continue; }
            let pts = "";
            D.storms.forEach(s => {
                s.c.forEach((cc, i) => {
                    if (cc === c) {
                        pts += '<circle cx="' + X(s.lon[i]) + '" cy="' + Y(s.lat[i]) + '" r="' + (1.35 + c * 0.22).toFixed(2) + '"/>';
                    }
                });
            });
            g += '<g fill="' + COLORS[c] + '" fill-opacity=".85">' + pts + "</g>";
        }
        /* selected storm */
        if (mapState.storm) {
            const s = mapState.storm;
            const d = s.lat.map((la, i) => (i ? "L" : "M") + X(s.lon[i]) + "," + Y(la)).join("");
            g += '<path d="' + d + '" fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/>';
            if (mapState.sample) {
                const i = s.t.indexOf(mapState.sample.time);
                if (i >= 0) {
                    g += '<circle cx="' + X(s.lon[i]) + '" cy="' + Y(s.lat[i]) + '" r="7" fill="' + COLORS[s.c[i]] +
                        '" stroke="#fff" stroke-width="2.5"><title>Sample image position</title></circle>';
                }
            }
            g += '<text class="sel" x="' + (W - 8) + '" y="' + (H - 8) + '" text-anchor="end">storm ' + esc(s.id) + " highlighted</text>";
        }
        return '<svg class="an-svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Map of all storm observations coloured by class">' + g + "</svg>";
    }

    function mapChips() {
        return '<div class="an-legend an-chips">' + CLASSES.map((c, i) =>
            '<button type="button" class="chip' + (mapState.visible.has(i) ? " on" : "") + '" data-cls="' + i + '">' +
            '<i style="background:' + COLORS[i] + '"></i>' + esc(c) + "</button>").join("") + "</div>";
    }

    function cardMap() {
        const S = D.sectors;
        const sec = (name, x) => "<b>" + name + "</b>: " + x.storms + " storms, " + num(x.records) + " observations, " +
            x.storms_reaching_very_severe_or_higher + " reached Very Severe or stronger";
        const body = '<div id="anMapBox">' + basinSvg() + "</div>" + mapChips() +
            '<p class="an-note">' + sec("West of 77.5&deg;E (Arabian Sea side)", S.west) + "<br>" +
            sec("Bay of Bengal side", S.east) + "</p>" +
            '<p class="an-note">' + esc(S.rule) + "</p>";
        return card("anMap", "Where the storms are", "All " + num(O.records) + " observations and " + O.storms + " tracks; click a class to show or hide it",
            T_DATA, body, "span2");
    }

    /* ---------------------------------------------------- seasonality -- */
    function cardMonthly() {
        const s = D.seasonality;
        const series = CLASSES.map((c, i) => ({ name: c, color: COLORS[i], values: s.records_by_class[c] }));
        const peakMonths = s.records_by_class[CLASSES[0]].map((_, m) =>
            CLASSES.reduce((a, c) => a + s.records_by_class[c][m], 0));
        const top = peakMonths.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(x => MON_LONG[x[1]]);
        const body = columns({
            labels: MONTHS, series, W: 560, H: 250, aria: "Observations per month by class",
            tipPrefix: i => MON_LONG[i]
        }) + legend() +
            '<p class="an-note">Two seasons stand out: pre-monsoon (Apr-Jun) and post-monsoon (Oct-Dec). ' +
            "Busiest months by observations: " + top.join(", ") + ". Storms active per month: " +
            MONTHS.map((m, i) => m + " " + s.storms_active[i]).filter((_, i) => s.storms_active[i] > 0).join(", ") + ".</p>";
        return card("anMonth", "Seasonality", "Observations per calendar month, coloured by class", T_DATA, body);
    }

    function cardYearly() {
        const y = D.seasonality.yearly;
        const body = columns({
            labels: y.map(r => "'" + String(r.year).slice(2)), W: 560, H: 250,
            series: [{ name: "storms", color: "#19b5fe", values: y.map(r => r.storms) }],
            tipPrefix: i => y[i].year + " (peak wind " + y[i].peak_wind_kt + " kt, " + y[i].records + " observations)",
            aria: "Storms per season year", xTitle: "Season year (from storm ID)"
        }) + '<p class="an-note">Only ' + O.storms + " storms over " + (O.season_years[1] - O.season_years[0] + 1) +
            " seasons: this is a small dataset, so year-to-year differences are not statistically meaningful.</p>";
        return card("anYear", "Storms per season", "Number of storms in the file for each year", T_DATA, body);
    }

    /* ------------------------------------------------ intensity dynamics */
    let transMode = "24h";
    function transHtml() {
        const t = D.transitions[transMode];
        return heat({
            matrix: t.matrix_counts, aria: "Class transition matrix for " + transMode,
            rowTitle: "class now", colTitle: "class after " + (transMode === "3h" ? "3 hours" : "24 hours")
        }) + '<p class="an-note">' + num(t.pairs) + " pairs. After " + (transMode === "3h" ? "3 h" : "24 h") + ": <b>" +
            num(t.same_class_pct, 1) + "%</b> stay in the same class, <b>" + num(t.one_class_change_pct, 1) +
            "%</b> move one class, <b>" + num(t.two_or_more_pct, 1) + "%</b> move two or more.</p>";
    }
    function cardTransitions() {
        const seg = '<div class="an-seg" id="anTransSeg"><button type="button" data-mode="3h">3 h</button>' +
            '<button type="button" data-mode="24h" class="active">24 h</button></div>';
        return card("anTrans", "How storms change class", "Where a storm's class is after 3 h or 24 h (each row sums to 100%)",
            T_DATA + seg, '<div id="anTransBody">' + transHtml() + "</div>");
    }

    function cardIntensityChange() {
        const edges = IC.bin_edges_kt, n = IC.counts.length;
        const labels = edges.slice(0, -1).map((e, i) => i === 0 ? "<-60" : i === n - 1 ? ">=70" : String(e));
        const body = columns({
            labels, W: 560, H: 220, aria: "Distribution of 24 hour wind change", B: 34,
            series: [{ name: "24 h windows", values: IC.counts, color: "#19b5fe", colorFn: i => edges[i] >= IC.ri_threshold_kt ? "#fbbf24" : "#19b5fe" }],
            tipPrefix: i => "change " + edges[i] + " to " + edges[i + 1] + " kt", xTitle: "Wind change over 24 h (kt)"
        }) + '<p class="an-note">Yellow bars = <b>rapid intensification</b> (wind up 30 kt or more in 24 h): <b>' +
            IC.ri_windows + "</b> of " + num(IC.windows_24h) + " windows (<b>" + num(IC.ri_rate_pct, 1) + "%</b>), spread over <b>" +
            IC.storms_with_ri + " of " + O.storms + "</b> storms. Mean change " + (IC.mean_change_kt >= 0 ? "+" : "") + IC.mean_change_kt + " kt.</p>" +
            '<p class="an-note">Share of 24 h windows that become rapid intensification, by starting class:</p>' +
            hbars(IC.ri_by_start_class.map((r, i) => ({
                label: r.class, color: COLORS[i], value: r.windows ? r.ri_windows / r.windows : 0,
                text: num(100 * r.ri_windows / Math.max(r.windows, 1), 1) + "%  (" + r.ri_windows + "/" + r.windows + ")"
            })));
        return card("anRI", "24-hour wind change", "Real change in maximum wind after 24 h", T_DATA, body);
    }

    /* -------------------------------------------------- model results -- */
    function evalReported() {
        const R = EV.reported, k = R.severe_cyclonic_storm_correct;
        const tiles = [
            ["~" + R.accuracy_pct_approx + "%", "exact accuracy", "on " + R.test_images + " unseen test images (" + R.test_storms + " storms)"],
            [R.within_one_class_pct + "%", "within one class", "off by at most one severity level"],
            [String(R.macro_f1), "macro-F1", "average over the 5 classes"],
            [R.ece_pct_approx != null ? "~" + R.ece_pct_approx + "%" : "n/a", "calibration gap (ECE)",
                R.ece_pct_approx != null ? "confidence is too high" : "not computed for this model - see Model Transparency"]
        ];
        const bars = hbars([
            { label: "Always guess Depression", value: R.test_share_depression_pct_approx, color: "#8fa5bf", text: "~" + num(R.test_share_depression_pct_approx, 1) + "%" },
            { label: "Final model (exact)", value: R.accuracy_pct_approx, color: "#19b5fe", text: "~" + R.accuracy_pct_approx + "%" },
            { label: "Final model (within 1 class)", value: R.within_one_class_pct, color: "#3ddc97", text: R.within_one_class_pct + "%" }
        ], 100);
        const kc = R.test_class_counts_known;
        const scsBlock = k
            ? "<h4>Severe Cyclonic Storm images classified correctly</h4>" + hbars([
                { label: "First model", value: k.first_model / k.of * 100, color: "#8fa5bf", text: k.first_model + "/" + k.of + " (" + num(100 * k.first_model / k.of, 1) + "%)" },
                { label: "Final model", value: k.final_model / k.of * 100, color: "#19b5fe", text: k.final_model + "/" + k.of + " (" + num(100 * k.final_model / k.of, 1) + "%)" }
            ], 100)
            : "<h4>Test set - class breakdown</h4>" +
                '<div class="an-kpis four">' + [
                    ["Cyclonic Storm", kc["Cyclonic Storm"]], ["Severe Cyclonic Storm", kc["Severe Cyclonic Storm"]],
                    ["Very Severe Cyclonic Storm", kc["Very Severe Cyclonic Storm"]], ["Extremely Severe/Super Cyclone", kc["Extremely Severe/Super Cyclone"]]
                ].map(t => '<div class="an-kpi"><strong>' + t[1] + "</strong><span>" + esc(t[0]) + "</span></div>").join("") + "</div>" +
                '<p class="an-note">Per-class correct-count for the current model isn\'t shown here: ' + esc(R.severe_cyclonic_storm_correct_note || "the training notebook only printed the aggregate accuracy above, not a per-image breakdown.") + "</p>";
        return '<div class="an-two"><div>' +
            '<div class="an-kpis four">' + tiles.map(t =>
                '<div class="an-kpi"><strong>' + esc(t[0]) + "</strong><span>" + esc(t[1]) + "</span><small>" + esc(t[2]) + "</small></div>").join("") + "</div>" +
            "<h4>Accuracy vs the do-nothing baseline</h4>" + bars +
            '<p class="an-note">"Always guess Depression" is right about ' + num(R.test_share_depression_pct_approx, 1) + '% of the time on this test set, so ~' +
            R.accuracy_pct_approx + '% exact-match is a modest gain on its own - ' + R.within_one_class_pct +
            '% within-one-class is the more meaningful number for a 5-class problem this fine-grained.</p>' +
            "</div><div>" +
            scsBlock +
            '<p class="an-note">Test set: ' + num(R.test_share_depression_pct_approx, 1) + "% Depression; only " +
            kc["Severe Cyclonic Storm"] + " Severe, " + kc["Very Severe Cyclonic Storm"] + " Very Severe and " +
            kc["Extremely Severe/Super Cyclone"] + " Extremely Severe images, so results for the strongest classes rest on very few images.</p>" +
            '<p class="an-note">Source: ' + esc(R.source) + ".</p>" +
            "</div></div>";
    }

    function evalComputed(c) {
        const isVal = c.split === "validation";
        let html = '<h4>' + (isVal ? "Validation set" : "Test set") + " - computed from " + esc(c.source_file) + " (" + num(c.n) + " images)" +
            (isVal ? ' <em class="tag plan">VALIDATION, not the test set</em>' : "") + " <em class=\"tag plan\">v2 MODEL - SUPERSEDED</em></h4>" +
            '<p class="an-note" style="color:#f2b73a"><strong>Heads up:</strong> this detailed confusion matrix, per-class table and reliability chart are real, ' +
            "unmodified predictions from the PRIOR v2 single-model classifier - kept here as a genuine worked example rather than deleted. " +
            "The model actually deployed today is the v3 wind-regression ensemble; its real aggregate numbers are the tiles above " +
            '(from MODAK.ipynb\'s own test run) and the full write-up is on the <a href="#modelTransparency">Model Transparency</a> panel. ' +
            "A per-image breakdown for the v3 ensemble (to regenerate this exact chart for the current model) needs an added export step " +
            "in the training notebook - it was not saved when the notebook last ran.</p>" +
            '<div class="an-kpis four">' + [
                [num(c.accuracy_pct, 1) + "%", "accuracy", "majority baseline " + num(c.majority_baseline_pct, 1) + "%"],
                [String(c.macro_f1), "macro-F1", c.macro_f1_note],
                [String(c.mean_error_severity_levels), "avg error", "severity levels; within one class: " + num(c.within_one_class_pct, 1) + "%"],
                [c.calibration ? num(c.calibration.ece_pct, 1) + "%" : "n/a", "calibration gap (ECE)", c.calibration ? "mean confidence " + num(c.calibration.mean_confidence_pct, 1) + "%" : "no confidence column"]
            ].map(t => '<div class="an-kpi"><strong>' + esc(t[0]) + "</strong><span>" + esc(t[1]) + "</span><small>" + esc(t[2]) + "</small></div>").join("") + "</div>";
        let calib = "";
        if (c.calibration && c.calibration.bins.length) {
            const pts = c.calibration.bins.map(b => [b.mean_confidence_pct, b.accuracy_pct, b]);
            calib = "<h4>Is the confidence trustworthy? (reliability)</h4>" + lines({
                W: 520, H: 250, xMin: 0, xMax: 100, yMin: 0, yMax: 100, xTitle: "Model confidence (%)", yTitle: "Actually correct (%)",
                xTicks: [0, 20, 40, 60, 80, 100].map(x => ({ x, label: String(x) })), aria: "Reliability diagram",
                series: [{ name: "perfect", color: "#607894", points: [[0, 0], [100, 100]], dash: "5 5", width: 1.6 },
                    { name: "model", color: "#19b5fe", points: pts, dots: true, tip: p => "confidence " + p[0] + "%, correct " + p[1] + "% (" + p[2].n + " images)" }]
            }) + '<p class="an-note">Points below the dashed line mean the model is more confident than it deserves. Points from bins with only a few images (hover to see the count) are noisy. Mean confidence when right: ' +
                pctOrNa(c.calibration.mean_confidence_correct_pct) + ", when wrong: " + pctOrNa(c.calibration.mean_confidence_wrong_pct) + ".</p>";
        }
        html += '<div class="an-two"><div>' + heat({ matrix: c.confusion_matrix, rowTitle: "true class", colTitle: "predicted class", aria: "Confusion matrix" }) + "</div><div>" +
            '<table class="an-table"><thead><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>' +
            c.per_class.map((r, i) => "<tr><td><i class=\"dot\" style=\"background:" + COLORS[i] + '"></i>' + esc(r.class) + "</td><td>" + r.support + "</td><td>" +
                (r.precision === null ? "-" : num(r.precision, 2)) + "</td><td>" + (r.recall === null ? "-" : num(r.recall, 2)) + "</td><td>" +
                (r.f1 === null ? "-" : num(r.f1, 2)) + "</td></tr>").join("") + "</tbody></table>" +
            '<p class="an-note">Precision = of the images predicted as this class, how many were right. Recall = of the images truly in this class, how many were found.</p>' +
            calib + "</div></div>";
        return html;
    }

    function cardEvaluation() {
        let body = evalReported();
        const sets = EV.computed || [];
        sets.forEach(c => { body += evalComputed(c); });
        if (!sets.length) {
            body += '<div class="an-empty"><strong>Not available: confusion matrix, per-class precision / recall / F1, reliability chart.</strong> ' +
                "They need the model's predictions on labelled images (columns <code>true_class, pred_class, confidence</code>). " +
                "The supplied files do not contain them, so nothing is drawn. Put the CSV in <code>analytics/inputs/test_predictions.csv</code> " +
                "and run <code>python analytics/build_analytics.py</code>.</div>";
        } else {
            const missing = [];
            if (!sets.some(c => c.split === "test")) { missing.push("per-image test-set predictions for the current model"); }
            if (!sets.some(c => c.calibration)) { missing.push("a confidence column, needed for the reliability chart"); }
            if (missing.length) {
                body += '<div class="an-empty"><strong>Still not available:</strong> ' + missing.join("; ") + ".</div>";
            }
        }
        return card("anEval", "Model performance", "How well the EfficientNet-B0 classifier does",
            (EV.computed && EV.computed.length) ? T_REP + T_COMP : T_REP, body, "span2");
    }

    /* ----------------------------------------------------- backtest ---- */
    function cardBacktest() {
        const lead = BT.lead_hours, pe = BT.position_error_km_mean, we = BT.wind_error_kt_mean;
        const at24 = i => i.at(-1);
        const pts = arr => arr.map((v, i) => [lead[i], v]);
        const xTicks = lead.filter(h => h % 6 === 0).map(h => ({ x: h, label: h ? "+" + h + "h" : "now" }));
        const posChart = lines({
            W: 560, H: 250, xMin: 0, xMax: 24, yMin: 0, yMax: niceMax(Math.max(at24(pe.analog), at24(pe.stay))), xTicks,
            yTitle: "Position error (km)", aria: "Position error by lead time",
            series: [
                { name: "Dashboard simulated outlook", color: "#fbbf24", points: pts(pe.analog), dots: true },
                { name: "Storm stays where it is", color: "#8fa5bf", points: pts(pe.stay), dash: "6 5" },
                { name: "Keeps its last 6 h motion", color: "#36d399", points: pts(pe.motion) }]
        });
        const windChart = lines({
            W: 560, H: 250, xMin: 0, xMax: 24, yMin: 0, yMax: niceMax(Math.max(at24(we.analog), at24(we.hold))), xTicks,
            yTitle: "Wind error (kt)", aria: "Wind error by lead time",
            series: [
                { name: "Dashboard simulated outlook", color: "#fbbf24", points: pts(we.analog), dots: true },
                { name: "Wind stays at its starting value", color: "#8fa5bf", points: pts(we.hold), dash: "6 5" }]
        });
        const a = at24(pe.analog), s = at24(pe.stay), m = at24(pe.motion);
        const trackVerdict = a > s
            ? "worse than simply assuming the storm stays put (" + num(s) + " km)"
            : "better than assuming the storm stays put (" + num(s) + " km)";
        const wa = at24(we.analog), wh = at24(we.hold);
        const windVerdict = wa > wh ? "worse than holding the wind constant (" + num(wh, 1) + " kt)" : "better than holding the wind constant (" + num(wh, 1) + " kt)";
        const ri = BT.rapid_intensification_flag;
        const base = 100 * ri.actual_ri_windows / BT.windows;
        const body = '<div class="an-two">' +
            "<div><h4>Track: average distance from the real position</h4>" + posChart + "</div>" +
            "<div><h4>Wind: average error</h4>" + windChart + "</div></div>" +
            '<div class="an-legend"><span><i style="background:#fbbf24"></i>Dashboard simulated outlook</span>' +
            '<span><i style="background:#8fa5bf"></i>Baseline: no change</span><span><i style="background:#36d399"></i>Baseline: keep last motion</span></div>' +
            '<div class="an-verdict"><b>Result on ' + num(BT.windows) + " historical 24 h windows:</b> after 24 h the simulated track is on average <b>" +
            num(a) + " km</b> from where the storm really went, " + trackVerdict + "; keeping the storm's last 6 h motion gives <b>" + num(m) + " km</b>. " +
            "The simulated wind is off by <b>" + num(wa, 1) + " kt</b>, " + windVerdict + ". " +
            "The simulated track is closer than the no-change guess in only <b>" + num(BT.analog_closer_than_stay_at_24h_pct, 1) + "%</b> of cases.</div>" +
            '<h4>Rapid-intensification flag (simulated) vs what happened</h4>' +
            '<table class="an-table"><thead><tr><th></th><th>Real RI</th><th>No real RI</th></tr></thead><tbody>' +
            "<tr><td>Flag raised</td><td>" + ri.true_positive + "</td><td>" + ri.false_positive + "</td></tr>" +
            "<tr><td>No flag</td><td>" + ri.false_negative + "</td><td>" + ri.true_negative + "</td></tr></tbody></table>" +
            '<p class="an-note">Precision <b>' + num(100 * ri.precision, 0) + "%</b> (flags that were right), recall <b>" + num(100 * ri.recall, 0) +
            "%</b> (real events caught). Real RI happens in " + num(base, 1) + "% of windows, so the flag is " +
            (100 * ri.precision > base ? "better" : "not better") + " than chance" +
            (ri.recall < 0.5 ? ", but it misses most real events" : "") +
            (ri.false_positive > ri.true_positive ? " and raises more false alarms than hits" : "") + ".</p>" +
            '<p class="an-note">' + esc(BT.assumptions) + " Method: leave-one-storm-out, i.e. the storm being scored is removed from the search, exactly as the dashboard does for its sample images.</p>";
        return card("anBack", "Honesty check: how good is the simulated outlook?",
            "The dashboard's track / wind / rapid-intensification outlook, scored against real IBTrACS history",
            T_COMP + tag("SIMULATED METHOD", "sim"), body, "span2");
    }

    /* ---------------------------------------- live: storm + session log */
    const session = [];   /* scored sample predictions, newest last */

    function cardStorm() {
        return card("anStorm", "Selected storm: real history", "Real IBTrACS wind history of the storm behind the selected sample image (Run Demo, or pick a sample above)",
            T_DATA, '<div id="anStormBody"><div class="an-empty">No sample image selected yet. Uploaded images have no known storm, so this panel works with the built-in sample images.</div></div>',
            "span2");
    }

    function stormBody(sample, pred, source) {
        const s = stormById(sample.storm_id);
        if (!s) { return '<div class="an-empty">Storm ' + esc(sample.storm_id) + " is not in the track file.</div>"; }
        const t0 = parseT(s.t[0]).getTime();
        const hrs = s.t.map(t => (parseT(t).getTime() - t0) / 3600000);
        const idx = s.t.indexOf(sample.time);
        const days = hrs[hrs.length - 1] / 24;
        const step = days > 10 ? 96 : days > 5 ? 48 : 24;
        const xt = [];
        for (let h = 0; h <= hrs[hrs.length - 1]; h += step) {
            const d = new Date(t0 + h * 3600000);
            xt.push({ x: h, label: d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] });
        }
        const starts = D.wind_hist.class_start_kt;
        const hl = starts.slice(1).map((kt, i) => ({ y: kt, label: SHORT[i + 1] + " " + kt + " kt", color: COLORS[i + 1] }));
        const peakV = Math.max(...s.v);
        const chart = lines({
            W: 720, H: 260, xMin: 0, xMax: hrs[hrs.length - 1], yMin: 0, yMax: niceMax(Math.max(peakV, 100)), xTicks: xt,
            yTitle: "Max wind (kt)", xTitle: "Date (UTC), storm " + s.id, aria: "Wind history of storm " + s.id,
            series: [{ name: "wind", color: "#19b5fe", points: hrs.map((h, i) => [h, s.v[i]]) }], hlines: hl,
            markers: idx >= 0 ? [{ x: hrs[idx], y: s.v[idx], color: COLORS[s.c[idx]], tip: "Sample image: " + fmtT(sample.time) + ", " + s.v[idx] + " kt" }] : []
        });
        const label = sample.true_class;
        let verdict = "";
        if (pred && source === "model") {
            const diff = Math.abs(CLASSES.indexOf(pred) - CLASSES.indexOf(label));
            verdict = pred === label ? "match" : "off by " + diff + " class" + (diff > 1 ? "es" : "");
        } else if (pred && source === "placeholder") {
            verdict = "placeholder (demo mode)";
        }
        const facts = [
            ["Storm", s.id + " (" + s.year + " season)"],
            ["Observed for", num(days, 1) + " days, " + s.t.length + " observations"],
            ["Peak wind", peakV + " kt (" + CLASSES[Math.max(...s.c)] + ")"],
            ["Image time", fmtT(sample.time)],
            ["Dataset label", label + " (" + sample.vmax + " kt)"],
            ["Model prediction", pred ? pred + (verdict ? " - " + verdict : "") : "-"]
        ];
        return '<div class="an-two wide"><div>' + chart + '</div><div class="an-facts">' +
            facts.map(f => "<div><span>" + esc(f[0]) + "</span><strong>" + esc(f[1]) + "</strong></div>").join("") + "</div></div>" +
            '<p class="an-note">Blue line: real IBTrACS maximum wind. Dashed lines: class boundaries. Coloured dot: the moment of the satellite image. ' +
            "The dataset label is the class of the IBTrACS wind at that time.</p>";
    }

    function cardSession() {
        return card("anSession", "Live prediction log", "Every sample image you run through the model in this session, scored against its dataset label",
            tag("MODEL vs DATA", "real"), '<div id="anSessionBody"></div>', "span2");
    }

    function sessionBody(note) {
        if (!session.length) {
            return '<div class="an-empty">' + (note || "Nothing scored yet. Run sample images through the model and each result appears here.") + "</div>";
        }
        const correct = session.filter(r => r.pred === r.label).length;
        const within1 = session.filter(r => Math.abs(CLASSES.indexOf(r.pred) - CLASSES.indexOf(r.label)) <= 1).length;
        const meanErr = session.reduce((a, r) => a + Math.abs(CLASSES.indexOf(r.pred) - CLASSES.indexOf(r.label)), 0) / session.length;
        const rows = session.map((r, i) => {
            const d = CLASSES.indexOf(r.pred) - CLASSES.indexOf(r.label);
            return "<tr><td>" + (i + 1) + "</td><td>" + esc(r.file) + "</td><td>" + esc(r.storm) + "</td><td><i class=\"dot\" style=\"background:" +
                COLORS[CLASSES.indexOf(r.label)] + '"></i>' + esc(r.label) + '</td><td><i class="dot" style="background:' + COLORS[CLASSES.indexOf(r.pred)] + '"></i>' +
                esc(r.pred) + "</td><td>" + num(r.conf, 1) + "%</td><td class=\"" + (d === 0 ? "ok" : "bad") + '">' +
                (d === 0 ? "correct" : "off by " + Math.abs(d) + (d > 0 ? " (too high)" : " (too low)")) + "</td></tr>";
        }).join("");
        return '<div class="an-kpis four">' + [
            [session.length + " / " + D.samples.length, "sample images scored", "each image counted once"],
            [correct + " correct", num(100 * correct / session.length, 0) + "% of scored", "not a test-set result"],
            [within1 + " within 1 class", num(100 * within1 / session.length, 0) + "% of scored", "adjacent-class misses"],
            [num(meanErr, 2), "average error", "severity levels"]
        ].map(t => '<div class="an-kpi"><strong>' + esc(t[0]) + "</strong><span>" + esc(t[1]) + "</span><small>" + esc(t[2]) + "</small></div>").join("") + "</div>" +
            '<div class="an-scroll"><table class="an-table"><thead><tr><th>#</th><th>Image</th><th>Storm</th><th>Dataset label</th><th>Model</th><th>Confidence</th><th>Result</th></tr></thead><tbody>' +
            rows + "</tbody></table></div>" +
            '<p class="an-note">Only ' + D.samples.length + " demo images exist here, so this is an illustration, not an accuracy measurement. " +
            'The real test-set numbers are in "Model performance". <button type="button" class="link-btn" id="anClearLog">Clear log</button></p>';
    }

    /* ------------------------------------------------------- assemble -- */
    root.innerHTML = header() + kpis() + '<div class="an-grid">' +
        cardStorm() + cardEvaluation() + cardClassDistribution() + cardWindHist() + cardMap() +
        cardMonthly() + cardYearly() + cardTransitions() + cardIntensityChange() + cardBacktest() + cardSession() +
        "</div>";
    el("anSessionBody").innerHTML = sessionBody();

    function el(id) { return document.getElementById(id); }

    /* ---------------------------------------------------------- events -- */
    root.addEventListener("click", ev => {
        const chip = ev.target.closest(".chip");
        if (chip) {
            const c = Number(chip.dataset.cls);
            if (mapState.visible.has(c)) { mapState.visible.delete(c); } else { mapState.visible.add(c); }
            chip.classList.toggle("on");
            el("anMapBox").innerHTML = basinSvg();
            return;
        }
        const seg = ev.target.closest("#anTransSeg button");
        if (seg) {
            transMode = seg.dataset.mode;
            root.querySelectorAll("#anTransSeg button").forEach(b => b.classList.toggle("active", b === seg));
            el("anTransBody").innerHTML = transHtml();
            return;
        }
        if (ev.target.id === "anClearLog") {
            session.length = 0;
            el("anSessionBody").innerHTML = sessionBody();
        }
    });

    /* the live link with app.js */
    document.addEventListener("vayuvega:prediction", ev => {
        const data = ev.detail || {};
        const meta = data.meta || {}, sources = data.sources || {}, pred = data.prediction || {};
        const sample = meta.image ? sampleByFile(meta.image) : null;

        if (!sample) {
            el("anStormBody").innerHTML = '<div class="an-empty">' +
                (meta.image ? "Uploaded images have no known storm ID, so there is no real history to show." :
                    "No image was used (rule-based result), so there is no storm history to show.") + "</div>";
            mapState.storm = null; mapState.sample = null;
            el("anMapBox").innerHTML = basinSvg();
            return;
        }

        el("anStormBody").innerHTML = stormBody(sample, pred.category, sources.category);
        mapState.storm = stormById(sample.storm_id) || null;
        mapState.sample = sample;
        el("anMapBox").innerHTML = basinSvg();

        if (sources.category === "model" && pred.category) {
            const rec = {
                file: sample.filename, storm: sample.storm_id, label: sample.true_class,
                pred: pred.category, conf: Number(pred.confidence)
            };
            const at = session.findIndex(r => r.file === rec.file);
            if (at >= 0) { session[at] = rec; } else { session.push(rec); }
            el("anSessionBody").innerHTML = sessionBody();
        } else if (sources.category === "placeholder") {
            el("anSessionBody").innerHTML = sessionBody(
                "Demo mode: model.pth is not loaded, so predictions are placeholders and are NOT scored. Load the real model to fill this log.");
        }
    });
})();
