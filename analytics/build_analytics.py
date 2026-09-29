"""VAYUVEGA - Data Visualisation & Analytics (Member 5)

Reads ONLY real project files and writes the numbers the dashboard's
Analytics section displays:

    backend/data/track_data.csv          IBTrACS-derived storm observations
    backend/samples/samples.csv          the 10 demo images and their labels
    inputs/reported_metrics.json         model numbers quoted in CLAUDE.md
    inputs/basemap.json                  Natural Earth coastline (offline map)
    inputs/val_predictions.csv           validation-set predictions (287 images)
    inputs/test_predictions.csv          OPTIONAL - Member 3's test-set predictions

Outputs (both hold identical content):
    analytics/analytics_data.json        for people / other tools
    analytics/analytics_data.js          loaded by index.html (works offline,
                                         even when the page is opened as a file)

Run from the PRISM-TC-Frontend folder:
    python analytics/build_analytics.py

Nothing here is invented: every number is computed from the files above.
The backtest re-uses Member 4's own logic.py (read-only) so it measures the
exact "simulated" outlook that the dashboard shows.
"""

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import logic  # noqa: E402  (Member 4's module, used read-only)

TRACK_CSV = BACKEND / "data" / "track_data.csv"
SAMPLES_CSV = BACKEND / "samples" / "samples.csv"
REPORTED_JSON = HERE / "inputs" / "reported_metrics.json"
BASEMAP_JSON = HERE / "inputs" / "basemap.json"
TEST_PRED_CSV = HERE / "inputs" / "test_predictions.csv"
VAL_PRED_CSV = HERE / "inputs" / "val_predictions.csv"
# column names accepted for the two required columns
ALIASES = {"actual_label": "true_class", "predicted_label": "pred_class"}
OUT_JSON = HERE / "analytics_data.json"
OUT_JS = HERE / "analytics_data.js"

CLASSES = logic.CLASS_ORDER
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
STEP_H = 3                      # IBTrACS/TCIR spacing used by the dataset
H24 = 8                         # 8 steps x 3 h = 24 h
SECTOR_SPLIT_LON = 77.5         # ~ southern tip of India (see sector note)
RI_KT = 30.0                    # rapid-intensification threshold (kt / 24 h)
EARTH_KM = 6371.0088


# ------------------------------------------------------------------ helpers
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def r(x, n=2):
    """round a numpy/py number to a plain float (None stays None)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, n)


def sha_short(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_tracks():
    db = logic.TrackDB(TRACK_CSV)
    df = db.df.copy()
    df["cls"] = df["category"].map(CLASS_IDX)
    if df["cls"].isna().any():
        bad = sorted(df.loc[df["cls"].isna(), "category"].unique())
        raise ValueError(f"track_data.csv has unknown categories: {bad}")
    df["cls"] = df["cls"].astype(int)
    df["year"] = df["storm_id"].str[:4].astype(int)     # season year from the ID
    return db, df


# ---------------------------------------------------------------- sections
def section_overview(df):
    gaps = 0
    for _, g in df.groupby("storm_id"):
        d = g["dt"].diff().dropna().dt.total_seconds() / 3600
        gaps += int((d != STEP_H).sum())
    return {
        "records": int(len(df)),
        "storms": int(df["storm_id"].nunique()),
        "first_obs": df["dt"].min().strftime("%Y-%m-%d"),
        "last_obs": df["dt"].max().strftime("%Y-%m-%d"),
        "season_years": [int(df["year"].min()), int(df["year"].max())],
        "step_hours": STEP_H,
        "irregular_steps": gaps,
        "lat_range": [r(df["lat"].min(), 1), r(df["lat"].max(), 1)],
        "lon_range": [r(df["lon"].min(), 1), r(df["lon"].max(), 1)],
        "wind_kt_range": [r(df["vmax"].min(), 0), r(df["vmax"].max(), 0)],
        "wind_kt_mean": r(df["vmax"].mean(), 1),
        "storms_per_record_mean": r(len(df) / df["storm_id"].nunique(), 1),
    }


def section_class_distribution(df):
    counts = df["cls"].value_counts().reindex(range(5), fill_value=0)
    peak = df.groupby("storm_id")["cls"].max().value_counts().reindex(range(5), fill_value=0)
    n = int(counts.sum())
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({
            "class": c, "index": i, "records": int(counts[i]),
            "pct": r(100 * counts[i] / n, 1),
            "storms_reaching_as_peak": int(peak[i]),
        })
    top = int(counts.max())
    return {
        "rows": rows,
        "majority_class": CLASSES[int(counts.idxmax())],
        "majority_baseline_pct": r(100 * top / n, 1),
        "imbalance_ratio_max_to_min": r(top / int(counts.min()), 1),
        "wind_bands_kt": logic.WIND_DISPLAY,
    }


def section_wind_hist(df):
    edges = list(range(10, 160, 10))
    by_class = {}
    for i, c in enumerate(CLASSES):
        h, _ = np.histogram(df.loc[df["cls"] == i, "vmax"], bins=edges)
        by_class[c] = [int(v) for v in h]
    return {"bin_edges_kt": edges, "counts_by_class": by_class,
            "class_start_kt": [0, 34, 48, 64, 90]}


def section_seasonality(df):
    month = df["dt"].dt.month
    by_class = {}
    for i, c in enumerate(CLASSES):
        m = month[df["cls"] == i].value_counts().reindex(range(1, 13), fill_value=0)
        by_class[c] = [int(v) for v in m.values]
    storms = df.groupby(month)["storm_id"].nunique().reindex(range(1, 13), fill_value=0)
    yearly = []
    for y, g in df.groupby("year"):
        yearly.append({"year": int(y), "storms": int(g["storm_id"].nunique()),
                       "records": int(len(g)), "peak_wind_kt": r(g["vmax"].max(), 0)})
    return {"months": list(range(1, 13)), "records_by_class": by_class,
            "storms_active": [int(v) for v in storms.values], "yearly": yearly}


def section_sectors(df):
    first = df.sort_values(["storm_id", "dt"]).groupby("storm_id").first()
    west_ids = set(first.index[first["lon"] < SECTOR_SPLIT_LON])
    df = df.assign(west=df["storm_id"].isin(west_ids))
    out = {}
    for key, flag in (("west", True), ("east", False)):
        g = df[df["west"] == flag]
        peak = g.groupby("storm_id")["cls"].max()
        out[key] = {
            "storms": int(g["storm_id"].nunique()),
            "records": int(len(g)),
            "mean_wind_kt": r(g["vmax"].mean(), 1),
            "peak_wind_kt": r(g["vmax"].max(), 0),
            "storms_reaching_very_severe_or_higher": int((peak >= 3).sum()),
        }
    return {
        "split_lon": SECTOR_SPLIT_LON,
        "rule": ("Each storm is assigned by the longitude of its FIRST recorded position: "
                 "west of 77.5E = Arabian Sea side, otherwise Bay of Bengal side. "
                 "This is an approximate rule made for this analysis; the basin field "
                 "of IBTrACS is not in the supplied file."),
        **out,
    }


def section_transitions(df):
    d = df.sort_values(["storm_id", "dt"]).reset_index(drop=True)
    out = {}
    for name, steps in (("3h", 1), ("24h", H24)):
        hours = steps * STEP_H
        nxt = d.groupby("storm_id")[["dt", "cls", "vmax"]].shift(-steps)
        ok = (nxt["dt"] - d["dt"]).dt.total_seconds() / 3600 == hours
        src, dst = d.loc[ok, "cls"].to_numpy(), nxt.loc[ok, "cls"].astype(int).to_numpy()
        m = np.zeros((5, 5), dtype=int)
        np.add.at(m, (src, dst), 1)
        diff = np.abs(src - dst)
        out[name] = {
            "pairs": int(ok.sum()),
            "matrix_counts": m.tolist(),
            "same_class_pct": r(100 * (diff == 0).mean(), 1),
            "one_class_change_pct": r(100 * (diff == 1).mean(), 1),
            "two_or_more_pct": r(100 * (diff >= 2).mean(), 1),
        }
    return out


def section_intensity_change(df):
    d = df.sort_values(["storm_id", "dt"]).reset_index(drop=True)
    nxt = d.groupby("storm_id")[["dt", "vmax"]].shift(-H24)
    ok = (nxt["dt"] - d["dt"]).dt.total_seconds() / 3600 == H24 * STEP_H
    delta = (nxt.loc[ok, "vmax"] - d.loc[ok, "vmax"]).to_numpy()
    start_cls = d.loc[ok, "cls"].to_numpy()
    edges = list(range(-70, 90, 10))
    h, _ = np.histogram(np.clip(delta, edges[0], edges[-1] - 1e-6), bins=edges)
    ri = delta >= RI_KT
    storms_ri = d.loc[ok][ri]["storm_id"].nunique()
    by_class = []
    for i, c in enumerate(CLASSES):
        m = start_cls == i
        by_class.append({"class": c, "windows": int(m.sum()), "ri_windows": int((ri & m).sum())})
    return {
        "windows_24h": int(ok.sum()),
        "bin_edges_kt": edges,
        "counts": [int(v) for v in h],
        "mean_change_kt": r(delta.mean(), 1),
        "ri_threshold_kt": RI_KT,
        "ri_windows": int(ri.sum()),
        "ri_rate_pct": r(100 * ri.mean(), 1),
        "storms_with_ri": int(storms_ri),
        "ri_by_start_class": by_class,
        "definition": "wind at +24 h minus wind now, both from IBTrACS; RI = rise of 30 kt or more",
    }


# ---------------------------------------------------------------- backtest
def _windows(df):
    """Rows k with 2 observations before and 8 after, all exactly 3 h apart."""
    d = df.sort_values(["storm_id", "dt"]).reset_index(drop=True)
    rows = []
    for sid, g in d.groupby("storm_id", sort=False):
        g = g.reset_index(drop=True)
        t = (g["dt"] - g["dt"].iloc[0]).dt.total_seconds().to_numpy()   # seconds
        for k in range(2, len(g) - H24):
            span = t[k - 2:k + H24 + 1]
            if np.all(np.diff(span) == STEP_H * 3600):
                rows.append((sid, g, k))
    return rows


def section_backtest(db, df):
    """Score the dashboard's simulated 24 h outlook against what really happened.

    Leave-one-storm-out: the storm being scored is excluded from the analog
    search (exactly what the dashboard does for its sample images).
    Start wind = midpoint of the TRUE class band, as the dashboard uses in
    image mode; so this is a best case that assumes the model got the class right.
    """
    lead_h = [STEP_H * i for i in range(H24 + 1)]
    err = {m: [[] for _ in lead_h] for m in ("analog", "stay", "motion")}
    werr = {m: [[] for _ in lead_h] for m in ("analog", "hold")}
    ri_tp = ri_fp = ri_fn = ri_tn = 0
    n_no_analog = 0
    wins = _windows(df)

    for sid, g, k in wins:
        lat0, lon0 = float(g.at[k, "lat"]), float(g.at[k, "lon"])
        cls = int(g.at[k, "cls"])
        w0 = logic.WIND_MID[cls]
        out = db.analog_outlook(cls, lat0, lon0, w0, exclude=(sid,))
        if out is None:
            n_no_analog += 1
            continue
        dlat = (lat0 - float(g.at[k - 2, "lat"])) / 2.0
        dlon = (lon0 - float(g.at[k - 2, "lon"])) / 2.0
        actual_w = []
        for i in range(H24 + 1):
            alat, alon = float(g.at[k + i, "lat"]), float(g.at[k + i, "lon"])
            aw = float(g.at[k + i, "vmax"])
            actual_w.append(aw)
            err["analog"][i].append(haversine_km(out["track"][i][0], out["track"][i][1], alat, alon))
            err["stay"][i].append(haversine_km(lat0, lon0, alat, alon))
            err["motion"][i].append(haversine_km(lat0 + dlat * i, lon0 + dlon * i, alat, alon))
            werr["analog"][i].append(abs(out["forecast"][i]["wind"] - aw))
            werr["hold"][i].append(abs(w0 - aw))
        pred_ri = logic.rapid_intensification([f["wind"] for f in out["forecast"]])
        true_ri = logic.rapid_intensification(actual_w)
        ri_tp += pred_ri and true_ri
        ri_fp += pred_ri and not true_ri
        ri_fn += (not pred_ri) and true_ri
        ri_tn += (not pred_ri) and (not true_ri)

    n = len(err["analog"][0])
    mean = lambda lst: [r(np.mean(x), 1) for x in lst]        # noqa: E731
    med = lambda lst: [r(np.median(x), 1) for x in lst]       # noqa: E731
    better = float(np.mean(np.array(err["analog"][H24]) < np.array(err["stay"][H24])))
    prec = ri_tp / (ri_tp + ri_fp) if (ri_tp + ri_fp) else None
    rec = ri_tp / (ri_tp + ri_fn) if (ri_tp + ri_fn) else None
    return {
        "windows": int(n),
        "windows_without_analog": int(n_no_analog),
        "lead_hours": lead_h,
        "position_error_km_mean": {m: mean(v) for m, v in err.items()},
        "position_error_km_median": {m: med(v) for m, v in err.items()},
        "wind_error_kt_mean": {m: mean(v) for m, v in werr.items()},
        "analog_closer_than_stay_at_24h_pct": r(100 * better, 1),
        "rapid_intensification_flag": {
            "true_positive": int(ri_tp), "false_positive": int(ri_fp),
            "false_negative": int(ri_fn), "true_negative": int(ri_tn),
            "precision": r(prec, 3), "recall": r(rec, 3),
            "actual_ri_windows": int(ri_tp + ri_fn),
        },
        "methods": {
            "analog": "Dashboard's simulated outlook (copies a similar historical storm), storm itself excluded",
            "stay": "Baseline: storm stays where it is",
            "motion": "Baseline: keeps moving at its last 6 h speed and direction",
            "hold": "Baseline for wind: stays at the starting (class-midpoint) wind",
        },
        "assumptions": ("Start wind = midpoint of the true IMD class band (what the dashboard "
                        "uses when an image is given), so this assumes the classifier was right. "
                        "Only windows with 6 h of past track and 24 h of future track at a "
                        "regular 3 h spacing are scored."),
    }


# -------------------------------------------------------------- evaluation
def evaluate_predictions(path, split="test"):
    """Confusion matrix, per-class P/R/F1, error and calibration from a CSV.

    Required columns : true_class, pred_class        (names as in CLASSES;
                       actual_label / predicted_label are accepted too)
    Optional column  : confidence                    (percent 0-100; 0-1 accepted)
    split            : "test" or "validation" - only used as a label
    """
    df = pd.read_csv(path).rename(columns=ALIASES)
    missing = {"true_class", "pred_class"} - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing column(s) {sorted(missing)}")
    for col in ("true_class", "pred_class"):
        unknown = sorted(set(df[col].astype(str)) - set(CLASSES))
        if unknown:
            raise ValueError(f"{Path(path).name}: unknown class name(s) in {col}: {unknown}")
    if df.empty:
        raise ValueError(f"{Path(path).name}: no rows")
    y = df["true_class"].map(CLASS_IDX).to_numpy()
    p = df["pred_class"].map(CLASS_IDX).to_numpy()
    n = len(df)
    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (y, p), 1)
    per_class = []
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        sup = int(cm[i].sum())
        pred_n = int(cm[:, i].sum())
        prec = tp / pred_n if pred_n else None
        rec = tp / sup if sup else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if sup else None)
        if sup:
            f1s.append(f1)
        per_class.append({"class": c, "support": sup, "predicted": pred_n,
                          "precision": r(prec, 3), "recall": r(rec, 3), "f1": r(f1, 3)})
    ordinal = np.abs(y - p)
    out = {
        "source_file": Path(path).name,
        "split": split,
        "n": int(n),
        "accuracy_pct": r(100 * (y == p).mean(), 1),
        "macro_f1": r(np.mean(f1s), 3),
        "macro_f1_note": "average of the F1 of classes that occur in this file",
        "mean_error_severity_levels": r(ordinal.mean(), 2),
        "within_one_class_pct": r(100 * (ordinal <= 1).mean(), 1),
        "majority_baseline_pct": r(100 * cm.sum(axis=1).max() / n, 1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "calibration": None,
    }
    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        if conf.isna().any() or (conf < 0).any() or conf.max() > 100:
            raise ValueError("confidence must be a number between 0 and 100 (percent)")
        conf = conf.to_numpy()
        if conf.max() <= 1.0:
            conf = conf * 100.0
        correct = (y == p).astype(float)
        bins = np.linspace(0, 100, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        rows, ece = [], 0.0
        for b in range(10):
            m = idx == b
            if not m.any():
                continue
            acc_b, conf_b = 100 * correct[m].mean(), conf[m].mean()
            ece += m.sum() / n * abs(acc_b - conf_b)
            rows.append({"bin": f"{int(bins[b])}-{int(bins[b + 1])}", "n": int(m.sum()),
                         "mean_confidence_pct": r(conf_b, 1), "accuracy_pct": r(acc_b, 1)})
        out["calibration"] = {
            "ece_pct": r(ece, 1),
            "mean_confidence_pct": r(conf.mean(), 1),
            "mean_confidence_correct_pct": r(conf[correct == 1].mean(), 1) if correct.any() else None,
            "mean_confidence_wrong_pct": r(conf[correct == 0].mean(), 1) if (correct == 0).any() else None,
            "bins": rows,
        }
    return out


def section_evaluation():
    reported = json.loads(REPORTED_JSON.read_text())
    computed = []                       # one entry per prediction file that exists
    if TEST_PRED_CSV.exists():
        computed.append(evaluate_predictions(TEST_PRED_CSV, "test"))
    if VAL_PRED_CSV.exists():
        computed.append(evaluate_predictions(VAL_PRED_CSV, "validation"))
    return {"reported": reported, "computed": computed}


# ------------------------------------------------------------------ output
def section_storms(df):
    storms = []
    d = df.sort_values(["storm_id", "dt"])
    first_lon = d.groupby("storm_id")["lon"].first()
    for sid, g in d.groupby("storm_id", sort=True):
        storms.append({
            "id": sid,
            "year": int(g["year"].iloc[0]),
            "sector": "west" if first_lon[sid] < SECTOR_SPLIT_LON else "east",
            "t": [int(x) for x in g["time"]],
            "lat": [r(x, 1) for x in g["lat"]],
            "lon": [r(x, 1) for x in g["lon"]],
            "v": [int(x) for x in g["vmax"]],
            "c": [int(x) for x in g["cls"]],
        })
    return storms


def section_samples():
    s = pd.read_csv(SAMPLES_CSV)
    return [{"filename": row.filename, "storm_id": row.storm_id, "time": int(row.time),
             "true_class": row.true_class, "vmax": float(row.vmax),
             "lat": float(row.lat), "lon": float(row.lon)} for row in s.itertuples()]


def build():
    db, df = load_tracks()
    print(f"loaded {len(df)} records / {df['storm_id'].nunique()} storms")
    data = {
        "meta": {
            "project": "VAYUVEGA (SIH26070)",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "inputs": {
                "backend/data/track_data.csv": sha_short(TRACK_CSV),
                "backend/samples/samples.csv": sha_short(SAMPLES_CSV),
                "test_predictions.csv": "present" if TEST_PRED_CSV.exists() else "not supplied",
                "val_predictions.csv": "present" if VAL_PRED_CSV.exists() else "not supplied",
            },
        },
        "classes": CLASSES,
        "overview": section_overview(df),
        "class_distribution": section_class_distribution(df),
        "wind_hist": section_wind_hist(df),
        "seasonality": section_seasonality(df),
        "sectors": section_sectors(df),
        "transitions": section_transitions(df),
        "intensity_change": section_intensity_change(df),
    }
    print("running backtest of the simulated outlook ...")
    data["backtest"] = section_backtest(db, df)
    data["evaluation"] = section_evaluation()
    data["storms"] = section_storms(df)
    data["samples"] = section_samples()
    data["basemap"] = json.loads(BASEMAP_JSON.read_text())

    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    OUT_JSON.write_text(text)
    OUT_JS.write_text(
        "/* generated by analytics/build_analytics.py - do not edit by hand */\n"
        "window.VAYUVEGA_ANALYTICS = " + text + ";\n")
    print(f"wrote {OUT_JSON.name} and {OUT_JS.name} ({len(text) / 1024:.0f} KB)")
    return data


if __name__ == "__main__":
    build()
