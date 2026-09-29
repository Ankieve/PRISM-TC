"""Checks for the Analytics pipeline.   Run:  python analytics/tests/test_analytics.py

Every expected number below is either recomputed independently from the raw
CSV or worked out by hand, so a pass means the dashboard numbers are right.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYTICS = HERE.parent
sys.path.insert(0, str(ANALYTICS))

import build_analytics as ba  # noqa: E402

RAW = pd.read_csv(ba.TRACK_CSV)
RAW["dt"] = pd.to_datetime(RAW["time"].astype(str), format="%Y%m%d%H")
DATA = json.loads(ba.OUT_JSON.read_text())


class TestHelpers(unittest.TestCase):
    def test_haversine(self):
        self.assertAlmostEqual(ba.haversine_km(10, 80, 10, 80), 0.0)
        self.assertAlmostEqual(ba.haversine_km(0, 0, 1, 0), 111.195, places=2)
        self.assertAlmostEqual(ba.haversine_km(0, 0, 0, 180), math.pi * ba.EARTH_KM, places=3)


class TestDataMatchesRawCsv(unittest.TestCase):
    def test_output_is_up_to_date(self):
        self.assertEqual(DATA["meta"]["inputs"]["backend/data/track_data.csv"], ba.sha_short(ba.TRACK_CSV),
                         "track_data.csv changed: run python analytics/build_analytics.py")

    def test_overview(self):
        o = DATA["overview"]
        self.assertEqual(o["records"], len(RAW))
        self.assertEqual(o["storms"], RAW["storm_id"].nunique())
        self.assertAlmostEqual(o["wind_kt_mean"], RAW["vmax"].mean(), places=1)

    def test_class_counts(self):
        vc = RAW["category"].value_counts()
        for row in DATA["class_distribution"]["rows"]:
            self.assertEqual(row["records"], int(vc[row["class"]]))
        self.assertEqual(sum(r["storms_reaching_as_peak"] for r in DATA["class_distribution"]["rows"]),
                         RAW["storm_id"].nunique())

    def test_histograms_and_months_cover_every_record(self):
        wh = DATA["wind_hist"]["counts_by_class"]
        self.assertEqual(sum(sum(v) for v in wh.values()), len(RAW))
        se = DATA["seasonality"]["records_by_class"]
        self.assertEqual(sum(sum(v) for v in se.values()), len(RAW))
        self.assertEqual(sum(y["records"] for y in DATA["seasonality"]["yearly"]), len(RAW))
        self.assertEqual(sum(y["storms"] for y in DATA["seasonality"]["yearly"]), RAW["storm_id"].nunique())

    def test_sectors_partition_the_storms(self):
        s = DATA["sectors"]
        self.assertEqual(s["west"]["storms"] + s["east"]["storms"], RAW["storm_id"].nunique())
        self.assertEqual(s["west"]["records"] + s["east"]["records"], len(RAW))
        first = RAW.sort_values(["storm_id", "dt"]).groupby("storm_id")["lon"].first()
        self.assertEqual(s["west"]["storms"], int((first < ba.SECTOR_SPLIT_LON).sum()))

    def test_transitions_manual(self):
        d = RAW.sort_values(["storm_id", "dt"]).reset_index(drop=True)
        d["cls"] = d["category"].map(ba.CLASS_IDX)
        pairs = 0
        m = np.zeros((5, 5), int)
        for _, g in d.groupby("storm_id"):
            t, c = g["dt"].tolist(), g["cls"].tolist()
            for i in range(len(g) - 1):
                if (t[i + 1] - t[i]).total_seconds() == 3 * 3600:
                    m[c[i], c[i + 1]] += 1
                    pairs += 1
        self.assertEqual(DATA["transitions"]["3h"]["pairs"], pairs)
        self.assertEqual(DATA["transitions"]["3h"]["matrix_counts"], m.tolist())

    def test_rapid_intensification_manual(self):
        ri = wins = 0
        for _, g in RAW.sort_values(["storm_id", "dt"]).groupby("storm_id"):
            t, v = g["dt"].tolist(), g["vmax"].tolist()
            for i in range(len(g) - 8):
                if (t[i + 8] - t[i]).total_seconds() == 24 * 3600:
                    wins += 1
                    ri += (v[i + 8] - v[i]) >= 30
        self.assertEqual(DATA["intensity_change"]["windows_24h"], wins)
        self.assertEqual(DATA["intensity_change"]["ri_windows"], int(ri))

    def test_storm_tracks_are_complete(self):
        self.assertEqual(sum(len(s["t"]) for s in DATA["storms"]), len(RAW))
        for s in DATA["storms"]:
            self.assertTrue(len(s["t"]) == len(s["lat"]) == len(s["lon"]) == len(s["v"]) == len(s["c"]))

    def test_every_sample_is_found_in_the_tracks(self):
        ids = {s["id"]: s for s in DATA["storms"]}
        for smp in DATA["samples"]:
            st = ids[smp["storm_id"]]
            i = st["t"].index(smp["time"])
            self.assertEqual(st["v"][i], smp["vmax"])
            self.assertEqual(ba.CLASSES[st["c"][i]], smp["true_class"])


class TestBacktest(unittest.TestCase):
    def test_structure(self):
        bt = DATA["backtest"]
        self.assertGreater(bt["windows"], 1000)
        for series in bt["position_error_km_mean"].values():
            self.assertEqual(len(series), 9)
            self.assertEqual(series[0], 0.0)

    def test_stay_error_matches_independent_calculation(self):
        """Recompute the 'storm stays where it is' 24 h error with numpy only."""
        errs = []
        for _, g in RAW.sort_values(["storm_id", "dt"]).groupby("storm_id"):
            t = g["dt"].tolist()
            lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
            for k in range(2, len(g) - 8):
                span = t[k - 2:k + 9]
                if all((b - a).total_seconds() == 3 * 3600 for a, b in zip(span, span[1:])):
                    errs.append(ba.haversine_km(lat[k], lon[k], lat[k + 8], lon[k + 8]))
        self.assertEqual(DATA["backtest"]["windows"], len(errs))
        self.assertAlmostEqual(DATA["backtest"]["position_error_km_mean"]["stay"][8], float(np.mean(errs)), places=1)

    def test_ri_confusion_adds_up(self):
        f = DATA["backtest"]["rapid_intensification_flag"]
        self.assertEqual(f["true_positive"] + f["false_positive"] + f["false_negative"] + f["true_negative"],
                         DATA["backtest"]["windows"])


class TestEvaluation(unittest.TestCase):
    """Hand-worked example: 10 images."""

    def make(self, with_conf=True, **over):
        true = [0, 0, 0, 0, 1, 1, 2, 2, 3, 4]
        pred = [0, 0, 0, 1, 1, 2, 2, 2, 4, 4]
        df = pd.DataFrame({"true_class": [ba.CLASSES[i] for i in true], "pred_class": [ba.CLASSES[i] for i in pred]})
        if with_conf:
            df["confidence"] = [90, 90, 90, 60, 80, 55, 85, 85, 50, 70]
        for k, v in over.items():
            df[k] = v
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        df.to_csv(f.name, index=False)
        return f.name

    def test_metrics(self):
        r = ba.evaluate_predictions(self.make())
        self.assertEqual(r["n"], 10)
        self.assertEqual(r["accuracy_pct"], 70.0)
        self.assertEqual(r["confusion_matrix"], [[3, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
        pc = {p["class"]: p for p in r["per_class"]}
        self.assertEqual((pc[ba.CLASSES[0]]["precision"], pc[ba.CLASSES[0]]["recall"], pc[ba.CLASSES[0]]["f1"]), (1.0, 0.75, 0.857))
        self.assertEqual((pc[ba.CLASSES[1]]["precision"], pc[ba.CLASSES[1]]["recall"]), (0.5, 0.5))
        self.assertIsNone(pc[ba.CLASSES[3]]["precision"])          # never predicted
        self.assertEqual(pc[ba.CLASSES[3]]["f1"], 0.0)
        self.assertEqual(r["macro_f1"], 0.565)
        self.assertEqual(r["mean_error_severity_levels"], 0.3)
        self.assertEqual(r["within_one_class_pct"], 100.0)
        self.assertEqual(r["majority_baseline_pct"], 40.0)

    def test_calibration(self):
        r = ba.evaluate_predictions(self.make())
        self.assertEqual(r["calibration"]["ece_pct"], 27.5)
        self.assertEqual(r["calibration"]["mean_confidence_wrong_pct"], 55.0)   # 60,55,50 wrong

    def test_without_confidence(self):
        self.assertIsNone(ba.evaluate_predictions(self.make(with_conf=False))["calibration"])

    def test_confidence_given_as_fraction(self):
        path = self.make()
        df = pd.read_csv(path)
        df["confidence"] = df["confidence"] / 100
        df.to_csv(path, index=False)
        self.assertEqual(ba.evaluate_predictions(path)["calibration"]["ece_pct"], 27.5)

    def test_validation_file_column_names_and_split(self):
        path = self.make(with_conf=False)
        df = pd.read_csv(path).rename(columns={"true_class": "actual_label", "pred_class": "predicted_label"})
        df.to_csv(path, index=False)
        r = ba.evaluate_predictions(path, "validation")
        self.assertEqual((r["split"], r["n"], r["accuracy_pct"]), ("validation", 10, 70.0))

    def test_shipped_validation_file(self):
        """Independent recomputation of accuracy and macro-F1 for the real val file."""
        df = pd.read_csv(ba.VAL_PRED_CSV)
        y = df["actual_label"].map(ba.CLASS_IDX).to_numpy()
        p = df["predicted_label"].map(ba.CLASS_IDX).to_numpy()
        ev = [e for e in DATA["evaluation"]["computed"] if e["split"] == "validation"][0]
        self.assertEqual(ev["n"], len(df))
        self.assertAlmostEqual(ev["accuracy_pct"], 100 * (y == p).mean(), places=1)
        f1 = []
        for i in range(5):
            tp = ((y == i) & (p == i)).sum()
            pr = tp / max((p == i).sum(), 1)
            rc = tp / max((y == i).sum(), 1)
            f1.append(0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
        self.assertAlmostEqual(ev["macro_f1"], float(np.mean(f1)), places=3)

    def test_bad_files_are_rejected(self):
        with self.assertRaises(ValueError):
            ba.evaluate_predictions(self.make(pred_class="Hurricane"))
        with self.assertRaises(ValueError):
            ba.evaluate_predictions(self.make(confidence=150))
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        pd.DataFrame({"a": [1]}).to_csv(f.name, index=False)
        with self.assertRaises(ValueError):
            ba.evaluate_predictions(f.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
