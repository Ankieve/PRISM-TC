"""PRISM-TC backend logic (no web code here, so it is easy to test).

What is REAL vs SIMULATED in this prototype
-------------------------------------------
REAL       : intensity class + confidence + class probabilities (trained
             EfficientNet-B0 model, see predict.py).
SIMULATED  : track outlook, wind outlook, rapid-intensification flag
             (analog method: copy how historical IBTrACS storms of the same
             intensity moved), pressure estimate and risk index (simple rules).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_ORDER = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe/Super Cyclone",
]

# IMD wind bands in knots: (low, high) - used for display and rule-based class
WIND_BANDS = [(0, 33), (34, 47), (48, 63), (64, 89), (90, 200)]
WIND_DISPLAY = ["up to 33", "34-47", "48-63", "64-89", "90+"]  # shown as "34-47"
WIND_MID = [25.0, 41.0, 56.0, 77.0, 100.0]

HORIZON_STEPS = 8          # 8 x 3 h = 24 h outlook
MIN_PAST_POINTS = 4


# ----------------------------------------------------------- simple rules ---
def class_from_wind(wind_kt):
    """IMD category index from a wind speed in knots (rule, not AI)."""
    for i, (lo, hi) in enumerate(WIND_BANDS):
        if wind_kt <= hi:
            return i
    return 4


def estimate_pressure(wind_kt):
    """Central pressure (hPa) from max wind via Atkinson-Holliday:
    Vmax = 6.7 * (1010 - Pc) ** 0.644. A rough estimate, not a measurement."""
    wind_kt = max(float(wind_kt), 1.0)
    return round(1010.0 - (wind_kt / 6.7) ** (1.0 / 0.644), 1)


def risk_index(class_index, wind_kt):
    """0-100 rule-based score: 60% from wind speed, 40% from class."""
    wind_part = min(max(wind_kt, 0.0) / 120.0, 1.0) * 100.0
    class_part = class_index / 4.0 * 100.0
    return round(0.6 * wind_part + 0.4 * class_part, 1)


def rapid_intensification(winds):
    """True if wind rises by >= 30 kt within the outlook (24 h) - standard RI test."""
    if len(winds) < 2:
        return False
    return (max(winds[1:]) - winds[0]) >= 30.0


def environment_favors_intensification(sst_c, wind_shear_kt, humidity_pct=None, *,
                                       sst_land=False):
    """Simple, documented meteorological heuristic - NOT machine learning and NOT
    connected to the analog track method above. It only fires when the user has
    actually supplied sst/wind_shear (both optional API fields); with either one
    missing it returns None so the frontend can show "not enough data" rather than
    a fabricated verdict.

    Thresholds (textbook / operationally common bands, not fitted to any data):
      - Sea-surface temperature >= 26.5 C is the widely-cited minimum needed to
        sustain and intensify a tropical cyclone (warm-ocean-heat-content rule of
        thumb used across tropical meteorology).
      - Vertical wind shear <= 10 kt is considered low/favorable for
        intensification; >= 20 kt is considered high/unfavorable (bands similar
        to those used in NHC-style operational discussions). 10-20 kt is treated
        as a neutral/mixed zone.

    humidity_pct (optional, 700 hPa mid-level relative humidity, e.g. from ERA5):
      - Dry mid-level air is a well-documented inhibitor of tropical cyclone
        intensification (entrainment of dry air suppresses convection - Gray
        1968; used operationally as one input to NHC's SHIPS rapid-
        intensification index). Below 40% is treated here as dry enough to
        downgrade an otherwise-"favorable" verdict to "neutral".
      - This is intentionally ONE-DIRECTIONAL and conservative: high humidity
        never upgrades an "unfavorable" SST/shear verdict to something better,
        since moisture alone does not overcome cold water or strong shear.
      - When humidity_pct is None (not supplied), behaviour is identical to
        before this parameter existed.

    Returns one of "favorable", "unfavorable", "neutral", "not_applicable", or None.
    "not_applicable" is returned ONLY when sst_land is True and no SST was
    supplied: the point is over land, ERA5 SST is ocean-only, and there is no
    sea surface to check - so no favourable/unfavourable verdict is possible.
    A null SST must never be treated as 0C (that would fake "unfavorable").
    """
    if sst_land and sst_c is None:
        return "not_applicable"
    if sst_c is None or wind_shear_kt is None:
        return None
    if sst_c >= 26.5 and wind_shear_kt <= 10.0:
        verdict = "favorable"
    elif sst_c < 26.0 or wind_shear_kt >= 20.0:
        verdict = "unfavorable"
    else:
        verdict = "neutral"
    if verdict == "favorable" and humidity_pct is not None and humidity_pct < 40.0:
        verdict = "neutral"
    return verdict


# ---------------------------------------------------------------- tracks ----
class TrackDB:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df["dt"] = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H")
        df = df.sort_values(["storm_id", "dt"]).reset_index(drop=True)
        df["k"] = df.groupby("storm_id").cumcount()
        df["n"] = df.groupby("storm_id")["k"].transform("max") + 1
        self.df = df

    def past_track(self, storm_id, dt, n_points=10):
        """Real observed positions of a storm up to and including time dt."""
        s = self.df[(self.df["storm_id"] == storm_id) & (self.df["dt"] <= dt)]
        s = s.tail(n_points)
        return [[float(a), float(b)] for a, b in zip(s["lat"], s["lon"])]

    def analog_outlook(self, class_index, lat, lon, wind_now, exclude=()):
        """Find the historical point of the same class that is closest to
        (lat, lon) and has a full 24 h of following positions; copy its
        relative motion and wind change to the requested position."""
        df = self.df
        cat = CLASS_ORDER[class_index]
        cand = df[(df["category"] == cat)
                  & (df["k"] + HORIZON_STEPS < df["n"])
                  & (~df["storm_id"].isin(list(exclude)))]
        if cand.empty:  # relax the class requirement
            cand = df[(df["k"] + HORIZON_STEPS < df["n"])
                      & (~df["storm_id"].isin(list(exclude)))]
        if cand.empty:
            return None
        coslat = math.cos(math.radians(lat))
        dist = np.hypot(cand["lat"].to_numpy() - lat,
                        (cand["lon"].to_numpy() - lon) * coslat)
        best = cand.iloc[int(np.argmin(dist))]
        seg = df[(df["storm_id"] == best["storm_id"])
                 & (df["k"] >= best["k"])
                 & (df["k"] <= best["k"] + HORIZON_STEPS)].sort_values("k")
        track, forecast = [], []
        v0 = float(seg["vmax"].iloc[0])
        t0 = seg["dt"].iloc[0]
        for _, r in seg.iterrows():
            track.append([round(lat + (float(r["lat"]) - float(seg["lat"].iloc[0])), 3),
                          round(lon + (float(r["lon"]) - float(seg["lon"].iloc[0])), 3)])
            hours = int((r["dt"] - t0).total_seconds() // 3600)
            wind = max(15.0, wind_now + (float(r["vmax"]) - v0))
            forecast.append({"time": f"T+{hours}h", "wind": round(wind, 1)})
        return {
            "track": track,
            "forecast": forecast,
            "analog_storm": str(best["storm_id"]),
            "analog_distance_deg": round(float(dist.min()), 2),
        }


def persistence_outlook(lat, lon, wind_now):
    """Fallback when no analog exists: storm stays where it is."""
    return {
        "track": [[lat, lon]] * 2,
        "forecast": [{"time": f"T+{3 * i}h", "wind": round(wind_now, 1)}
                     for i in range(HORIZON_STEPS + 1)],
        "analog_storm": None,
        "analog_distance_deg": None,
    }


# --------------------------------------------------------- build response ---
def build_response(*, class_index, category, confidence, probs, lat, lon,
                   wind_input, pressure_input, mode, source, tracks,
                   exclude_storms=(), past_track=None, warnings=None,
                   extra_meta=None, sst=None, wind_shear=None,
                   humidity=None, vorticity=None, environment_source="input",
                   sst_land=False):
    """Assemble the JSON the frontend expects.

    mode   : "image" (wind is an estimate from the class) or
             "rule-based" (class derived from the wind the user typed)
    source : who produced category/confidence: "model", "placeholder" or "rule"
    """
    warnings = list(warnings or [])
    if mode == "image":
        wind = WIND_MID[class_index]
        wind_display = WIND_DISPLAY[class_index].replace("up to 33", "\u226433")
        pressure = estimate_pressure(wind)
        wind_src = "estimate"
        pressure_src = "simulated"
    else:
        wind = float(wind_input)
        wind_display = f"{wind:.0f}"
        pressure = float(pressure_input)
        wind_src = "input"
        pressure_src = "input"

    outlook = tracks.analog_outlook(class_index, lat, lon, wind, exclude_storms) \
        or persistence_outlook(lat, lon, wind)
    if outlook["analog_storm"] is None:
        warnings.append("No matching historical storm found; outlook is a flat placeholder.")
    winds = [f["wind"] for f in outlook["forecast"]]

    # SST/wind-shear(/humidity) rule-based environment check (see
    # environment_favors_intensification docstring). This is separate from and does
    # not change the analog-based rapid_intensification flag above - the two are
    # surfaced independently so neither is misrepresented as informing the other.
    env_verdict = environment_favors_intensification(
        sst, wind_shear, humidity, sst_land=sst_land)
    if env_verdict == "not_applicable":
        warnings.append(
            "Over land: SST/rapid-intensification check not applicable (ERA5 SST is "
            "ocean-only - no sea surface here to check). Wind shear and humidity "
            "above are still real and shown for context (meteorological rule of "
            "thumb, not the AI model or the analog outlook)."
        )
    elif env_verdict is not None:
        humidity_note = f", humidity={humidity:.0f}%" if humidity is not None else ""
        warnings.append(
            f"Rule-based environment check: SST={sst:.1f}C, wind shear={wind_shear:.1f}kt"
            f"{humidity_note} -> conditions look {env_verdict} for intensification "
            f"(meteorological rule of thumb, not the AI model or the analog outlook)."
        )
    if vorticity is not None:
        warnings.append(
            f"850hPa relative vorticity: {vorticity:.2e} s⁻¹ (context only - higher "
            f"values are historically associated with cyclogenesis, but this app does not "
            f"use it in the verdict above)."
        )

    return {
        "prediction": {
            "category": category,
            "class_index": class_index,
            "confidence": None if confidence is None else round(float(confidence), 1),
            "wind": round(wind, 1),
            "wind_display": wind_display,
            "pressure": round(pressure, 1),
            "risk_index": risk_index(class_index, wind),
            "rapid_intensification": rapid_intensification(winds),
            "environment_favorability": env_verdict,
        },
        "probabilities": probs,
        "track": outlook["track"],
        "past_track": past_track or [],
        "forecast": outlook["forecast"],
        "sources": {
            "category": source,
            "confidence": source if confidence is not None else "n/a",
            "wind": wind_src,
            "pressure": pressure_src,
            "risk": "simulated",
            "track": "simulated",
            "forecast": "simulated",
            "rapid": "simulated",
            "environment_favorability": "rule-based" if env_verdict is not None else "n/a",
        },
        "meta": {
            "mode": mode,
            "position": [lat, lon],
            "analog_storm": outlook["analog_storm"],
            "analog_distance_deg": outlook["analog_distance_deg"],
            "warnings": warnings,
            "environment_input_source": environment_source if env_verdict is not None else "n/a",
            **(extra_meta or {}),
        },
    }
