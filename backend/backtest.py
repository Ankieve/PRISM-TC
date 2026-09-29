"""
Backtest mode for the analog track/wind outlook (backend/logic.py TrackDB).

What this tests and what it does NOT test
------------------------------------------
This does NOT test the AI image classifier (predict.py) - there is no image
involved here at all. It tests the historical-analog method in
logic.TrackDB.analog_outlook(): "given a class + position + wind, find the
closest matching historical storm and copy its next 24h forward." Because
backend/data/track_data.csv already contains real historical positions and
winds, we can pick a real storm's starting point, ask analog_outlook() the
same question it is asked in production, and compare the answer against what
that storm actually did next - since we already have the real answer on file.

Honesty rule enforced here: the storm being tested is always excluded from
its own candidate pool (exclude=[storm_id]) in the analog search, otherwise
the method could trivially "predict" a storm's future by matching itself in
the historical database, which would not be a fair test and would silently
inflate the backtest's apparent accuracy.
"""

import math

import logic

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def list_backtestable_storms(tracks):
    """Storms that have at least one point with a full 24h (HORIZON_STEPS x 3h)
    of real recorded future positions - i.e. storms analog_outlook() can
    actually be tested against. Uses each storm's earliest such point."""
    df = tracks.df
    cand = df[df["k"] + logic.HORIZON_STEPS < df["n"]]
    out = []
    for storm_id, group in cand.groupby("storm_id"):
        row = group.sort_values("k").iloc[0]
        out.append({
            "storm_id": str(storm_id),
            "start_time": row["dt"].strftime("%Y-%m-%d %H:%M UTC"),
            "category": row["category"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "vmax_kt": float(row["vmax"]),
        })
    out.sort(key=lambda r: r["storm_id"])
    return out


def run_backtest(tracks, storm_id):
    """Backtest analog_outlook() against one storm's earliest valid starting
    point. Returns a dict with the prediction, the real recorded future, and
    error metrics - or {"error": "..."} if the storm can't be backtested."""
    df = tracks.df
    storm_id = str(storm_id)
    rows = df[(df["storm_id"].astype(str) == storm_id)
              & (df["k"] + logic.HORIZON_STEPS < df["n"])].sort_values("k")
    if rows.empty:
        return {"error": f"Storm '{storm_id}' has no starting point with a full "
                          f"24h of recorded future track - cannot backtest it."}
    start = rows.iloc[0]
    class_index = logic.CLASS_ORDER.index(start["category"])
    lat0, lon0, wind0 = float(start["lat"]), float(start["lon"]), float(start["vmax"])

    predicted = tracks.analog_outlook(class_index, lat0, lon0, wind0, exclude=[storm_id])
    if predicted is None:
        return {"error": f"No other historical storm could serve as an analog for "
                          f"'{storm_id}' (with itself excluded) - cannot backtest it."}

    actual_seg = df[(df["storm_id"].astype(str) == storm_id)
                     & (df["k"] >= start["k"])
                     & (df["k"] <= start["k"] + logic.HORIZON_STEPS)].sort_values("k")
    actual_track = [[float(a), float(b)] for a, b in zip(actual_seg["lat"], actual_seg["lon"])]
    actual_winds = [float(v) for v in actual_seg["vmax"]]
    t0 = actual_seg["dt"].iloc[0]
    actual_forecast = [
        {"time": f"T+{int((r['dt'] - t0).total_seconds() // 3600)}h", "wind": round(float(r["vmax"]), 1)}
        for _, r in actual_seg.iterrows()
    ]

    n = min(len(predicted["track"]), len(actual_track))
    track_errors_km = [
        round(haversine_km(predicted["track"][i][0], predicted["track"][i][1],
                            actual_track[i][0], actual_track[i][1]), 1)
        for i in range(n)
    ]
    m = min(len(predicted["forecast"]), len(actual_winds))
    wind_errors_kt = [
        round(abs(predicted["forecast"][i]["wind"] - actual_winds[i]), 1)
        for i in range(m)
    ]

    return {
        "storm_id": storm_id,
        "start_time": start["dt"].strftime("%Y-%m-%d %H:%M UTC"),
        "true_category": start["category"],
        "analog_storm_used": predicted["analog_storm"],
        "analog_distance_deg": predicted["analog_distance_deg"],
        "predicted_track": predicted["track"],
        "actual_track": actual_track,
        "predicted_forecast": predicted["forecast"],
        "actual_forecast": actual_forecast,
        "track_error_km": {
            "by_step": track_errors_km,
            "mean": round(sum(track_errors_km) / len(track_errors_km), 1) if track_errors_km else None,
            "max": max(track_errors_km) if track_errors_km else None,
        },
        "wind_error_kt": {
            "by_step": wind_errors_kt,
            "mean": round(sum(wind_errors_kt) / len(wind_errors_kt), 1) if wind_errors_kt else None,
            "max": max(wind_errors_kt) if wind_errors_kt else None,
        },
        "note": "The tested storm is excluded from its own candidate pool during the "
                "analog search, so this is not the method 'finding itself'.",
    }
