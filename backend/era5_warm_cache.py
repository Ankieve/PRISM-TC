"""Warm the repo-bundled ERA5 seed cache for the demo storms.

Render's disk is ephemeral: anything fetched at runtime vanishes on restart.
This script does ONE real CDS fetch per demo point (needs your CDS account)
and stores each result as backend/data/era5/seed/<lat>_<lon>.json, which is
committed to git. The deployed app serves those seeds with zero live CDS
requests (see era5._read_seed) - the demo can never OOM on ERA5 again.

Needs:  pip install -r backend/requirements-era5.txt
        CDSAPI_URL + CDSAPI_KEY env vars (or ~/.cdsapirc locally)

Run:    python backend/era5_warm_cache.py            # all samples.csv points
        python backend/era5_warm_cache.py --lat 15.5 --lon 85.0
"""
import csv
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import era5  # noqa: E402


def warm(lat: float, lon: float) -> bool:
    print(f"fetching live CDS data for lat={lat} lon={lon} ...")
    try:
        result = era5.fetch_environment(lat, lon, use_cache=False, allow_live=True)
    except (era5.ERA5NotConfigured, era5.ERA5Error) as exc:
        print(f"  FAILED: {exc}")
        return False
    era5.ensure_dirs()
    path = era5.SEED_DIR / era5._seed_name(lat, lon)
    seed = {k: v for k, v in result.items() if k != "cached"}
    path.write_text(json.dumps(seed, indent=1))
    print(f"  ok: sst={result['sst_c']}C shear={result['wind_shear_kt']}kt "
          f"humidity={result['humidity_pct']}% valid={result.get('valid_time_utc')} "
          f"-> {path}")
    return True


def main() -> int:
    if "--lat" in sys.argv and "--lon" in sys.argv:
        points = [(float(sys.argv[sys.argv.index("--lat") + 1]),
                   float(sys.argv[sys.argv.index("--lon") + 1]))]
    else:
        points = []
        with open(BACKEND_DIR / "samples" / "samples.csv") as f:
            for row in csv.DictReader(f):
                points.append((float(row["lat"]), float(row["lon"])))
    ok = sum(warm(la, lo) for la, lo in points)
    print(f"\nw armed {ok}/{len(points)} seed files in {era5.SEED_DIR}")
    print("Commit them: git add backend/data/era5/seed/")
    return 0 if ok == len(points) else 1


if __name__ == "__main__":
    sys.exit(main())
