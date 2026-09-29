"""One-off helper: build the offline coastline used by the Analytics map.

Source : Natural Earth 'land' 1:50m (public domain), via the npm package
         world-atlas (land-50m.json).  Convert it to GeoJSON first:
             npm install world-atlas topojson-client
             node tools/topo2geo.js            -> land50.geojson
Then run:   python tools/make_basemap.py land50.geojson

Output  : inputs/basemap.json  ({bbox, scale, path}) -- an SVG path of the
          land polygons clipped to the North Indian Ocean window.  This is
          NOT needed to run the dashboard; inputs/basemap.json is committed.
"""
import json
import sys
from pathlib import Path

LON0, LON1, LAT0, LAT1 = 44.0, 106.0, 0.0, 32.0   # window shown on the map
SCALE = 12                                        # SVG units per degree


def clip(poly, inside, intersect):
    """One Sutherland-Hodgman pass."""
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        a, b = inside(cur), inside(prev)
        if a:
            if not b:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif b:
            out.append(intersect(prev, cur))
    return out


def clip_bbox(ring):
    def edge(axis, value, keep_greater):
        def inside(p):
            return p[axis] >= value if keep_greater else p[axis] <= value

        def intersect(p, q):
            t = (value - p[axis]) / (q[axis] - p[axis])
            r = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
            r[axis] = value
            return r
        return inside, intersect

    for axis, value, greater in ((0, LON0, True), (0, LON1, False),
                                 (1, LAT0, True), (1, LAT1, False)):
        if not ring:
            break
        ring = clip(ring, *edge(axis, value, greater))
    return ring


def to_path(ring):
    pts = [((lon - LON0) * SCALE, (LAT1 - lat) * SCALE) for lon, lat in ring]
    if len(pts) < 3:
        return ""
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"


def main(src):
    gj = json.loads(Path(src).read_text())
    paths = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                if max(lons) < LON0 or min(lons) > LON1 or max(lats) < LAT0 or min(lats) > LAT1:
                    continue
                d = to_path(clip_bbox([list(p) for p in ring]))
                if d:
                    paths.append(d)
    out = {
        "source": "Natural Earth land 1:50m (public domain) via npm world-atlas 2.0.2",
        "lon0": LON0, "lon1": LON1, "lat0": LAT0, "lat1": LAT1, "scale": SCALE,
        "path": " ".join(paths),
    }
    dest = Path(__file__).resolve().parents[1] / "inputs" / "basemap.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", dest, f"{dest.stat().st_size / 1024:.0f} KB,", len(paths), "polygons")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "land50.geojson")
