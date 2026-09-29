"""North Indian Ocean region crop for a Himawari-8/9 full-disk image.

HONESTY NOTE (read this before trusting the crop):
NICT's real-time full-disk image is a rendered quicklook meant for public
display. It does not ship per-pixel geolocation metadata. To crop it to a
lat/lon box we compute where each corner of the box would project to on the
satellite's own view, using the standard geostationary ("fixed grid")
projection - the same maths behind Cartopy's Geostationary CRS / the
`+proj=geos` projection and the CGMS/EUMETSAT LRIT-HRIT reference formulas.
This is legitimate, published spherical-geometry, not a guess. What IS an
assumption (because it could not be confirmed from this environment - see
satellite/README.md) is that NICT's frame exactly frames the Earth limb to
limb with no extra margin. That assumption should be sanity-checked once
against a real downloaded image (see satellite/test_connection.py, step 3:
it saves the cropped PNG so you can eyeball that the Indian coastline falls
inside the crop box before trusting it for anything else).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# WGS84 ellipsoid, used by the standard geostationary projection formula.
_EARTH_EQUATORIAL_RADIUS_M = 6378137.0
_EARTH_POLAR_RADIUS_M = 6356752.3
# Himawari-8/9 nominal sub-satellite longitude and orbital radius (distance
# from Earth's centre), per JMA's published orbital parameters.
HIMAWARI_SUBLON_DEG = 140.7
_SAT_DISTANCE_FROM_CENTER_M = 42164000.0


@dataclass
class PixelBox:
    left: int
    top: int
    right: int
    bottom: int

    def valid(self) -> bool:
        return self.right > self.left and self.bottom > self.top


def _geodetic_to_geocentric_lat(lat_deg: float) -> float:
    ratio = (_EARTH_POLAR_RADIUS_M ** 2) / (_EARTH_EQUATORIAL_RADIUS_M ** 2)
    return math.atan(ratio * math.tan(math.radians(lat_deg)))


def _local_earth_radius(geocentric_lat_rad: float) -> float:
    a, b = _EARTH_EQUATORIAL_RADIUS_M, _EARTH_POLAR_RADIUS_M
    flattening_term = 1.0 - (b ** 2) / (a ** 2)
    return b / math.sqrt(1.0 - flattening_term * math.cos(geocentric_lat_rad) ** 2)


def latlon_to_scan_angles(lat_deg: float, lon_deg: float, sublon_deg: float = HIMAWARI_SUBLON_DEG):
    """Standard forward geostationary projection: geographic lat/lon ->
    the satellite's own (x, y) scan angles in radians. Returns None if the
    point is on the far side of the Earth (not visible to the satellite).

    Visibility test: let P be the target point's position vector (from
    Earth's centre) and S the satellite's position vector. P is visible
    from S only if the local outward normal at P has a non-negative dot
    product with the point-to-satellite vector, i.e. the satellite is
    above P's local horizon. Working this out for a geostationary
    satellite on the equatorial plane reduces to the closed form below
    (cos(geocentric lat) * cos(longitude difference) >= local_radius / H).
    An earlier version of this function checked the wrong quantity (r1,
    which stays positive everywhere and so never rejected far-side
    points) - this was caught by backend/satellite/test_offline.py.
    """
    phi_c = _geodetic_to_geocentric_lat(lat_deg)
    r_local = _local_earth_radius(phi_c)
    e = math.radians(lon_deg - sublon_deg)
    cos_e = math.cos(e)
    cos_phi_c = math.cos(phi_c)

    if cos_phi_c * cos_e < (r_local / _SAT_DISTANCE_FROM_CENTER_M):
        return None  # point is beyond the visible horizon from this satellite

    r1 = _SAT_DISTANCE_FROM_CENTER_M - r_local * cos_phi_c * cos_e
    r2 = -r_local * cos_phi_c * math.sin(e)
    r3 = r_local * math.sin(phi_c)
    rn = math.sqrt(r1 * r1 + r2 * r2 + r3 * r3)

    x = math.atan2(-r2, r1)
    y = math.asin(max(-1.0, min(1.0, -r3 / rn)))
    return x, y


def full_disk_limb_angle() -> float:
    """Half-angle (radians) from the satellite to the Earth's visible limb.
    Used to normalise scan angles to the [-1, 1] range that we assume maps
    to the edges of NICT's full-disk image (the one unverified assumption
    documented in the module docstring)."""
    return math.asin(_EARTH_EQUATORIAL_RADIUS_M / _SAT_DISTANCE_FROM_CENTER_M)


@dataclass
class CoverageResult:
    box: "PixelBox | None"
    requested_points: int
    visible_points: int

    @property
    def fraction_visible(self) -> float:
        return 0.0 if self.requested_points == 0 else self.visible_points / self.requested_points

    @property
    def fully_visible(self) -> bool:
        return self.requested_points > 0 and self.visible_points == self.requested_points


def latlon_box_to_pixel_box(lat_min, lat_max, lon_min, lon_max, img_width, img_height,
                              sublon_deg: float = HIMAWARI_SUBLON_DEG, samples_per_edge: int = 25):
    """Project a lat/lon bounding box into full-disk image pixel space.

    A geostationary satellite may only see PART of a requested box - for
    example Himawari-8/9 at 140.7E cannot see the western Arabian Sea at
    all (it is beyond the visible horizon; verified analytically and by
    backend/satellite/test_offline.py). Rather than silently sampling just
    the 4 corners (which can badly misjudge a partially-visible box - this
    is exactly what an earlier version of this function got wrong), this
    samples densely along the box's edges, keeps only the points the
    satellite can actually see, and returns the bounding pixel box of that
    VISIBLE SUBSET, plus how much of the requested box that subset covers.
    Returns a CoverageResult; result.box is None if nothing was visible.
    """
    limb = full_disk_limb_angle()

    def edge(lat0, lat1, lon0, lon1, n):
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            yield lat0 + (lat1 - lat0) * t, lon0 + (lon1 - lon0) * t

    points = []
    points += list(edge(lat_min, lat_max, lon_min, lon_min, samples_per_edge))  # west edge
    points += list(edge(lat_min, lat_max, lon_max, lon_max, samples_per_edge))  # east edge
    points += list(edge(lat_min, lat_min, lon_min, lon_max, samples_per_edge))  # south edge
    points += list(edge(lat_max, lat_max, lon_min, lon_max, samples_per_edge))  # north edge
    # A few interior points too, in case an odd projection makes the visible
    # region's extreme x/y fall inside rather than on the boundary.
    for lat_t in (0.25, 0.5, 0.75):
        for lon_t in (0.25, 0.5, 0.75):
            points.append((lat_min + (lat_max - lat_min) * lat_t,
                           lon_min + (lon_max - lon_min) * lon_t))

    xs, ys, visible = [], [], 0
    for lat, lon in points:
        angles = latlon_to_scan_angles(lat, lon, sublon_deg)
        if angles is None:
            continue
        visible += 1
        x_scan, y_scan = angles
        x_frac = 0.5 + (x_scan / limb) * 0.5
        y_frac = 0.5 - (y_scan / limb) * 0.5  # image y grows downward
        xs.append(x_frac * img_width)
        ys.append(y_frac * img_height)

    if not xs or not ys:
        return CoverageResult(None, len(points), 0)

    left = max(0, int(math.floor(min(xs))))
    right = min(img_width, int(math.ceil(max(xs))))
    top = max(0, int(math.floor(min(ys))))
    bottom = min(img_height, int(math.ceil(max(ys))))
    box = PixelBox(left, top, right, bottom)
    return CoverageResult(box if box.valid() else None, len(points), visible)


def crop_north_indian_ocean(image, lat_min, lat_max, lon_min, lon_max, sublon_deg=HIMAWARI_SUBLON_DEG):
    """image: a PIL.Image of a geostationary full disk. Returns
    (cropped_image, CoverageResult). cropped_image is None if none of the
    requested box is visible from this satellite. Check
    CoverageResult.fully_visible / fraction_visible to know whether you got
    the whole requested region or only part of it (e.g. Himawari cannot see
    the western Arabian Sea - see the module docstring)."""
    coverage = latlon_box_to_pixel_box(lat_min, lat_max, lon_min, lon_max,
                                        image.width, image.height, sublon_deg)
    if coverage.box is None:
        return None, coverage
    box = coverage.box
    return image.crop((box.left, box.top, box.right, box.bottom)), coverage
