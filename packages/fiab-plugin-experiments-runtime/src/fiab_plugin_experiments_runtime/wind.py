from typing import Any

import numpy as np

EARTH_RADIUS_M = 6371000.0


def _haversine(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in meters between a point and arrays of points, all in degrees"""
    phi1, phi2 = np.radians(lat), np.radians(lats)
    dphi = phi2 - phi1
    dlam = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def interpolate_to_point(values: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float, count: int = 4) -> float:
    """Inverse-distance weighted mean of the `count` grid points closest to the location. Works on any (also unstructured) grid."""
    distance = _haversine(lat, lon, lats, lons)
    count = min(count, distance.size)
    nearest = np.argpartition(distance, count - 1)[:count]
    d = distance[nearest]
    if d.min() < 1.0:  # within a meter of a grid point
        return float(values[nearest[np.argmin(d)]])
    weights = 1.0 / d**2
    return float(np.sum(weights * values[nearest]) / np.sum(weights))


def project_wind(u: float, v: float, direction: float) -> float:
    """Component of wind (u east, v north, both in the direction the wind blows towards) that pushes onto a surface facing `direction`.

    `direction` is the compass bearing of the surface's outward normal (0 north, 90 east). Wind blowing against the surface, ie
    towards the opposite of the normal, is positive; wind blowing away from it is negative.
    """
    theta = np.radians(direction)
    return float(-(u * np.sin(theta) + v * np.cos(theta)))


def wind_point_direction(fields: Any, latitude: float, longitude: float, direction: float, u_param: str = "u", v_param: str = "v") -> float:
    """Wind speed against a surface at a location, from an earthkit data object containing the u and v components

    Each component is interpolated to the location (inverse-distance weighting over the four closest grid points),
    and then projected onto the direction -- see `project_wind`.
    """
    components = {}
    for name, param in (("u", u_param), ("v", v_param)):
        selected = fields.sel(param=param)
        if len(selected) != 1:
            raise ValueError(f"expected exactly one field for param {param}, got {len(selected)}")
        field = selected[0]
        lats, lons = _latlon(field)
        components[name] = interpolate_to_point(np.asarray(field.values).flatten(), lats, lons, latitude, longitude)
    return project_wind(components["u"], components["v"], direction)


def _latlon(field: Any) -> tuple[np.ndarray, np.ndarray]:
    ll = field.to_latlon(flatten=True)
    return np.asarray(ll["lat"]), np.asarray(ll["lon"])
