from typing import Any

import earthkit.data
import numpy as np


def ekd_example(param: str, step: int) -> Any:
    """Ad-hoc single-field earthkit data object on a tiny regular lat-lon grid, values derived from param and step."""
    lats, lons = np.meshgrid(np.linspace(90, -90, 7), np.linspace(0, 350, 12), indexing="ij")
    seed = sum(ord(c) for c in param) + step
    values = np.sin(np.radians(lons) + seed) * np.cos(np.radians(lats))
    metadata = {
        "param": param,
        "step": step,
        "latitudes": lats.flatten(),
        "longitudes": lons.flatten(),
    }
    return earthkit.data.from_source("list-of-dicts", [{**metadata, "values": values.flatten()}])
