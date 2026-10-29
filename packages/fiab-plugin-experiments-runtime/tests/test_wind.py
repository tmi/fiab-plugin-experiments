import pathlib

import earthkit.data
import numpy as np
import pytest

from fiab_plugin_experiments_runtime.source import ekd_example
from fiab_plugin_experiments_runtime.wind import project_wind, wind_point_direction


def _value(result: object) -> float:
    assert len(result) == 1  # ty:ignore[invalid-argument-type]
    return float(result[0].values[0])  # ty:ignore[not-subscriptable]


def _uniform(u: float, v: float) -> object:
    lats, lons = np.meshgrid(np.linspace(90, -90, 7), np.linspace(0, 350, 12), indexing="ij")
    base = {"latitudes": lats.flatten(), "longitudes": lons.flatten()}
    return earthkit.data.from_source(
        "list-of-dicts",
        [
            {**base, "param": "u", "values": np.full(lats.size, u)},
            {**base, "param": "v", "values": np.full(lats.size, v)},
        ],
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [(180.0, 3.0), (0.0, -3.0), (90.0, 0.0), (270.0, 0.0)],
)
def test_southerly_wind(direction: float, expected: float) -> None:
    # v=3 means blowing towards the north: hits south-facing surfaces
    assert _value(wind_point_direction(_uniform(0.0, 3.0), 12.3, 45.6, direction)) == pytest.approx(expected, abs=1e-9)


def test_diagonal() -> None:
    # wind from the west (u=4) hits east-facing... no, west-facing surface
    assert project_wind(4.0, 0.0, 270.0) == pytest.approx(4.0)
    assert _value(wind_point_direction(_uniform(2.0, 2.0), -80.0, 359.0, 225.0)) == pytest.approx(2 * np.sqrt(2))


def test_exact_gridpoint_and_example() -> None:
    fields = ekd_example("u", 0)
    both = fields + ekd_example("v", 0)
    result = wind_point_direction(both, 30.0, 0.0, 90.0)
    assert np.isfinite(_value(result))


def test_zarr_sink_compatible(tmp_path: pathlib.Path) -> None:
    # as the fiab zarr sink does: several steps concatenated into one field list, written via to_target
    fields = [wind_point_direction(ekd_example("u", s) + ekd_example("v", s), 30.0, 10.0, 90.0) for s in (0, 6, 12)]
    merged = fields[0] + fields[1] + fields[2]
    merged.to_target("zarr", xarray_to_zarr_kwargs={"store": str(tmp_path / "out.zarr"), "mode": "w"})
    assert (tmp_path / "out.zarr").exists()
