from typing import Any

from fiab_core.fable import BlockFactoryId, BlockInstance, BlockInstanceOutput, ConfigurationOptionId, QubedOutput
from qubed import Qube

from fiab_plugin_experiments import plugin


def test_validate_is_qubed() -> None:
    p = plugin()
    result = p.validator(BlockFactoryId("ekdExample"), BlockInstance(configuration_values={}), {})
    output = result.result.get_or_raise()
    assert isinstance(output, QubedOutput)
    assert set(output.dataqube.axes()) == {"param", "step"}


def test_compile() -> None:
    p = plugin()
    result = p.compiler({}, BlockFactoryId("ekdExample"), BlockInstance(configuration_values={}))
    action = result.get_or_raise()
    assert len(action.nodes) > 0


def _ekd_output() -> QubedOutput:
    p = plugin()
    return p.validator(BlockFactoryId("ekdExample"), BlockInstance(configuration_values={}), {}).result.get_or_raise()  # ty:ignore[invalid-return-type]


WIND_CONFIG: dict[ConfigurationOptionId, Any] = {
    ConfigurationOptionId("latitude"): 10.0,
    ConfigurationOptionId("longitude"): 20.0,
    ConfigurationOptionId("direction"): 180.0,
}


def test_wind_expand() -> None:
    p = plugin()
    assert [e.factory for e in p.expander(_ekd_output())] == [BlockFactoryId("windPointDirection")]
    no_wind = QubedOutput(dataqube=Qube.from_datacube({"param": ["2t"], "step": [0]}))
    assert p.expander(no_wind) == []
    zarr_like = QubedOutput(
        dataqube=Qube.from_datacube({"param": ["u", "v"], "level": [500, 850], "latitude": [1.0, 2.0], "longitude": [3.0, 4.0]})
    )
    assert len(p.expander(zarr_like)) == 1


def test_wind_validate() -> None:
    p = plugin()
    inputs: dict[str, BlockInstanceOutput] = {"dataset": _ekd_output()}
    result = p.validator(BlockFactoryId("windPointDirection"), BlockInstance(configuration_values=WIND_CONFIG), inputs)
    output = result.result.get_or_raise()
    assert isinstance(output, QubedOutput)
    assert output.dataqube.axes() == {"step": {0, 6, 12}, "param": {"wind_point_direction"}}


def test_wind_validate_rejects() -> None:
    p = plugin()
    bad = QubedOutput(dataqube=Qube.from_datacube({"param": ["2t"], "step": [0]}))
    result = p.validator(BlockFactoryId("windPointDirection"), BlockInstance(configuration_values=WIND_CONFIG), {"dataset": bad})
    assert result.result.e is not None
    cfg = {**WIND_CONFIG, ConfigurationOptionId("latitude"): 100.0}
    result = p.validator(BlockFactoryId("windPointDirection"), BlockInstance(configuration_values=cfg), {"dataset": _ekd_output()})
    assert result.result.e is not None


def test_wind_compile() -> None:
    p = plugin()
    source = p.compiler({}, BlockFactoryId("ekdExample"), BlockInstance(configuration_values={})).get_or_raise()
    block = BlockInstance(configuration_values=WIND_CONFIG, input_ids={"dataset": "src"})
    action = p.compiler({"src": source}, BlockFactoryId("windPointDirection"), block).get_or_raise()  # ty:ignore[invalid-argument-type]
    assert len(action.nodes) > 0
