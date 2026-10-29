from fiab_core.fable import BlockFactoryId, BlockInstance, QubedOutput

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
