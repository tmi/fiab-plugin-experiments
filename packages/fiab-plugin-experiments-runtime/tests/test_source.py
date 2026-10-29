from fiab_plugin_experiments_runtime.source import ekd_example


def test_ekd_example() -> None:
    result = ekd_example("u", 6)
    assert len(result) == 1
    assert result[0].values.shape == (7 * 12,)
