import importlib.metadata

import numpy as np
from cascade.low.func import Either
from earthkit.workflows.fluent import Action, create_task_instance, from_source
from earthkit.workflows.metadata import NodeMetadata, Requirements
from fiab_core.fable import (
    ActionLookup,
    BlockInstanceOutput,
    ConfigurationOptionRestriction,
    QubedOutput,
)
from fiab_core.plugin import Error
from fiab_core.tools.blocks import BlockInstanceRich, Source
from qubed import Qube

RUNTIME_DIST = "fiab-plugin-experiments-runtime"
RUNTIME_MODULE = "fiab_plugin_experiments_runtime"

PARAMS = ["2t", "msl"]
STEPS = [0, 6, 12]


def runtime_environment() -> list[str]:
    """Worker environment for the runtime wheel. It is not a dependency of this wheel, so it is requested by name.

    If the runtime happens to be installed locally as an editable (dev venv), the worker gets that source tree instead.
    """
    try:
        from fiab_core.tools.plugins import _detect_editable_install

        return [_detect_editable_install(RUNTIME_DIST)]
    except importlib.metadata.PackageNotFoundError:
        return [RUNTIME_DIST]


class EkdExample(Source):
    title: str = "Earthkit Data Example"
    description: str = "Returns an ad-hoc earthkit data object, one field per param and step"
    configuration_options = {}
    inputs: list[str] = []

    def validate(
        self, block: BlockInstanceRich, inputs: dict[str, QubedOutput], restrictions: ConfigurationOptionRestriction
    ) -> BlockInstanceOutput:
        return QubedOutput(dataqube=Qube.from_datacube({"param": PARAMS, "step": STEPS}), datatype="earthkit.data.FieldList")

    def compile(self, inputs: ActionLookup, block: BlockInstanceRich) -> Either[Action, Error]:  # ty:ignore[invalid-type-arguments] # semigroup
        tasks = np.asarray(
            [
                [
                    create_task_instance(f"{RUNTIME_MODULE}.source.ekd_example", static_input_kw={"param": param, "step": step})
                    for param in PARAMS
                ]
                for step in STEPS
            ]
        )
        action = from_source(
            tasks,
            dims=["step", "param"],
            coords={"step": STEPS, "param": PARAMS},
            node_metadata=NodeMetadata(requirements=Requirements(environment=runtime_environment())),
        )
        return Either.ok(action)
