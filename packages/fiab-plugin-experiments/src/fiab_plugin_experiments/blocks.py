import importlib.metadata

import numpy as np
from cascade.low.func import Either
from earthkit.workflows.fluent import Action, create_task_instance, from_source
from earthkit.workflows.metadata import NodeMetadata, Requirements
from fiab_core.fable import (
    ActionLookup,
    BlockConfigurationOption,
    BlockInstanceOutput,
    ConfigurationOptionId,
    ConfigurationOptionRestriction,
    QubedOutput,
)
from fiab_core.plugin import Error
from fiab_core.tools.blocks import BlockInstanceConfigurationError, BlockInstanceRich, Product, Source
from fiab_core.types import FloatType
from qubed import Qube

RUNTIME_DIST = "fiab-plugin-experiments-runtime"
RUNTIME_MODULE = "fiab_plugin_experiments_runtime"

PARAMS = ["u", "v"]
STEPS = [0, 6, 12]


PARAM = "param"
LAT_AXES = ("latitude", "lat")
LON_AXES = ("longitude", "lon")
# (u, v) param names, in order of preference: pressure-level/generic, 10m names, GRIB paramIds
WIND_PARAM_PAIRS = [("u", "v"), ("10u", "10v"), ("131", "132"), ("165", "166")]
WIND_OUTPUT_PARAM = "wind_point_direction"

LATITUDE = ConfigurationOptionId("latitude")
LONGITUDE = ConfigurationOptionId("longitude")
DIRECTION = ConfigurationOptionId("direction")


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
    description: str = "Returns an ad-hoc earthkit data object, one field per param (u and v wind components) and step"
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


def find_wind_pair(params: set[str] | list[str]) -> tuple[str, str] | None:
    """The (u, v) param names present in the given params, if any"""
    available = {str(p) for p in params}
    for u, v in WIND_PARAM_PAIRS:
        if u in available and v in available:
            return u, v
    return None


def _with_axis(qube: Qube, key: str, values: list[str]) -> Qube:
    value_group = Qube.from_datacube({key: values}).children[0].values
    return Qube.make_root([Qube.make_node(key, value_group, qube.children)])


class WindPointDirection(Product):
    title: str = "Wind Speed at Point/Direction"
    description: str = (
        "Interpolates the u and v wind components to a location and projects them on a direction: the result is the speed of "
        "the wind blowing against a surface (eg window screen) at that location which faces that direction. Positive means the "
        "wind pushes onto the surface, negative that it blows away from it (suction)."
    )
    configuration_options: dict[ConfigurationOptionId, BlockConfigurationOption] = {
        LATITUDE: BlockConfigurationOption(title="Latitude", description="Latitude of the location, degrees north", value_type=FloatType()),
        LONGITUDE: BlockConfigurationOption(
            title="Longitude", description="Longitude of the location, degrees east", value_type=FloatType()
        ),
        DIRECTION: BlockConfigurationOption(
            title="Facing direction",
            description="Compass bearing in degrees that the surface faces (outward normal): 0 north, 90 east, 180 south, 270 west",
            value_type=FloatType(),
        ),
    }
    inputs: list[str] = ["dataset"]

    def _check_config(self, block: BlockInstanceRich) -> None:
        lat = block.config_as_float(LATITUDE)
        lon = block.config_as_float(LONGITUDE)
        block.config_as_float(DIRECTION)
        if not -90.0 <= lat <= 90.0:
            raise BlockInstanceConfigurationError(f"latitude must be within [-90, 90], got {lat}")
        if not np.isfinite(lon):
            raise BlockInstanceConfigurationError(f"longitude must be finite, got {lon}")

    def validate(
        self, block: BlockInstanceRich, inputs: dict[str, QubedOutput], restrictions: ConfigurationOptionRestriction
    ) -> BlockInstanceOutput:
        dataset = inputs.get("dataset")
        if not isinstance(dataset, QubedOutput):
            raise BlockInstanceConfigurationError(f"Unsupported input type for 'dataset': {type(dataset).__name__}")
        self._check_config(block)
        params = dataset.dataqube.axes().get(PARAM)
        if params is None or find_wind_pair(set(params)) is None:
            raise ValueError(f"input must contain wind components, ie, one of {WIND_PARAM_PAIRS} on the {PARAM} axis")
        # NOTE we select just the wind pair, and then collapse the params and the (optional) horizontal axes
        pair = find_wind_pair(set(params))
        qube = dataset.dataqube.select({PARAM: list(pair or [])})
        spatial = [k for k in qube.axes() if k in LAT_AXES + LON_AXES]
        if spatial:
            qube = qube.remove_by_key(spatial)
        qube = _with_axis(qube.remove_by_key([PARAM]), PARAM, [WIND_OUTPUT_PARAM])
        return QubedOutput(dataqube=qube, datatype="earthkit.data.FieldList")

    def compile(self, inputs: ActionLookup, block: BlockInstanceRich) -> Either[Action, Error]:  # ty:ignore[invalid-type-arguments] # semigroup
        self._check_config(block)
        action = inputs[block.input_ids["dataset"]]
        pair = find_wind_pair(set(action.nodes.coords[PARAM].values.tolist()) if PARAM in action.nodes.coords else set())
        if pair is None:
            return Either.error(f"input must contain wind components, ie, one of {WIND_PARAM_PAIRS} on the {PARAM} axis")
        u, v = pair
        result = (
            action.select({PARAM: [u, v]}, expand=True)
            .concatenate(dim=PARAM)
            .map(
                create_task_instance(
                    f"{RUNTIME_MODULE}.wind.wind_point_direction",
                    static_input_kw={
                        "latitude": block.config_as_float(LATITUDE),
                        "longitude": block.config_as_float(LONGITUDE),
                        "direction": block.config_as_float(DIRECTION),
                        "u_param": u,
                        "v_param": v,
                    },
                ),
                node_metadata=NodeMetadata(requirements=Requirements(environment=runtime_environment())),
            )
        )
        return Either.ok(result)

    def intersect(self, other: QubedOutput) -> bool:
        params = other.dataqube.axes().get(PARAM)
        return params is not None and find_wind_pair(set(params)) is not None
