"""Validated, transport-independent sweep specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from kinematics.core.enums import Axis, PointID, TargetValueMode
from kinematics.core.primitives.geometry import Direction3, extract_array
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.decoding import (
    AxisValue,
    PointIDValue,
    SideValue,
    TargetValueModeValue,
)
from kinematics.core.targeting import (
    ACTUATOR_POSITION_TARGET_IDS,
    ELEMENT_LENGTH_TARGET_IDS,
    ChassisAxisSystem,
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    ScalarTarget,
    SweepConfig,
    TargetKind,
    validate_sweep_controls,
)

if TYPE_CHECKING:
    from kinematics.core.suspensions.base import Suspension

AXIS_VECTORS: dict[Axis, np.ndarray] = {
    Axis.X: ChassisAxisSystem.X.data,
    Axis.Y: ChassisAxisSystem.Y.data,
    Axis.Z: ChassisAxisSystem.Z.data,
}


def vector_to_axis(vector: np.ndarray) -> Axis | None:
    """Return the principal axis represented by a vector, if any."""
    vector_data = extract_array(vector)
    for axis, axis_vector in AXIS_VECTORS.items():
        if np.allclose(vector_data, axis_vector):
            return axis
    return None


class DirectionSpec(BaseModel):
    """Target direction specified by either an axis or a custom vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: AxisValue | None = None
    vector: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_exactly_one(self) -> "DirectionSpec":
        if (self.axis is None) == (self.vector is None):
            raise ValueError("Specify exactly one of 'axis' or 'vector'")
        return self

    @field_serializer("axis")
    def serialize_axis(self, axis: Axis | None) -> str | None:
        """Write principal axes using their canonical lowercase names."""
        return axis.name.lower() if axis is not None else None

    def to_unit_vector(self) -> np.ndarray:
        """Convert this specification to a normalized three-dimensional vector."""
        if self.axis is not None:
            return AXIS_VECTORS[self.axis]
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(f"Vector must be 3D, got shape {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero")
        return vector / norm


class SweepValueSpec(BaseModel):
    """Value/range fields shared by every scalar sweep target specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    side: SideValue | None = None
    mode: TargetValueModeValue = TargetValueMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_side(self) -> "SweepValueSpec":
        if self.side == Side.CENTER:
            raise ValueError("Sweep target side must be 'left' or 'right'.")
        return self

    def expand_values(self, default_steps: int | None) -> list[float]:
        """Expand explicit values or a start-stop range exactly once."""
        if self.values is not None:
            return [float(value) for value in self.values]
        target_name = self.name or self.coordinate_name
        if self.start is None or self.stop is None:
            raise ValueError(
                f"Target '{target_name}': must specify either 'values' or both "
                "'start' and 'stop'"
            )
        if default_steps is None:
            raise ValueError(
                f"Target '{target_name}': no 'steps' count available "
                "(specify at target or file level)"
            )
        return list(np.linspace(float(self.start), float(self.stop), default_steps))

    @property
    def coordinate_name(self) -> str:
        """Return the coordinate identifier used in validation errors."""
        return type(self).__name__


class TargetSpec(SweepValueSpec):
    """One point-coordinate target dimension in a suspension sweep."""

    type: Literal["point"]
    point: PointIDValue
    direction: DirectionSpec

    @property
    def coordinate_name(self) -> str:
        return self.point.name.lower()

    @model_validator(mode="after")
    def check_point(self) -> "TargetSpec":
        if self.point is PointID.NOT_ASSIGNED:
            raise ValueError(
                "Point target ID 'not_assigned' is reserved and cannot be used"
            )
        return self

    @field_serializer("point")
    def serialize_point(self, point: PointIDValue) -> str:
        """Write point identifiers using their canonical wire names."""
        return point.name.lower()


class ElementLengthTargetSpec(SweepValueSpec):
    """One stable topology-owned element-length target dimension."""

    type: Literal["element_length"]
    element: str

    @property
    def coordinate_name(self) -> str:
        return self.element

    @model_validator(mode="after")
    def check_element(self) -> "ElementLengthTargetSpec":
        if not self.element.strip():
            raise ValueError("Element target ID must not be empty")
        if self.element not in ELEMENT_LENGTH_TARGET_IDS:
            expected = ", ".join(ELEMENT_LENGTH_TARGET_IDS)
            raise ValueError(
                f"Unknown element-length target ID '{self.element}'. "
                f"Expected one of: {expected}."
            )
        return self


class ActuatorPositionTargetSpec(SweepValueSpec):
    """One topology-owned actuator position along an authored direction."""

    type: Literal["actuator_position"]
    actuator: str
    direction: DirectionSpec

    @property
    def coordinate_name(self) -> str:
        return self.actuator

    @model_validator(mode="after")
    def check_actuator(self) -> "ActuatorPositionTargetSpec":
        if self.actuator not in ACTUATOR_POSITION_TARGET_IDS:
            expected = ", ".join(ACTUATOR_POSITION_TARGET_IDS)
            raise ValueError(
                f"Unknown actuator-position target ID '{self.actuator}'. "
                f"Expected one of: {expected}."
            )
        return self


SweepTargetSpec = Annotated[
    TargetSpec | ActuatorPositionTargetSpec | ElementLengthTargetSpec,
    Field(discriminator="type"),
]


class SteeringProbeAnalysisSpec(BaseModel):
    """Analysis-only selection of a topology-declared steering response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    isolation: str = "layout_default"

    @model_validator(mode="after")
    def check_isolation(self) -> "SteeringProbeAnalysisSpec":
        if not self.isolation.strip():
            raise ValueError("Steering-probe isolation ID must not be empty")
        return self


class SweepAnalysisSpec(BaseModel):
    """Optional calculations requested alongside the authored state sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steering_probe: SteeringProbeAnalysisSpec = Field(
        default_factory=SteeringProbeAnalysisSpec
    )


class SweepSpec(BaseModel):
    """Validated sweep file or API specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    steps: int | None = Field(default=None, ge=1)
    targets: list[SweepTargetSpec] = Field(min_length=1)
    analysis: SweepAnalysisSpec = Field(default_factory=SweepAnalysisSpec)

    @model_validator(mode="after")
    def check_version(self) -> "SweepSpec":
        if self.version != 1:
            raise ValueError(f"Unsupported sweep version: {self.version}")
        return self

    @property
    def n_steps(self) -> int:
        """Return the validated number of values in each target dimension."""
        lengths = {len(target.expand_values(self.steps)) for target in self.targets}
        if len(lengths) > 1:
            raise ValueError(
                f"All targets must have the same length, got: {sorted(lengths)}"
            )
        return next(iter(lengths), 0)


def build_sweep_config(
    spec: SweepSpec,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """Expand a validated sweep and resolve optional side-qualified targets."""
    target_sequences = [target.expand_values(spec.steps) for target in spec.targets]
    lengths = {len(sequence) for sequence in target_sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"All targets must have the same length, got: {sorted(lengths)}"
        )

    dimensions: list[list[ScalarTarget]] = []
    for target_index, (target_spec, values) in enumerate(
        zip(spec.targets, target_sequences)
    ):
        try:
            if isinstance(target_spec, ElementLengthTargetSpec):
                if suspension is None:
                    raise ValueError(
                        f"Element target '{target_spec.element}' requires a "
                        "suspension context to resolve its endpoints."
                    )
                coordinate = suspension.resolve_drive_coordinate(
                    target_spec.element,
                    target_spec.side,
                    TargetKind.ELEMENT_LENGTH,
                )
                dimensions.append(
                    [coordinate.target(value, target_spec.mode) for value in values]
                )
                continue

            unit_vector = target_spec.direction.to_unit_vector()
            axis = vector_to_axis(unit_vector)
            direction: PointTargetAxis | PointTargetVector
            if axis is not None:
                direction = PointTargetAxis(axis)
            else:
                direction = PointTargetVector(Direction3(unit_vector))

            if isinstance(target_spec, ActuatorPositionTargetSpec):
                if suspension is None:
                    raise ValueError(
                        f"Actuator target '{target_spec.actuator}' requires a "
                        "suspension context to resolve its physical coordinate."
                    )
                coordinate = suspension.resolve_drive_coordinate(
                    target_spec.actuator,
                    target_spec.side,
                    TargetKind.ACTUATOR_POSITION,
                )
                dimensions.append(
                    [
                        coordinate.position_target(
                            direction,
                            value,
                            target_spec.mode,
                        )
                        for value in values
                    ]
                )
                continue

            if suspension is not None:
                point_key = suspension.resolve_target_key(
                    target_spec.point, target_spec.side
                )
            else:
                if target_spec.side is not None:
                    raise ValueError(
                        f"Sweep target for '{target_spec.point.name}' specifies a "
                        "'side', which requires a suspension context to resolve."
                    )
                point_key = target_spec.point

            dimensions.append(
                [
                    PointTarget(
                        point_id=point_key,
                        direction=direction,
                        value=value,
                        mode=target_spec.mode,
                    )
                    for value in values
                ]
            )
        except ValueError as error:
            raise ValueError(f"Sweep target {target_index}: {error}") from error
    sweep_config = SweepConfig(
        dimensions,
        steering_probe_isolation=spec.analysis.steering_probe.isolation,
    )
    if suspension is not None:
        suspension.validate_sweep_targets(
            target for dimension in dimensions for target in dimension
        )
        validate_sweep_controls(sweep_config, suspension.actuator_dofs())
        # Selection is analysis-only: validate it after the state-driving
        # target basis is built, without adding a residual to the sweep solve.
        suspension.resolve_steering_probe(sweep_config.steering_probe_isolation)
    return sweep_config
