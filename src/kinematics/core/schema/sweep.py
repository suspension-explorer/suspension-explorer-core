"""Validated, transport-independent sweep specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)

from kinematics.core.coordinates import (
    ActuatorCoordinate,
    ChassisAxisSystem,
    CoordinateAxis,
    CoordinateDirection,
    CoordinateTarget,
    CoordinateType,
    CoordinateVector,
    PointCoordinate,
    validate_sweep_controls,
)
from kinematics.core.enums import Axis, PointID, Scope, TargetValueMode
from kinematics.core.holds import CoordinateHold
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
    SweepConfig,
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
    hold: bool = False
    mode: TargetValueModeValue = TargetValueMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_side(self) -> "SweepValueSpec":
        if self.side == Side.CENTER:
            raise ValueError("Sweep target side must be 'left' or 'right'.")
        return self

    @model_validator(mode="after")
    def check_hold_values(self) -> "SweepValueSpec":
        """A held coordinate has no authored value schedule."""
        if not self.hold:
            return self
        conflicting = self.model_fields_set.intersection(
            {"mode", "start", "stop", "values"}
        )
        if conflicting:
            rendered = ", ".join(sorted(conflicting))
            raise ValueError(
                f"Held coordinate '{self.coordinate_name}' cannot specify {rendered}."
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_hold_without_schedule(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Keep explicit holds round-trip safe despite ordinary value defaults."""
        data = dict(handler(self))
        if self.hold:
            for field_name in ("mode", "start", "stop", "values"):
                data.pop(field_name, None)
        return data

    def expand_values(self, default_steps: int | None) -> list[float]:
        """Expand explicit values or a start-stop range exactly once."""
        if self.hold:
            raise ValueError(
                f"Held coordinate '{self.coordinate_name}' has no sweep values"
            )
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


class VirtualSteeringAnalysisSpec(BaseModel):
    """Analysis-only configuration for the virtual steering response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suspension_hold: str = "layout_default"

    @model_validator(mode="after")
    def check_suspension_hold(self) -> "VirtualSteeringAnalysisSpec":
        if not self.suspension_hold.strip():
            raise ValueError("Suspension-hold ID must not be empty")
        return self


class SweepAnalysisSpec(BaseModel):
    """Optional calculations requested alongside the authored state sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    virtual_steering: VirtualSteeringAnalysisSpec = Field(
        default_factory=VirtualSteeringAnalysisSpec
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

    @model_validator(mode="after")
    def check_swept_target(self) -> "SweepSpec":
        if not any(not target.hold for target in self.targets):
            raise ValueError("A sweep requires at least one non-held target.")
        return self

    @property
    def n_steps(self) -> int:
        """Return the validated number of values in each target dimension."""
        sequences = _expanded_swept_values(self)
        return len(sequences[0]) if sequences else 0


def _expanded_swept_values(spec: SweepSpec) -> list[list[float]]:
    """Expand swept targets once and validate their common length."""
    sequences = [
        target.expand_values(spec.steps) for target in spec.targets if not target.hold
    ]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"All targets must have the same length, got: {sorted(lengths)}"
        )
    return sequences


def build_sweep_config(
    spec: SweepSpec,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """Expand a validated sweep and resolve optional side-qualified targets."""
    target_sequences = _expanded_swept_values(spec)
    if not target_sequences:
        raise ValueError("A sweep requires at least one non-held target.")

    dimensions: list[list[CoordinateTarget]] = []
    held_coordinates = []
    swept_values = iter(target_sequences)
    for target_index, target_spec in enumerate(spec.targets):
        values = [] if target_spec.hold else next(swept_values)
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
                    CoordinateType.ELEMENT_LENGTH,
                )
                if target_spec.hold:
                    held_coordinates.append(coordinate)
                    continue
                dimensions.append(
                    [coordinate.target(value, target_spec.mode) for value in values]
                )
                continue

            unit_vector = target_spec.direction.to_unit_vector()
            axis = vector_to_axis(unit_vector)
            direction: CoordinateDirection
            if axis is not None:
                direction = CoordinateAxis(axis)
            else:
                direction = CoordinateVector(Direction3(unit_vector))

            if isinstance(target_spec, ActuatorPositionTargetSpec):
                if suspension is None:
                    raise ValueError(
                        f"Actuator target '{target_spec.actuator}' requires a "
                        "suspension context to resolve its physical coordinate."
                    )
                coordinate = suspension.resolve_drive_coordinate(
                    target_spec.actuator,
                    target_spec.side,
                    CoordinateType.ACTUATOR_POSITION,
                )
                if not isinstance(coordinate, ActuatorCoordinate):
                    raise TypeError("Resolved actuator coordinate has the wrong type")
                coordinate = coordinate.with_direction(direction)
                if target_spec.hold:
                    held_coordinates.append(coordinate)
                    continue
                dimensions.append(
                    [coordinate.target(value, target_spec.mode) for value in values]
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

            point_coordinate = PointCoordinate(
                point=point_key,
                direction=direction,
                scope=(
                    Scope.AXLE
                    if suspension is not None
                    and suspension.is_axle
                    and target_spec.side is None
                    else Scope.CORNER
                ),
            )
            if target_spec.hold:
                held_coordinates.append(point_coordinate)
                continue
            dimensions.append(
                [point_coordinate.target(value, target_spec.mode) for value in values]
            )
        except ValueError as error:
            raise ValueError(f"Sweep target {target_index}: {error}") from error
    sweep_config = SweepConfig(
        dimensions,
        hold=CoordinateHold(tuple(held_coordinates)),
        suspension_hold_id=spec.analysis.virtual_steering.suspension_hold,
    )
    if suspension is not None:
        suspension.validate_sweep_targets(
            (target for dimension in dimensions for target in dimension),
            held_targets=sweep_config.hold.materialize(
                suspension.initial_state().positions
            ),
        )
        validate_sweep_controls(
            sweep_config,
            suspension.required_actuator_coordinates(),
        )
        # Selection is analysis-only: validate it after the state-driving
        # target basis is built, without adding a residual to the sweep solve.
        suspension.resolve_suspension_hold(sweep_config.suspension_hold_id)
    return sweep_config
