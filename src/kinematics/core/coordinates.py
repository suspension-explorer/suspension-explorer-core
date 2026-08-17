"""Scalar suspension coordinates shared by sweeps and local responses.

A coordinate owns its geometry, identity, measurement, analytical gradient,
and presentation metadata.  A :class:`CoordinateTarget` adds only a value and
interpretation mode.  Giving a coordinate varying target values drives a
nonlinear sweep; materialising its current value through
:mod:`kinematics.core.holds` fixes the same quantity either throughout a sweep
or in a local tangent problem.  This keeps *what is measured* independent from
*when its reference value is captured*.

The concrete coordinate classes make invalid combinations unrepresentable:
point and actuator positions always have a direction, element lengths always
have exactly two endpoints, and arm angles always carry their hinge geometry.

The signed arm angle is the generalized travel coordinate for a rigid arm
rotating about a fixed chassis hinge.  For hinge-axis unit vector ``a`` and
projected radial vector ``r``, its exact local gradient is

``d(theta)/dP = (a x r) / (r . r)``.

This row is evaluated analytically by the common residual/Jacobian machinery.
No finite displacement, numerical perturbation, or chassis-axis proxy is
involved.  The signed angle is measured from the design radial direction with
``atan2``; the sign is immaterial to a zero-rate hold but remains deterministic
for diagnostics and future reuse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from kinematics.core.enums import Axis, Scope, TargetValueMode, Units
from kinematics.core.jacobians import jac_distance
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import (
    PointKey,
    PointRef,
    Side,
    point_key_name,
)
from kinematics.core.primitives.soft_math import softnorm
from kinematics.core.primitives.vector_utils.generic import project_coordinate

if TYPE_CHECKING:
    from kinematics.core.targeting import SweepConfig


class CoordinateType(StrEnum):
    """Supported scalar-coordinate types."""

    POINT = "point"
    ACTUATOR_POSITION = "actuator_position"
    ELEMENT_LENGTH = "element_length"
    ARM_ANGLE = "arm_angle"


@dataclass(slots=True, frozen=True)
class CoordinateAxis:
    """A projected coordinate direction along one chassis axis."""

    axis: Axis


@dataclass(slots=True, frozen=True)
class CoordinateVector:
    """A projected coordinate direction along an arbitrary unit vector."""

    vector: Direction3


type CoordinateDirection = CoordinateAxis | CoordinateVector


def _frozen_unit_axis(values: tuple[float, float, float]) -> np.ndarray:
    result = np.array(values, dtype=np.float64)
    result.flags.writeable = False
    return result


class ChassisAxisSystem:
    """Immutable chassis-space unit directions."""

    X: Final[Direction3] = Direction3.from_trusted(_frozen_unit_axis((1.0, 0.0, 0.0)))
    Y: Final[Direction3] = Direction3.from_trusted(_frozen_unit_axis((0.0, 1.0, 0.0)))
    Z: Final[Direction3] = Direction3.from_trusted(_frozen_unit_axis((0.0, 0.0, 1.0)))


def resolve_direction(direction: CoordinateDirection) -> Direction3:
    """Resolve a coordinate direction to a chassis-space unit vector."""
    if isinstance(direction, CoordinateVector):
        return direction.vector
    if direction.axis is Axis.X:
        return ChassisAxisSystem.X
    if direction.axis is Axis.Y:
        return ChassisAxisSystem.Y
    if direction.axis is Axis.Z:
        return ChassisAxisSystem.Z
    raise ValueError(f"Unsupported axis: {direction.axis!r}")


def _axis_name(direction: CoordinateDirection) -> str | None:
    if isinstance(direction, CoordinateAxis):
        return direction.axis.name.lower()
    return None


def _projected_identity(
    coordinate_type: CoordinateType,
    identifier: object,
    direction: CoordinateDirection,
) -> tuple[object, ...]:
    resolved = resolve_direction(direction)
    return (
        coordinate_type.value,
        identifier,
        *(float(value) for value in resolved.data),
    )


def _point_id(point: PointKey) -> str:
    physical_point = point.point if isinstance(point, PointRef) else point
    return physical_point.name.lower()


def _point_side(point: PointKey) -> Side | None:
    if isinstance(point, PointRef) and point.side is not Side.CENTER:
        return point.side
    return None


class _Coordinate:
    """Shared value-binding operations for concrete scalar coordinates."""

    @property
    def coordinate_description(self: ScalarCoordinate) -> str:
        name = self.type.value.replace("_", "-")
        return f"{name} coordinate '{self.id}'"

    @property
    def selector_point(self) -> PointKey | None:
        return None

    @property
    def driven_points(self) -> tuple[PointKey, ...]:
        return ()

    @property
    def parameter_point(self) -> str | None:
        return None

    @property
    def parameter_axis(self) -> str | None:
        return None

    @property
    def parameter_actuator(self) -> str | None:
        return None

    @property
    def parameter_element(self) -> str | None:
        return None

    @property
    def export_id(self: ScalarCoordinate) -> str:
        return self.id

    def target(
        self: ScalarCoordinate,
        value: float,
        mode: TargetValueMode = TargetValueMode.RELATIVE,
    ) -> CoordinateTarget:
        return CoordinateTarget(self, value, mode)

    def current_value_target(
        self: ScalarCoordinate,
        positions: Mapping[PointKey, Point3],
    ) -> CoordinateTarget:
        return self.target(self.measure(positions), TargetValueMode.ABSOLUTE)


@dataclass(frozen=True)
class PointCoordinate(_Coordinate):
    """Position of one physical point projected along one direction."""

    point: PointKey
    direction: CoordinateDirection
    scope: Scope
    side: Side | None = None
    unit: str = Units.MILLIMETERS.symbol

    def __post_init__(self) -> None:
        if self.side is None:
            object.__setattr__(self, "side", _point_side(self.point))

    @property
    def type(self) -> CoordinateType:
        return CoordinateType.POINT

    @property
    def id(self) -> str:
        suffix = _axis_name(self.direction) or "projection"
        return f"{_point_id(self.point)}_{suffix}"

    @property
    def label(self) -> str:
        return self.id.replace("_", " ").title()

    @property
    def point_keys(self) -> tuple[PointKey]:
        return (self.point,)

    @property
    def required_points(self) -> tuple[PointKey]:
        return self.point_keys

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return _projected_identity(self.type, self.point, self.direction)

    @property
    def selector_point(self) -> PointKey:
        return self.point

    @property
    def driven_points(self) -> tuple[PointKey]:
        return self.point_keys

    @property
    def parameter_point(self) -> str:
        return point_key_name(self.point)

    @property
    def parameter_axis(self) -> str | None:
        return _axis_name(self.direction)

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        return project_coordinate(
            positions[self.point], resolve_direction(self.direction)
        )

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        _ = positions
        return ((self.point, resolve_direction(self.direction).data),)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> Self:
        return replace(
            self,
            point=mapping(self.point),
            side=self.side if side is None else side,
        )


@dataclass(frozen=True, kw_only=True)
class ActuatorCoordinate(_Coordinate):
    """Named actuator position measured at its canonical physical point."""

    id: str
    label: str
    unit: str
    point_keys: tuple[PointKey, ...]
    direction: CoordinateDirection
    scope: Scope
    side: Side | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Actuator coordinate ID must not be empty")
        if not self.point_keys:
            raise ValueError("Actuator coordinates require a physical point")

    @property
    def type(self) -> CoordinateType:
        return CoordinateType.ACTUATOR_POSITION

    @property
    def required_points(self) -> tuple[PointKey]:
        return (self.point_keys[0],)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return _projected_identity(self.type, self.id, self.direction)

    @property
    def selector_point(self) -> PointKey:
        return self.point_keys[0]

    @property
    def driven_points(self) -> tuple[PointKey]:
        return self.required_points

    @property
    def parameter_axis(self) -> str | None:
        return _axis_name(self.direction)

    @property
    def parameter_actuator(self) -> str:
        return self.id

    def with_direction(self, direction: CoordinateDirection) -> Self:
        return replace(self, direction=direction)

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        point = self.point_keys[0]
        return project_coordinate(positions[point], resolve_direction(self.direction))

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        _ = positions
        return ((self.point_keys[0], resolve_direction(self.direction).data),)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> Self:
        return replace(
            self,
            point_keys=tuple(mapping(point) for point in self.point_keys),
            side=self.side if side is None else side,
        )


@dataclass(frozen=True, kw_only=True)
class ElementLengthCoordinate(_Coordinate):
    """True pin-centre length between two declared element endpoints."""

    id: str
    label: str
    unit: str
    point_a: PointKey
    point_b: PointKey
    scope: Scope
    side: Side | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Element coordinate ID must not be empty")

    @property
    def type(self) -> CoordinateType:
        return CoordinateType.ELEMENT_LENGTH

    @property
    def point_keys(self) -> tuple[PointKey, PointKey]:
        return self.point_a, self.point_b

    @property
    def required_points(self) -> tuple[PointKey, PointKey]:
        return self.point_keys

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return (self.type.value, self.id, *self.point_keys)

    @property
    def parameter_element(self) -> str:
        return self.id

    @property
    def export_id(self) -> str:
        return self.id if self.id.endswith("_length") else f"{self.id}_length"

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        delta = positions[self.point_b] - positions[self.point_a]
        return float(softnorm(delta.squared_norm()))

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        derivatives = jac_distance(
            positions[self.point_a].data,
            positions[self.point_b].data,
        )
        return (
            (self.point_a, derivatives[:3]),
            (self.point_b, derivatives[3:]),
        )

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> Self:
        return replace(
            self,
            point_a=mapping(self.point_a),
            point_b=mapping(self.point_b),
            side=self.side if side is None else side,
        )


def _frozen_vector(values: np.ndarray) -> np.ndarray:
    """Return one validated immutable three-vector copy."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Arm geometry vectors must be finite three-vectors")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ArmAngleCoordinate(_Coordinate):
    """Topology-owned signed angle of a carried point about a fixed hinge."""

    id: str
    label: str
    hinge_point_a: PointKey
    hinge_point_b: PointKey
    carried_point: PointKey
    axis_origin: np.ndarray
    axis_direction: np.ndarray
    reference_radial: np.ndarray
    scope: Scope
    side: Side | None = None
    unit: str = "rad"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Arm-angle coordinate ID must not be empty")
        origin = _frozen_vector(self.axis_origin)
        axis = _frozen_vector(self.axis_direction)
        reference = _frozen_vector(self.reference_radial)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= EPS_GEOMETRIC:
            raise ValueError(f"Arm-angle coordinate '{self.id}' has a zero axis")
        axis = _frozen_vector(axis / axis_norm)
        reference = reference - axis * float(np.dot(axis, reference))
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= EPS_GEOMETRIC:
            raise ValueError(
                f"Arm-angle coordinate '{self.id}' has no radial lever arm"
            )
        object.__setattr__(self, "axis_origin", origin)
        object.__setattr__(self, "axis_direction", axis)
        object.__setattr__(
            self, "reference_radial", _frozen_vector(reference / reference_norm)
        )

    @classmethod
    def from_positions(
        cls,
        *,
        id: str,
        label: str,
        hinge_point_a: PointKey,
        hinge_point_b: PointKey,
        carried_point: PointKey,
        positions: Mapping[PointKey, Point3],
        scope: Scope,
        side: Side | None = None,
    ) -> Self:
        """Capture a fixed hinge and deterministic zero angle from design state."""
        origin = positions[hinge_point_a].data
        return cls(
            id=id,
            label=label,
            hinge_point_a=hinge_point_a,
            hinge_point_b=hinge_point_b,
            carried_point=carried_point,
            axis_origin=origin,
            axis_direction=positions[hinge_point_b].data - origin,
            reference_radial=positions[carried_point].data - origin,
            scope=scope,
            side=side,
        )

    @property
    def type(self) -> CoordinateType:
        return CoordinateType.ARM_ANGLE

    @property
    def point_keys(self) -> tuple[PointKey, PointKey, PointKey]:
        return self.hinge_point_a, self.hinge_point_b, self.carried_point

    @property
    def required_points(self) -> tuple[PointKey]:
        # The hinge axis is captured design geometry; only the carried point moves.
        return (self.carried_point,)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return (self.type.value, self.id, *self.point_keys)

    def _radial(self, positions: Mapping[PointKey, Point3]) -> np.ndarray:
        offset = positions[self.carried_point].data - self.axis_origin
        return offset - self.axis_direction * float(np.dot(self.axis_direction, offset))

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Measure signed rotation from the captured design radial direction."""
        radial = self._radial(positions)
        radial_squared = float(np.dot(radial, radial))
        if radial_squared <= EPS_GEOMETRIC**2:
            raise ValueError(
                f"Arm-angle coordinate '{self.id}' has no radial lever arm"
            )
        quadrature = np.cross(self.axis_direction, self.reference_radial)
        cosine_component = float(np.dot(self.reference_radial, radial))
        sine_component = float(np.dot(quadrature, radial))
        return float(np.arctan2(sine_component, cosine_component))

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        """Return the exact local angular gradient at the carried point."""
        radial = self._radial(positions)
        radial_squared = float(np.dot(radial, radial))
        if radial_squared <= EPS_GEOMETRIC**2:
            raise ValueError(
                f"Arm-angle coordinate '{self.id}' has no radial lever arm"
            )
        gradient = np.cross(self.axis_direction, radial) / radial_squared
        return ((self.carried_point, gradient),)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> Self:
        """Remap structural keys while retaining captured physical geometry."""
        return replace(
            self,
            hinge_point_a=mapping(self.hinge_point_a),
            hinge_point_b=mapping(self.hinge_point_b),
            carried_point=mapping(self.carried_point),
            side=self.side if side is None else side,
        )


type ScalarCoordinate = (
    PointCoordinate | ActuatorCoordinate | ElementLengthCoordinate | ArmAngleCoordinate
)


@dataclass(frozen=True)
class CoordinateTarget:
    """One coordinate paired with a requested scalar value and mode."""

    coordinate: ScalarCoordinate
    value: float
    mode: TargetValueMode = TargetValueMode.RELATIVE

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(
                f"Target for coordinate '{self.coordinate.id}' must be finite, "
                f"got {self.value}."
            )
        if (
            isinstance(self.coordinate, ElementLengthCoordinate)
            and self.mode is TargetValueMode.ABSOLUTE
            and self.value < EPS_GEOMETRIC
        ):
            raise ValueError(
                f"Absolute element length for '{self.coordinate.id}' must be at "
                f"least {EPS_GEOMETRIC:g} {self.coordinate.unit}, got {self.value}."
            )

    def with_value(self, value: float, mode: TargetValueMode) -> Self:
        return replace(self, value=value, mode=mode)

    def map_points(self, mapping: Callable[[PointKey], PointKey]) -> Self:
        return replace(self, coordinate=self.coordinate.map_points(mapping))


def target_side(target: CoordinateTarget) -> Side | None:
    """Return the structural side of a resolved target, if any."""
    return target.coordinate.side


def target_export_column_name(target: CoordinateTarget) -> str:
    """Return the deterministic measured-coordinate export column name."""
    side = target_side(target)
    side_suffix = f"_{side.name.lower()}" if side is not None else ""
    return f"target_{target.coordinate.export_id}{side_suffix}"


def actuator_coordinate_matches(
    required: ActuatorCoordinate,
    candidate: CoordinateTarget | ScalarCoordinate,
) -> bool:
    """Whether a target or hold controls one required actuator coordinate."""
    coordinate = (
        candidate.coordinate if isinstance(candidate, CoordinateTarget) else candidate
    )
    if not isinstance(coordinate, ActuatorCoordinate):
        return False
    if coordinate.id != required.id:
        return False
    alignment = abs(
        float(
            np.dot(
                resolve_direction(coordinate.direction).data,
                resolve_direction(required.direction).data,
            )
        )
    )
    return alignment >= 1.0 - EPS_GEOMETRIC


def validate_sweep_controls(
    sweep_config: SweepConfig,
    actuator_coordinates: tuple[ActuatorCoordinate, ...],
) -> None:
    """Require exactly one target for every required actuator coordinate."""
    for step_index in range(sweep_config.n_steps):
        step_targets = [
            *(target_sweep[step_index] for target_sweep in sweep_config.target_sweeps),
            *sweep_config.hold.coordinates,
        ]
        for actuator in actuator_coordinates:
            matches = [
                target
                for target in step_targets
                if actuator_coordinate_matches(actuator, target)
            ]
            if len(matches) != 1:
                name = (
                    "steering rack" if actuator.id == "rack" else actuator.label.lower()
                )
                raise ValueError(
                    f"Sweep requires exactly one target for actuator '{name}' along "
                    f"its motion axis; found {len(matches)} at step {step_index}."
                )
