"""
Sweep target definitions and chassis-direction resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Literal, NamedTuple, Protocol, Self, TypedDict, Union

import numpy as np

from kinematics.core.enums import (
    ActuatorPositionCoordinateID,
    Axis,
    ElementLengthCoordinateID,
    PointID,
    Scope,
    TargetValueMode,
    Units,
)
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


def frozen_unit_axis(values: tuple[float, float, float]) -> np.ndarray:
    """
    Create a frozen (immutable) unit-axis array from the given values.

    Args:
        values: A tuple of three floats representing the axis direction.

    Returns:
        A numpy array with writeable flag set to False to prevent mutation.
    """
    # Build a non-writeable unit-axis array so the shared ChassisAxisSystem
    # constants cannot be mutated through their .data attribute.
    arr = np.array(values, dtype=np.float64)
    arr.flags.writeable = False
    return arr


class ChassisAxisSystem:
    """
    Chassis-space unit axis directions.

    Usage:
        ChassisAxisSystem.X  # -> Direction3 along [1, 0, 0]
        ChassisAxisSystem.Y  # -> Direction3 along [0, 1, 0]
        ChassisAxisSystem.Z  # -> Direction3 along [0, 0, 1]
    """

    X: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((1.0, 0.0, 0.0)))
    Y: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((0.0, 1.0, 0.0)))
    Z: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((0.0, 0.0, 1.0)))


@dataclass
class SweepConfig:
    """
    Configuration for a parametric sweep over multiple target dimensions.

    Each inner list represents one sweep dimension (e.g., bump travel, steering angle).
    All dimensions must have the same length - the sweep will iterate through
    corresponding indices across all dimensions simultaneously.

    Example:
        bump_targets = [PointTarget(..., value=-30), ..., PointTarget(..., value=30)]
        steer_targets = [PointTarget(..., value=-10), ..., PointTarget(..., value=10)]
        config = SweepConfig([bump_targets, steer_targets])
    """

    target_sweeps: Sequence[Sequence["ScalarTarget"]]
    steering_probe_isolation: str = "layout_default"

    def __post_init__(self):
        if not self.steering_probe_isolation.strip():
            raise ValueError("Steering-probe isolation ID must not be empty")
        if not self.target_sweeps:
            return

        lengths = [len(sweep) for sweep in self.target_sweeps]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All sweep dimensions must have the same length. Got: {lengths}"
            )

        for step_index in range(lengths[0]):
            targets = [dimension[step_index] for dimension in self.target_sweeps]
            indices_by_identity: dict[tuple[object, ...], list[int]] = {}
            for target_index, target in enumerate(targets):
                indices_by_identity.setdefault(target.coordinate_identity, []).append(
                    target_index
                )
            duplicate = next(
                (
                    (identity, indices)
                    for identity, indices in indices_by_identity.items()
                    if len(indices) > 1
                ),
                None,
            )
            if duplicate is not None:
                _identity, target_indices = duplicate
                description = targets[target_indices[0]].coordinate_description
                rendered_indices = ", ".join(str(index) for index in target_indices)
                raise ValueError(
                    f"Sweep targets {rendered_indices} drive the same {description} "
                    f"at step {step_index}."
                )

    @property
    def n_steps(self) -> int:
        """
        Number of steps in the sweep.
        """
        if not self.target_sweeps:
            return 0
        return len(self.target_sweeps[0])


class PointTarget(NamedTuple):
    """
    Defines a target constraint for a specific point during kinematic solving.

    The mode determines how the value is interpreted initially, but all targets
    are converted to absolute coordinates before solving begins.

    Attributes:
        point_id: The point to constrain
        direction: Direction along which to apply the target
        value: Target value (interpretation depends on mode)
        mode: Whether value is relative displacement or absolute coordinate
    """

    point_id: PointKey
    direction: "PointTargetDirection"
    value: float
    mode: TargetValueMode = TargetValueMode.RELATIVE

    @property
    def kind(self) -> "TargetKind":
        """Return the scalar-coordinate kind."""
        return TargetKind.POINT

    @property
    def required_points(self) -> tuple[PointKey, ...]:
        """Return the point required to evaluate this coordinate."""
        return (self.point_id,)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        """Return a stable identity independent of value and mode."""
        return _projected_coordinate_identity(self.kind, self.point_id, self.direction)

    @property
    def selector_point(self) -> PointKey:
        """Return the point used by point-coordinate derivative drivers."""
        return self.point_id

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Evaluate the projected scalar coordinate."""
        return _measure_projected_coordinate(self.point_id, self.direction, positions)

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        """Return analytical partials with respect to involved point positions."""
        return _projected_coordinate_partials(self.point_id, self.direction, positions)

    def with_value(
        self,
        value: float,
        mode: TargetValueMode,
    ) -> "PointTarget":
        """Return this coordinate with a new scalar value and mode."""
        return self._replace(value=value, mode=mode)

    def map_points(self, mapping: Callable[[PointKey], PointKey]) -> "PointTarget":
        """Return this target with its point key remapped."""
        return self._replace(point_id=mapping(self.point_id))

    @property
    def coordinate_id(self) -> str:
        """Return a deterministic presentation identifier."""
        suffix = _target_axis_name(self.direction) or "projection"
        point = (
            self.point_id.point
            if isinstance(self.point_id, PointRef)
            else self.point_id
        )
        return f"{point.name.lower()}_{suffix}"

    @property
    def label(self) -> str:
        """Return a concise display label."""
        return self.coordinate_id.replace("_", " ").title()

    @property
    def unit(self) -> str:
        return Units.MILLIMETERS.symbol

    @property
    def structural_side(self) -> Side | None:
        return _point_key_side(self.point_id)

    @property
    def parameter_point(self) -> str:
        return point_key_name(self.point_id)

    @property
    def parameter_axis(self) -> str | None:
        return _target_axis_name(self.direction)

    @property
    def parameter_actuator(self) -> None:
        return None

    @property
    def parameter_element(self) -> None:
        return None

    @property
    def driven_points(self) -> tuple[PointKey, ...]:
        return self.required_points

    @property
    def drive_coordinate_key(self) -> None:
        return None

    @property
    def actuator_coordinate_id(self) -> None:
        return None

    @property
    def actuator_direction(self) -> None:
        return None

    @property
    def export_coordinate_id(self) -> str:
        return self.coordinate_id

    @property
    def coordinate_description(self) -> str:
        return f"point coordinate '{self.coordinate_id}'"


@dataclass(slots=True, frozen=True)
class PointTargetAxis:
    """
    A target direction defined by one of the principal axes.

    Attributes:
        axis (Axis): The axis to use as the target direction.
    """

    axis: Axis


@dataclass(slots=True, frozen=True)
class PointTargetVector:
    """
    A target direction defined by an arbitrary vector.

    Attributes:
        vector (Direction3): The direction defining the target.
    """

    vector: Direction3


PointTargetDirection = Union[PointTargetAxis, PointTargetVector]


def _target_axis_name(direction: PointTargetDirection) -> str | None:
    if isinstance(direction, PointTargetAxis):
        return direction.axis.name.lower()
    return None


def _point_key_side(point: PointKey) -> Side | None:
    if isinstance(point, PointRef) and point.side is not Side.CENTER:
        return point.side
    return None


def _projected_coordinate_identity(
    kind: "TargetKind",
    identifier: object,
    direction: PointTargetDirection,
) -> tuple[object, ...]:
    resolved = resolve_target(direction)
    return (
        kind.value,
        identifier,
        *(float(value) for value in resolved.data),
    )


def _measure_projected_coordinate(
    point: PointKey,
    direction: PointTargetDirection,
    positions: Mapping[PointKey, Point3],
) -> float:
    return project_coordinate(positions[point], resolve_target(direction))


def _projected_coordinate_partials(
    point: PointKey,
    direction: PointTargetDirection,
    positions: Mapping[PointKey, Point3],
) -> tuple[tuple[PointKey, np.ndarray], ...]:
    _ = positions
    return ((point, resolve_target(direction).data),)


class ActuatorPositionTarget(NamedTuple):
    """One named actuator position measured at a canonical physical point."""

    actuator_id: str
    point_id: PointKey
    direction: PointTargetDirection
    value: float
    mode: TargetValueMode = TargetValueMode.RELATIVE
    label: str = "Actuator Position"

    @property
    def kind(self) -> "TargetKind":
        """Return the scalar-coordinate kind."""
        return TargetKind.ACTUATOR_POSITION

    @property
    def required_points(self) -> tuple[PointKey, ...]:
        """Return the canonical point used to measure actuator position."""
        return (self.point_id,)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        """Return a stable identity independent of value and mode."""
        return _projected_coordinate_identity(
            self.kind,
            self.actuator_id,
            self.direction,
        )

    @property
    def selector_point(self) -> PointKey:
        """Return the physical point used by derivative drivers."""
        return self.point_id

    @property
    def coordinate_id(self) -> str:
        """Return the stable topology-owned actuator identifier."""
        return self.actuator_id

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Evaluate actuator position along the authored direction."""
        return _measure_projected_coordinate(self.point_id, self.direction, positions)

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        """Return the coordinate partial at the canonical physical point."""
        return _projected_coordinate_partials(self.point_id, self.direction, positions)

    def with_value(
        self,
        value: float,
        mode: TargetValueMode,
    ) -> "ActuatorPositionTarget":
        """Return this coordinate with a new scalar value and mode."""
        return self._replace(value=value, mode=mode)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
    ) -> "ActuatorPositionTarget":
        """Return this actuator coordinate remapped to another point namespace."""
        return self._replace(point_id=mapping(self.point_id))

    @property
    def unit(self) -> str:
        return ActuatorPositionCoordinateID(self.actuator_id).unit

    @property
    def structural_side(self) -> None:
        return None

    @property
    def parameter_point(self) -> None:
        return None

    @property
    def parameter_axis(self) -> str | None:
        return _target_axis_name(self.direction)

    @property
    def parameter_actuator(self) -> str:
        return self.actuator_id

    @property
    def parameter_element(self) -> None:
        return None

    @property
    def driven_points(self) -> tuple[PointKey, ...]:
        return self.required_points

    @property
    def drive_coordinate_key(
        self,
    ) -> tuple[TargetKind, str, tuple[PointKey, ...] | None, None]:
        return (self.kind, self.actuator_id, None, None)

    @property
    def actuator_coordinate_id(self) -> str:
        return self.actuator_id

    @property
    def actuator_direction(self) -> Direction3:
        return resolve_target(self.direction)

    @property
    def export_coordinate_id(self) -> str:
        return self.coordinate_id

    @property
    def coordinate_description(self) -> str:
        return f"actuator-position coordinate '{self.coordinate_id}'"


class TargetKind(StrEnum):
    """Supported scalar sweep-coordinate kinds."""

    POINT = "point"
    ACTUATOR_POSITION = "actuator_position"
    ELEMENT_LENGTH = "element_length"
    HINGE_ANGLE = "hinge_angle"


# This is a syntactic wire vocabulary, not a promise that every point exists or
# can move in every suspension. Topology availability remains a build-time rule.
POINT_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    point.name.lower() for point in PointID if point is not PointID.NOT_ASSIGNED
)
ACTUATOR_POSITION_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    coordinate.value for coordinate in ActuatorPositionCoordinateID
)
ELEMENT_LENGTH_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    coordinate.value for coordinate in ElementLengthCoordinateID
)
SidePolicy = Literal["corner", "shared"]


def sweep_target_side_policy(kind: TargetKind, coordinate_id: str) -> SidePolicy:
    """Return static side ownership for one globally recognized coordinate."""
    try:
        policy = {
            TargetKind.POINT: lambda: PointID[coordinate_id.upper()].sweep_side_policy,
            TargetKind.ACTUATOR_POSITION: lambda: ActuatorPositionCoordinateID(
                coordinate_id
            ).side_policy,
            TargetKind.ELEMENT_LENGTH: lambda: ElementLengthCoordinateID(
                coordinate_id
            ).side_policy,
        }[kind]()
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Unknown {kind.value} coordinate ID '{coordinate_id}'."
        ) from error
    return policy.value


def resolve_published_target_side(
    description: str,
    candidate_sides: Sequence[Side | None],
    requested_side: Side | None,
) -> Side | None:
    """Resolve an explicit side against published coordinate ownership."""
    available_sides = tuple(dict.fromkeys(candidate_sides))
    if requested_side in available_sides:
        return requested_side

    if requested_side is None:
        names = " or ".join(
            side.name.lower() for side in available_sides if side is not None
        )
        raise ValueError(f"{description} requires side {names}.")

    if None in available_sides:
        raise ValueError(f"{description} is shared and does not accept a side.")

    names = ", ".join(side.name.lower() for side in available_sides if side is not None)
    raise ValueError(
        f"{description} is unavailable on side '{requested_side.name.lower()}'. "
        f"Available sides: {names}."
    )


class PointTargetVocabularyItem(TypedDict):
    """One JSON-native point-target vocabulary entry."""

    type: Literal["point"]
    id: str
    label: str
    featured: bool
    side_policy: SidePolicy


class ActuatorPositionVocabularyItem(TypedDict):
    """One JSON-native actuator-position vocabulary entry."""

    type: Literal["actuator_position"]
    id: str
    label: str
    featured: bool
    side_policy: SidePolicy


class ElementLengthVocabularyItem(TypedDict):
    """One JSON-native element-length vocabulary entry."""

    type: Literal["element_length"]
    id: str
    label: str
    unit: str
    featured: bool
    side_policy: SidePolicy


class SweepTargetVocabulary(TypedDict):
    """Geometry-independent identifiers accepted by sweep documents."""

    positions: list[PointTargetVocabularyItem | ActuatorPositionVocabularyItem]
    element_lengths: list[ElementLengthVocabularyItem]


def _point_target_label(point_id: str) -> str:
    """Humanize one canonical point ID for geometry-independent editors."""
    label = point_id.replace("_", " ").title()
    return (
        label.replace("Arb", "ARB").replace("U Bar", "U-Bar").replace("T Bar", "T-Bar")
    )


def sweep_target_vocabulary() -> SweepTargetVocabulary:
    """Return the stable, JSON-native sweep target vocabulary.

    Position entries distinguish physical point coordinates from named actuator
    coordinates. ``side_policy`` describes each coordinate's static ownership;
    a selected geometry still decides whether it is available, fixed,
    derived-output-only, or otherwise valid to drive. Element entries are
    globally known coordinate IDs; topology compatibility is likewise resolved
    when the sweep is built.
    """
    featured_point_order = ("wheel_center",)
    featured_point_ids = frozenset(featured_point_order)
    ordered_point_ids = (
        *featured_point_order,
        *(
            point_id
            for point_id in POINT_TARGET_IDS
            if point_id not in featured_point_ids
        ),
    )
    positions: list[PointTargetVocabularyItem | ActuatorPositionVocabularyItem] = [
        ActuatorPositionVocabularyItem(
            type="actuator_position",
            id=ActuatorPositionCoordinateID.RACK.value,
            label="Rack",
            featured=True,
            side_policy=sweep_target_side_policy(
                TargetKind.ACTUATOR_POSITION,
                ActuatorPositionCoordinateID.RACK.value,
            ),
        )
    ]
    positions.extend(
        PointTargetVocabularyItem(
            type="point",
            id=point_id,
            label=_point_target_label(point_id),
            featured=point_id in featured_point_ids,
            side_policy=sweep_target_side_policy(TargetKind.POINT, point_id),
        )
        for point_id in ordered_point_ids
    )
    return {
        "positions": positions,
        "element_lengths": [
            {
                "type": "element_length",
                "id": coordinate.value,
                "label": coordinate.label,
                "unit": coordinate.unit,
                "featured": coordinate is ElementLengthCoordinateID.DAMPER,
                "side_policy": sweep_target_side_policy(
                    TargetKind.ELEMENT_LENGTH,
                    coordinate.value,
                ),
            }
            for coordinate in ElementLengthCoordinateID
        ],
    }


@dataclass(frozen=True)
class ElementLengthTarget:
    """Drive the pin-centre length between two declared element endpoints."""

    element_id: str
    point_a: PointKey
    point_b: PointKey
    value: float
    mode: TargetValueMode = TargetValueMode.RELATIVE
    label: str = "Element Length"
    unit: str = Units.MILLIMETERS.symbol
    scope: Scope = Scope.CORNER
    side: Side | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("Element target ID must not be empty")
        if not np.isfinite(self.value):
            raise ValueError(
                f"Element-length target '{self.element_id}' must be finite, "
                f"got {self.value}."
            )
        if self.mode is TargetValueMode.ABSOLUTE and self.value < EPS_GEOMETRIC:
            raise ValueError(
                f"Absolute element length for '{self.element_id}' must be "
                f"at least {EPS_GEOMETRIC:g} {self.unit}, got {self.value}."
            )

    @property
    def kind(self) -> TargetKind:
        return TargetKind.ELEMENT_LENGTH

    @property
    def required_points(self) -> tuple[PointKey, ...]:
        return (self.point_a, self.point_b)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return (self.kind.value, self.element_id, self.point_a, self.point_b)

    @property
    def selector_point(self) -> None:
        """Element-length fields do not select point-coordinate drivers."""
        return None

    @property
    def coordinate_id(self) -> str:
        return self.element_id

    @property
    def structural_side(self) -> Side | None:
        return self.side

    @property
    def parameter_point(self) -> None:
        return None

    @property
    def parameter_axis(self) -> None:
        return None

    @property
    def parameter_actuator(self) -> None:
        return None

    @property
    def parameter_element(self) -> str:
        return self.element_id

    @property
    def driven_points(self) -> tuple[PointKey, ...]:
        return ()

    @property
    def drive_coordinate_key(
        self,
    ) -> tuple[TargetKind, str, tuple[PointKey, ...], Side | None]:
        return (self.kind, self.element_id, self.required_points, self.side)

    @property
    def actuator_coordinate_id(self) -> None:
        return None

    @property
    def actuator_direction(self) -> None:
        return None

    @property
    def export_coordinate_id(self) -> str:
        return (
            self.coordinate_id
            if self.coordinate_id.endswith("_length")
            else f"{self.coordinate_id}_length"
        )

    @property
    def coordinate_description(self) -> str:
        return f"element-length coordinate '{self.coordinate_id}'"

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Return true pin-centre length using the solver's soft-norm policy."""
        delta = positions[self.point_b] - positions[self.point_a]
        return float(softnorm(delta.squared_norm()))

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        """Return analytical length partials for both endpoints."""
        point_a = positions[self.point_a]
        point_b = positions[self.point_b]
        derivatives = jac_distance(point_a.data, point_b.data)
        return (
            (self.point_a, derivatives[:3]),
            (self.point_b, derivatives[3:]),
        )

    def with_value(
        self,
        value: float,
        mode: TargetValueMode,
    ) -> "ElementLengthTarget":
        return replace(self, value=value, mode=mode)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
    ) -> "ElementLengthTarget":
        return replace(
            self,
            point_a=mapping(self.point_a),
            point_b=mapping(self.point_b),
        )


type ScalarTarget = PointTarget | ActuatorPositionTarget | ElementLengthTarget


class ScalarCoordinateTarget(Protocol):
    """Structural interface consumed by residual and tangent calculations.

    Authored sweep targets are the closed :data:`ScalarTarget` union above.
    Analysis probes may introduce additional scalar coordinates—such as a
    signed chassis-hinge angle—without making those coordinates part of the
    public sweep-target vocabulary.  The solver needs only this small common
    analytical interface.
    """

    @property
    def value(self) -> float: ...

    @property
    def mode(self) -> TargetValueMode: ...

    @property
    def kind(self) -> TargetKind: ...

    @property
    def required_points(self) -> tuple[PointKey, ...]: ...

    @property
    def coordinate_identity(self) -> tuple[object, ...]: ...

    @property
    def coordinate_id(self) -> str: ...

    @property
    def selector_point(self) -> PointKey | None: ...

    @property
    def coordinate_description(self) -> str: ...

    @property
    def actuator_coordinate_id(self) -> str | None: ...

    @property
    def actuator_direction(self) -> Direction3 | None: ...

    def measure(self, positions: Mapping[PointKey, Point3]) -> float: ...

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]: ...

    def with_value(self, value: float, mode: TargetValueMode) -> Self: ...

    def map_points(self, mapping: Callable[[PointKey], PointKey]) -> Self: ...


def target_side(target: ScalarTarget) -> Side | None:
    """Return the structural side of a resolved target, if any."""
    return target.structural_side


def target_export_column_name(target: ScalarTarget) -> str:
    """Return the deterministic measured-coordinate export column name."""
    coordinate = target.export_coordinate_id
    side = target_side(target)
    side_suffix = f"_{side.name.lower()}" if side is not None else ""
    return f"target_{coordinate}{side_suffix}"


@dataclass(frozen=True)
class DriveCoordinate:
    """One explicitly driveable topology-owned scalar coordinate."""

    id: str
    kind: TargetKind
    label: str
    unit: str
    point_keys: tuple[PointKey, ...]
    scope: Scope
    side: Side | None = None

    def __post_init__(self) -> None:
        """Require enough physical points for the declared coordinate kind."""
        if self.kind is TargetKind.ELEMENT_LENGTH and len(self.point_keys) != 2:
            raise ValueError("Element-length coordinates require exactly two points")
        if self.kind is TargetKind.ACTUATOR_POSITION and not self.point_keys:
            raise ValueError("Actuator-position coordinates require a physical point")

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        """Return a stable topology identity independent of target value."""
        return (self.kind.value, self.id, *self.point_keys)

    def target(
        self,
        value: float,
        mode: TargetValueMode = TargetValueMode.RELATIVE,
    ) -> ElementLengthTarget:
        """Build a resolved element target for this coordinate."""
        if self.kind is not TargetKind.ELEMENT_LENGTH:
            raise ValueError(f"Unsupported drive coordinate kind: {self.kind}")
        return ElementLengthTarget(
            element_id=self.id,
            point_a=self.point_keys[0],
            point_b=self.point_keys[1],
            value=value,
            mode=mode,
            label=self.label,
            unit=self.unit,
            scope=self.scope,
            side=self.side,
        )

    def position_target(
        self,
        direction: PointTargetDirection,
        value: float,
        mode: TargetValueMode = TargetValueMode.RELATIVE,
    ) -> ActuatorPositionTarget:
        """Build a named actuator-position target using its canonical point."""
        if self.kind is not TargetKind.ACTUATOR_POSITION:
            raise ValueError(f"Unsupported position coordinate kind: {self.kind}")
        return ActuatorPositionTarget(
            actuator_id=self.id,
            point_id=self.point_keys[0],
            direction=direction,
            value=value,
            mode=mode,
            label=self.label,
        )

    def current_value_target(
        self,
        positions: Mapping[PointKey, Point3],
        *,
        direction: PointTargetDirection | None = None,
    ) -> ScalarTarget:
        """Build an absolute target using this coordinate's value in ``positions``.

        Element-length coordinates are self-describing.  A generic actuator
        coordinate additionally needs its physical direction, supplied by its
        :class:`ActuatorDOF`; this keeps the direction out of drive-coordinate
        metadata that is also used by the sweep editor.
        """
        if self.kind is TargetKind.ELEMENT_LENGTH:
            target = self.target(0.0)
        elif self.kind is TargetKind.ACTUATOR_POSITION:
            if direction is None:
                raise ValueError(
                    "Actuator current-value targets require an actuator direction"
                )
            target = self.position_target(direction, 0.0)
        else:  # pragma: no cover - TargetKind is exhaustive at runtime.
            raise ValueError(f"Unsupported drive coordinate kind: {self.kind}")
        return target.with_value(target.measure(positions), TargetValueMode.ABSOLUTE)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> "DriveCoordinate":
        """Remap structural keys for a composed suspension namespace."""
        return replace(
            self,
            point_keys=tuple(mapping(point) for point in self.point_keys),
            side=self.side if side is None else side,
        )


def resolve_target(target: PointTargetDirection) -> Direction3:
    """Resolve a target direction specification to a chassis-space unit direction."""
    if isinstance(target, PointTargetAxis):
        if target.axis is Axis.X:
            return ChassisAxisSystem.X
        if target.axis is Axis.Y:
            return ChassisAxisSystem.Y
        if target.axis is Axis.Z:
            return ChassisAxisSystem.Z
        raise ValueError(f"Unsupported axis: {target.axis!r}")

    if isinstance(target, PointTargetVector):
        return target.vector

    raise TypeError(f"Unsupported target type: {type(target)!r}")


@dataclass(frozen=True)
class ActuatorDOF:
    """One physical actuator coordinate that a sweep must control."""

    id: str
    name: str
    point_keys: tuple[PointKey, ...]
    direction: Direction3

    def matches(self, target: ScalarCoordinateTarget) -> bool:
        """Whether a target controls this actuator coordinate."""
        if target.actuator_coordinate_id != self.id:
            return False
        target_direction = target.actuator_direction
        if target_direction is None:
            return False
        alignment = abs(float(np.dot(target_direction.data, self.direction.data)))
        return alignment >= 1.0 - EPS_GEOMETRIC

    def current_value_target(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> ActuatorPositionTarget:
        """Build this actuator's absolute target measured at ``positions``."""
        target = ActuatorPositionTarget(
            actuator_id=self.id,
            point_id=self.point_keys[0],
            direction=PointTargetVector(self.direction),
            value=0.0,
            mode=TargetValueMode.RELATIVE,
            label=self.name,
        )
        return target.with_value(target.measure(positions), TargetValueMode.ABSOLUTE)


def validate_sweep_controls(
    sweep_config: SweepConfig,
    actuator_dofs: tuple[ActuatorDOF, ...],
) -> None:
    """Require exactly one target for every physical actuator coordinate."""
    for step_index in range(sweep_config.n_steps):
        step_targets = [
            target_sweep[step_index] for target_sweep in sweep_config.target_sweeps
        ]
        for actuator in actuator_dofs:
            matching_targets = [
                target for target in step_targets if actuator.matches(target)
            ]
            if len(matching_targets) != 1:
                raise ValueError(
                    f"Sweep requires exactly one target for actuator "
                    f"'{actuator.name}' along its motion axis; found "
                    f"{len(matching_targets)} at step {step_index}."
                )
