"""Scalar suspension coordinates shared by sweeps and local responses.

A coordinate owns only measurement, identity, and analytical-gradient
geometry.  Giving it a varying target drives a nonlinear sweep; materialising
its current value through :mod:`kinematics.core.holds` fixes the same quantity
either throughout a sweep or in a local tangent problem.  This keeps the
meaning of *what is fixed* independent from *when the reference is captured*.

``PhysicalCoordinate`` covers projected point/actuator positions and installed
element lengths.  ``ArmAngleCoordinate`` supplies the topology-owned travel
coordinate needed by ideal wishbone and lower-control-arm mechanisms.

The signed arm angle is the generalized travel coordinate
for a rigid arm rotating about a fixed chassis hinge.  The authored hinge
points define an immutable axis; the carried point defines a radial vector.
For axis unit vector ``a`` and projected radial vector ``r``, its exact local
gradient is

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

import numpy as np

from kinematics.core.enums import Scope, TargetValueMode
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.targeting import (
    ActuatorPositionTarget,
    ElementLengthTarget,
    PointTarget,
    PointTargetDirection,
    ScalarCoordinateTarget,
    ScalarTarget,
    SweepConfig,
    TargetKind,
    resolve_target,
)


def _frozen_vector(values: np.ndarray) -> np.ndarray:
    """Return one validated immutable three-vector copy."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Hinge geometry vectors must be finite three-vectors")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PhysicalCoordinate:
    """One measurable point, actuator, or element-length coordinate.

    Projected coordinates carry their direction once it is resolved. Topology
    actuator declarations bind the physical motion direction so the same
    object can describe both a driveable coordinate and a required control.
    An authored target may replace that direction with :meth:`with_direction`;
    control validation then checks alignment with the physical declaration.
    """

    id: str
    kind: TargetKind
    label: str
    unit: str
    point_keys: tuple[PointKey, ...]
    scope: Scope
    side: Side | None = None
    direction: PointTargetDirection | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Physical coordinate ID must not be empty")
        expected_points = {
            TargetKind.POINT: 1,
            TargetKind.ELEMENT_LENGTH: 2,
        }.get(self.kind)
        if self.kind is TargetKind.ACTUATOR_POSITION:
            if not self.point_keys:
                raise ValueError(
                    "Actuator-position coordinates require a physical point"
                )
        elif expected_points is None:
            raise ValueError(f"Unsupported physical coordinate kind: {self.kind}")
        elif len(self.point_keys) != expected_points:
            raise ValueError(
                f"{self.kind.value} coordinates require exactly "
                f"{expected_points} physical point(s)"
            )
        if self.kind is TargetKind.ELEMENT_LENGTH and self.direction is not None:
            raise ValueError("Element-length coordinates do not accept a direction")

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        """Return stable physical identity independent of target value."""
        if self.kind is TargetKind.ELEMENT_LENGTH:
            return (self.kind.value, self.id, *self.point_keys)
        if self.direction is None:
            raise ValueError(f"Projected coordinate '{self.id}' requires a direction")
        prototype = self.target(0.0, TargetValueMode.RELATIVE)
        return prototype.coordinate_identity

    @property
    def coordinate_description(self) -> str:
        """Describe the resolved scalar coordinate for diagnostics."""
        return self.target(0.0).coordinate_description

    def with_direction(self, direction: PointTargetDirection) -> "PhysicalCoordinate":
        """Bind a projected point or actuator coordinate to one direction."""
        if self.kind is TargetKind.ELEMENT_LENGTH:
            raise ValueError("Element-length coordinates do not accept a direction")
        return replace(self, direction=direction)

    def position_target(
        self,
        direction: PointTargetDirection,
        value: float,
        mode: TargetValueMode = TargetValueMode.RELATIVE,
    ) -> ScalarTarget:
        """Build a projected target while binding its physical direction."""
        return self.with_direction(direction).target(value, mode)

    def target(
        self,
        value: float,
        mode: TargetValueMode = TargetValueMode.RELATIVE,
    ) -> ScalarTarget:
        """Build one scalar target from this resolved coordinate."""
        if self.kind is TargetKind.ELEMENT_LENGTH:
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
        if self.direction is None:
            raise ValueError(f"Projected coordinate '{self.id}' requires a direction")
        if self.kind is TargetKind.ACTUATOR_POSITION:
            return ActuatorPositionTarget(
                actuator_id=self.id,
                point_id=self.point_keys[0],
                direction=self.direction,
                value=value,
                mode=mode,
                label=self.label,
            )
        if self.kind is TargetKind.POINT:
            return PointTarget(
                point_id=self.point_keys[0],
                direction=self.direction,
                value=value,
                mode=mode,
            )
        raise ValueError(f"Unsupported physical coordinate kind: {self.kind}")

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Measure the coordinate using its ordinary target implementation."""
        return self.target(0.0).measure(positions)

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        """Return the coordinate's analytical point gradients."""
        return self.target(0.0).point_partials(positions)

    def current_value_target(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> ScalarTarget:
        """Build an absolute target fixed at the supplied state."""
        target = self.target(0.0)
        return target.with_value(target.measure(positions), TargetValueMode.ABSOLUTE)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> "PhysicalCoordinate":
        """Remap structural keys for a composed suspension namespace."""
        return replace(
            self,
            point_keys=tuple(mapping(point) for point in self.point_keys),
            side=self.side if side is None else side,
        )


@dataclass(frozen=True)
class ArmAngleTarget:
    """Absolute or relative signed rotation about one fixed chassis hinge."""

    coordinate: "ArmAngleCoordinate"
    value: float
    mode: TargetValueMode = TargetValueMode.ABSOLUTE

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(f"Arm-angle target '{self.coordinate.id}' must be finite")

    @property
    def kind(self) -> TargetKind:
        return TargetKind.ARM_ANGLE

    @property
    def required_points(self) -> tuple[PointKey, ...]:
        # The hinge axis is fixed design geometry; only the carried point moves.
        return (self.coordinate.carried_point,)

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return self.coordinate.coordinate_identity

    @property
    def coordinate_id(self) -> str:
        return self.coordinate.id

    @property
    def selector_point(self) -> None:
        return None

    @property
    def coordinate_description(self) -> str:
        return f"arm-angle coordinate '{self.coordinate.id}'"

    @property
    def driven_points(self) -> tuple[()]:
        return ()

    @property
    def drive_coordinate_key(self) -> None:
        return None

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        return self.coordinate.measure(positions)

    def point_partials(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[tuple[PointKey, np.ndarray], ...]:
        return self.coordinate.point_partials(positions)

    def with_value(
        self,
        value: float,
        mode: TargetValueMode,
    ) -> "ArmAngleTarget":
        return replace(self, value=value, mode=mode)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
    ) -> "ArmAngleTarget":
        return replace(self, coordinate=self.coordinate.map_points(mapping))


@dataclass(frozen=True)
class ArmAngleCoordinate:
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
        reference = _frozen_vector(reference / reference_norm)
        object.__setattr__(self, "axis_origin", origin)
        object.__setattr__(self, "axis_direction", axis)
        object.__setattr__(self, "reference_radial", reference)

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
    ) -> "ArmAngleCoordinate":
        """Capture a fixed hinge and deterministic zero angle from design state."""
        origin = positions[hinge_point_a].data
        axis = positions[hinge_point_b].data - origin
        radial = positions[carried_point].data - origin
        return cls(
            id=id,
            label=label,
            hinge_point_a=hinge_point_a,
            hinge_point_b=hinge_point_b,
            carried_point=carried_point,
            axis_origin=origin,
            axis_direction=axis,
            reference_radial=radial,
            scope=scope,
            side=side,
        )

    @property
    def kind(self) -> TargetKind:
        return TargetKind.ARM_ANGLE

    @property
    def point_keys(self) -> tuple[PointKey, PointKey, PointKey]:
        return self.hinge_point_a, self.hinge_point_b, self.carried_point

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return (self.kind.value, self.id, *self.point_keys)

    @property
    def coordinate_description(self) -> str:
        return f"arm-angle coordinate '{self.id}'"

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

    def current_value_target(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> ArmAngleTarget:
        """Materialize an absolute zero-rate hold at the supplied state."""
        return ArmAngleTarget(
            coordinate=self,
            value=self.measure(positions),
            mode=TargetValueMode.ABSOLUTE,
        )

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> "ArmAngleCoordinate":
        """Remap structural keys while retaining captured physical geometry."""
        return replace(
            self,
            hinge_point_a=mapping(self.hinge_point_a),
            hinge_point_b=mapping(self.hinge_point_b),
            carried_point=mapping(self.carried_point),
            side=self.side if side is None else side,
        )


type ScalarCoordinate = PhysicalCoordinate | ArmAngleCoordinate


def actuator_coordinate_matches(
    required: PhysicalCoordinate,
    candidate: ScalarCoordinateTarget | ScalarCoordinate,
) -> bool:
    """Whether a target or hold controls one required actuator coordinate."""
    if required.kind is not TargetKind.ACTUATOR_POSITION:
        raise ValueError(
            f"Required actuator coordinate '{required.id}' has kind "
            f"'{required.kind.value}'."
        )
    if required.direction is None:
        raise ValueError(
            f"Required actuator coordinate '{required.id}' has no motion direction."
        )

    if isinstance(candidate, ActuatorPositionTarget):
        candidate_id = candidate.actuator_id
        candidate_direction = candidate.direction
    elif (
        isinstance(candidate, PhysicalCoordinate)
        and candidate.kind is TargetKind.ACTUATOR_POSITION
        and candidate.direction is not None
    ):
        candidate_id = candidate.id
        candidate_direction = candidate.direction
    else:
        return False

    if candidate_id != required.id:
        return False
    alignment = abs(
        float(
            np.dot(
                resolve_target(candidate_direction).data,
                resolve_target(required.direction).data,
            )
        )
    )
    return alignment >= 1.0 - EPS_GEOMETRIC


def validate_sweep_controls(
    sweep_config: SweepConfig,
    actuator_coordinates: tuple[PhysicalCoordinate, ...],
) -> None:
    """Require exactly one target for every required actuator coordinate."""
    for step_index in range(sweep_config.n_steps):
        step_targets = [
            *(target_sweep[step_index] for target_sweep in sweep_config.target_sweeps),
            *sweep_config.hold.coordinates,
        ]
        for actuator in actuator_coordinates:
            matching_targets = [
                target
                for target in step_targets
                if actuator_coordinate_matches(actuator, target)
            ]
            if len(matching_targets) != 1:
                actuator_name = (
                    "steering rack" if actuator.id == "rack" else actuator.label.lower()
                )
                raise ValueError(
                    f"Sweep requires exactly one target for actuator "
                    f"'{actuator_name}' along its motion axis; found "
                    f"{len(matching_targets)} at step {step_index}."
                )
