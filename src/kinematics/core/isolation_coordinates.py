"""Analytical scalar coordinates used to isolate counterfactual motion.

Sweep targets describe how a configuration is reached.  An isolation
coordinate instead supplies a zero-rate row to a local tangent problem at an
already solved state.  Keeping that distinction explicit lets a topology ask
"what is steering with suspension travel fixed?" without editing the sweep or
running another nonlinear solve.

The signed hinge angle implemented here is the generalized travel coordinate
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
from kinematics.core.targeting import TargetKind


def _frozen_vector(values: np.ndarray) -> np.ndarray:
    """Return one validated immutable three-vector copy."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Hinge geometry vectors must be finite three-vectors")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class HingeAngleTarget:
    """Absolute or relative signed rotation about one fixed chassis hinge."""

    coordinate: "HingeAngleCoordinate"
    value: float
    mode: TargetValueMode = TargetValueMode.ABSOLUTE

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(
                f"Hinge-angle target '{self.coordinate.id}' must be finite"
            )

    @property
    def kind(self) -> TargetKind:
        return TargetKind.HINGE_ANGLE

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
        return f"hinge-angle coordinate '{self.coordinate.id}'"

    @property
    def actuator_coordinate_id(self) -> None:
        return None

    @property
    def actuator_direction(self) -> None:
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
    ) -> "HingeAngleTarget":
        return replace(self, value=value, mode=mode)

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
    ) -> "HingeAngleTarget":
        return replace(self, coordinate=self.coordinate.map_points(mapping))


@dataclass(frozen=True)
class HingeAngleCoordinate:
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
            raise ValueError("Hinge-angle coordinate ID must not be empty")
        origin = _frozen_vector(self.axis_origin)
        axis = _frozen_vector(self.axis_direction)
        reference = _frozen_vector(self.reference_radial)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= EPS_GEOMETRIC:
            raise ValueError(f"Hinge-angle coordinate '{self.id}' has a zero axis")
        axis = _frozen_vector(axis / axis_norm)
        reference = reference - axis * float(np.dot(axis, reference))
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= EPS_GEOMETRIC:
            raise ValueError(
                f"Hinge-angle coordinate '{self.id}' has no radial lever arm"
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
    ) -> "HingeAngleCoordinate":
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
        return TargetKind.HINGE_ANGLE

    @property
    def point_keys(self) -> tuple[PointKey, PointKey, PointKey]:
        return self.hinge_point_a, self.hinge_point_b, self.carried_point

    @property
    def coordinate_identity(self) -> tuple[object, ...]:
        return (self.kind.value, self.id, *self.point_keys)

    def _radial(self, positions: Mapping[PointKey, Point3]) -> np.ndarray:
        offset = positions[self.carried_point].data - self.axis_origin
        return offset - self.axis_direction * float(np.dot(self.axis_direction, offset))

    def measure(self, positions: Mapping[PointKey, Point3]) -> float:
        """Measure signed rotation from the captured design radial direction."""
        radial = self._radial(positions)
        radial_squared = float(np.dot(radial, radial))
        if radial_squared <= EPS_GEOMETRIC**2:
            raise ValueError(
                f"Hinge-angle coordinate '{self.id}' has no radial lever arm"
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
                f"Hinge-angle coordinate '{self.id}' has no radial lever arm"
            )
        gradient = np.cross(self.axis_direction, radial) / radial_squared
        return ((self.carried_point, gradient),)

    def current_value_target(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> HingeAngleTarget:
        """Materialize an absolute zero-rate hold at the supplied state."""
        return HingeAngleTarget(
            coordinate=self,
            value=self.measure(positions),
            mode=TargetValueMode.ABSOLUTE,
        )

    def map_points(
        self,
        mapping: Callable[[PointKey], PointKey],
        *,
        side: Side | None = None,
    ) -> "HingeAngleCoordinate":
        """Remap structural keys while retaining captured physical geometry."""
        return replace(
            self,
            hinge_point_a=mapping(self.hinge_point_a),
            hinge_point_b=mapping(self.hinge_point_b),
            carried_point=mapping(self.carried_point),
            side=self.side if side is None else side,
        )
