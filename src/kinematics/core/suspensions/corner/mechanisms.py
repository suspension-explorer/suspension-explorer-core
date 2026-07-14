"""Typed actuation and spring mechanisms for suspension corners."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from math import degrees
from typing import TYPE_CHECKING, cast

import numpy as np

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    RockerElement,
    RockerPickup,
    RockerPickupType,
    SuspensionElement,
    TorsionElement,
    VariableLengthLinkElement,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3, extract_array
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    signed_angle_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    chiral_rigid_point_constraints,
)

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow


PUSHROD_POINTS = frozenset({PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD})
# A -> B defines the axis direction; neither datum implies vehicle orientation.
ROCKER_AXIS_POINTS = frozenset({PointID.ROCKER_AXIS_A, PointID.ROCKER_AXIS_B})
COIL_SPRING_POINTS = frozenset({PointID.STRUT_TOP, PointID.STRUT_BOTTOM})


@dataclass(frozen=True)
class ActuationDirect:
    """
    Direct connection between a corner member and its selected spring.

    The locating architecture supplies spring_pickup_body: three points on
    the rigid body that carries the moving spring pickup (for a double
    wishbone, the lower wishbone). The mechanism owns no architecture
    geometry of its own.
    """

    spring_pickup_body: tuple[PointID, PointID, PointID]

    @property
    def required_points(self) -> frozenset[PointID]:
        """Return points owned by direct actuation itself."""
        return frozenset()

    @property
    def free_points(self) -> tuple[PointID, ...]:
        """Return moving points owned by direct actuation itself."""
        return ()

    @property
    def output_points(self) -> tuple[PointID, ...]:
        """Return output points owned by direct actuation itself."""
        return ()

    @property
    def torsion_axis(self) -> tuple[PointID, PointID] | None:
        """Direct torsion geometry is not yet defined."""
        return None

    def validate(self, hardpoints: Mapping[PointKey, Point3]) -> None:
        """Validate direct actuation geometry."""

    def constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Direct actuation adds no constraint without a selected spring."""
        return []

    def spring_constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Attach a moving coil-spring pickup rigidly to the supplied body."""
        return chiral_rigid_point_constraints(
            initial,
            PointID.STRUT_BOTTOM,
            self.spring_pickup_body,
        )

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Direct actuation adds no derivative metrics itself."""
        return ()

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        side: Side,
    ) -> "MetricRow":
        """Direct actuation adds no state metrics itself."""
        return OrderedDict()

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Direct actuation adds no physical element itself."""
        return ()


@dataclass(frozen=True)
class ActuationPushrodRocker:
    """
    Pushrod and rocker actuation with explicitly requested external pickups.

    The locating architecture supplies pushrod_outboard_body: points on the
    rigid body that carries the outboard pushrod end (for a double wishbone,
    the upright). The rocker group itself is mechanism-owned.
    """

    pushrod_outboard_body: tuple[PointID, ...]
    external_pickups: tuple[RockerPickup, ...] = ()

    @property
    def external_point_ids(self) -> tuple[PointID, ...]:
        """Return side-local point identifiers for external rocker connections."""
        points: list[PointID] = []
        for pickup in self.external_pickups:
            if not isinstance(pickup.point, PointID):
                raise TypeError("Corner rocker pickups must use PointID values")
            points.append(pickup.point)
        return tuple(points)

    @property
    def rocker_mounted_point_ids(self) -> tuple[PointID, ...]:
        """Return pushrod and external pickups constrained to the rocker."""
        return PointID.PUSHROD_INBOARD, *self.external_point_ids

    @property
    def required_points(self) -> frozenset[PointID]:
        """Return pushrod, rocker, and external pickup points."""
        return PUSHROD_POINTS | ROCKER_AXIS_POINTS | frozenset(self.external_point_ids)

    @property
    def free_points(self) -> tuple[PointID, ...]:
        """Return moving pushrod and rocker pickup points."""
        return (
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
            *self.external_point_ids,
        )

    @property
    def output_points(self) -> tuple[PointID, ...]:
        """Return explicit actuation output points."""
        return (
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
            *self.external_point_ids,
        )

    @property
    def torsion_axis(self) -> tuple[PointID, PointID]:
        """Return the authored rocker rotation axis."""
        return PointID.ROCKER_AXIS_A, PointID.ROCKER_AXIS_B

    def validate(self, hardpoints: Mapping[PointKey, Point3]) -> None:
        """Validate the outboard anchors, rocker axis, and pickup radii."""
        # Three non-collinear anchors are the minimum to fix a point rigidly
        # to a body; collinear anchors leave rotation about their line free.
        if len(self.pushrod_outboard_body) < 3:
            raise ValueError(
                "Pushrod actuation requires at least three outboard body anchors"
            )
        anchor_a, anchor_b, anchor_c = (
            hardpoints[point] for point in self.pushrod_outboard_body[:3]
        )
        if compute_point_point_distance(anchor_a, anchor_b) <= EPS_GEOMETRIC:
            raise ValueError("Pushrod outboard body anchors must be distinct")
        anchor_line = (anchor_b - anchor_a).normalize()
        if compute_point_to_line_distance(anchor_c, anchor_a, anchor_line) <= (
            EPS_GEOMETRIC
        ):
            raise ValueError(
                "The first three pushrod outboard body anchors must not be collinear"
            )
        axis_a = hardpoints[PointID.ROCKER_AXIS_A]
        axis_b = hardpoints[PointID.ROCKER_AXIS_B]
        if compute_point_point_distance(axis_a, axis_b) <= EPS_GEOMETRIC:
            raise ValueError("Rocker axis points must be distinct")
        axis_direction = (axis_b - axis_a).normalize()
        for point in self.rocker_mounted_point_ids:
            radius = compute_point_to_line_distance(
                hardpoints[point], axis_a, axis_direction
            )
            if radius <= EPS_GEOMETRIC:
                raise ValueError(f"{point.name} must not lie on the rocker axis")

    def constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Build fixed pushrod and rigid rocker pickup constraints."""
        positions = initial.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        # The first three anchors hold the outboard pushrod end rigidly with
        # authored handedness; further anchors add plain redundant distances.
        primary_anchors = cast(
            "tuple[PointID, PointID, PointID]",
            self.pushrod_outboard_body[:3],
        )
        constraints: list[Constraint] = list(
            chiral_rigid_point_constraints(
                initial,
                PointID.PUSHROD_OUTBOARD,
                primary_anchors,
            )
        )
        constraints.extend(
            distance(PointID.PUSHROD_OUTBOARD, anchor)
            for anchor in self.pushrod_outboard_body[3:]
        )
        constraints.extend(
            (
                distance(PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD),
                distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_A),
                distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_B),
            )
        )
        for point in self.external_point_ids:
            constraints.extend(
                chiral_rigid_point_constraints(
                    initial,
                    point,
                    (
                        PointID.ROCKER_AXIS_A,
                        PointID.ROCKER_AXIS_B,
                        PointID.PUSHROD_INBOARD,
                    ),
                )
            )
        return constraints

    def spring_constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Attach a moving coil-spring pickup rigidly to the rocker."""
        return chiral_rigid_point_constraints(
            initial,
            PointID.STRUT_BOTTOM,
            (
                PointID.ROCKER_AXIS_A,
                PointID.ROCKER_AXIS_B,
                PointID.PUSHROD_INBOARD,
            ),
        )

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare rocker rotation relative to hub vertical travel."""
        return (
            self.rotation_derivative(
                initial,
                side,
                "rocker_angle",
                "Rocker Angle",
            ),
        )

    def rotation_derivative(
        self,
        initial: SuspensionState,
        side: Side,
        response_name: str,
        response_label: str,
    ) -> DerivativeMetricDefinition:
        """Build one rocker-rotation derivative with the requested response name."""
        axis_a = extract_array(initial.positions[PointID.ROCKER_AXIS_A])
        axis_b = extract_array(initial.positions[PointID.ROCKER_AXIS_B])
        axis_direction = axis_b - axis_a
        axis_length = float(np.linalg.norm(axis_direction))
        if axis_length < EPS_GEOMETRIC:
            raise ValueError(
                "Rocker angle derivative requires distinct rocker axis points."
            )
        axis_direction /= axis_length
        design_pickup = extract_array(initial.positions[PointID.PUSHROD_INBOARD])

        def rocker_rotation(positions):
            return side.lateral_sign * kernels.rotation_about_fixed_axis_deg(
                positions,
                PointID.PUSHROD_INBOARD,
                design_pickup,
                axis_a,
                axis_direction,
            )

        return DerivativeMetricDefinition(
            response=CallableScalarResponse(
                rocker_rotation,
                name=response_name,
                unit=MetricUnit.DEG,
                label=response_label,
            ),
            driver=PointCoordinateResponse.from_world_axis(
                PointID.WHEEL_CENTER,
                Axis.Z,
                name="hub_z",
                unit=MetricUnit.MM,
                label="Hub Z",
            ),
        )

    def rocker_angle(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        side: Side,
    ) -> float:
        """Return signed rocker rotation from the design state in degrees."""
        axis_a = initial.get(PointID.ROCKER_AXIS_A)
        axis = (initial.get(PointID.ROCKER_AXIS_B) - axis_a).normalize()
        return (
            degrees(
                signed_angle_about_axis(
                    initial.get(PointID.PUSHROD_INBOARD),
                    state.get(PointID.PUSHROD_INBOARD),
                    axis_a,
                    axis,
                )
            )
            * side.lateral_sign
        )

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        side: Side,
    ) -> "MetricRow":
        """Return rocker rotation from the design state."""
        return OrderedDict([("rocker_angle", self.rocker_angle(state, initial, side))])

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the pushrod and rocker declarations."""
        rotation_axis: tuple[PointKey, PointKey] = self.torsion_axis
        return (
            RigidLinkElement(
                label="Pushrod",
                type=ElementType.PUSHROD,
                point_a=PointID.PUSHROD_OUTBOARD,
                point_b=PointID.PUSHROD_INBOARD,
            ),
            RockerElement(
                label="Rocker",
                rotation_axis=rotation_axis,
                pickups=(
                    RockerPickup(
                        PointID.PUSHROD_INBOARD,
                        RockerPickupType.PUSHROD,
                    ),
                    *self.external_pickups,
                ),
            ),
        )


type Actuation = ActuationDirect | ActuationPushrodRocker


@dataclass(frozen=True)
class CornerSpringNone:
    """Explicit absence of a corner spring mechanism."""

    required_points: frozenset[PointID] = frozenset()
    free_points: tuple[PointID, ...] = ()
    output_points: tuple[PointID, ...] = ()
    damper_points: tuple[PointID, PointID] | None = None

    def validate(self, actuation: Actuation) -> None:
        """Accept either actuation without a spring."""

    def constraints(
        self,
        initial: SuspensionState,
        actuation: Actuation,
    ) -> list[Constraint]:
        """Add no spring constraints."""
        return []

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Add no spring derivative metrics."""
        return ()

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> "MetricRow":
        """Add no spring state metrics."""
        return OrderedDict()

    def elements(self, actuation: Actuation) -> tuple[SuspensionElement, ...]:
        """Add no spring elements."""
        return ()


@dataclass(frozen=True)
class CornerSpringCoilover:
    """Linear corner coil spring or coilover."""

    required_points: frozenset[PointID] = COIL_SPRING_POINTS
    free_points: tuple[PointID, ...] = (PointID.STRUT_BOTTOM,)
    output_points: tuple[PointID, ...] = (
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
    )
    damper_points: tuple[PointID, PointID] = (
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
    )

    def validate(self, actuation: Actuation) -> None:
        """Both implemented actuation types support a linear corner spring."""

    def constraints(
        self,
        initial: SuspensionState,
        actuation: Actuation,
    ) -> list[Constraint]:
        """Attach the moving spring pickup to the selected actuation."""
        return actuation.spring_constraints(initial)

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare damper length relative to hub vertical travel."""
        return (
            DerivativeMetricDefinition(
                response=PointDistanceResponse(
                    PointID.STRUT_TOP,
                    PointID.STRUT_BOTTOM,
                    name="damper_length",
                    unit=MetricUnit.MM,
                    label="Damper Length",
                ),
                driver=PointCoordinateResponse.from_world_axis(
                    PointID.WHEEL_CENTER,
                    Axis.Z,
                    name="hub_z",
                    unit=MetricUnit.MM,
                    label="Hub Z",
                ),
            ),
        )

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> "MetricRow":
        """The shared metric catalog calculates installed damper length."""
        return OrderedDict()

    def elements(self, actuation: Actuation) -> tuple[SuspensionElement, ...]:
        """Return the physical spring/damper link."""
        return (
            VariableLengthLinkElement(
                label="Spring/Damper",
                type=ElementType.SPRING_DAMPER,
                point_a=PointID.STRUT_TOP,
                point_b=PointID.STRUT_BOTTOM,
            ),
        )


@dataclass(frozen=True)
class CornerSpringTorsionBar:
    """Corner torsion spring driven by a compatible rotary actuation."""

    required_points: frozenset[PointID] = frozenset()
    free_points: tuple[PointID, ...] = ()
    output_points: tuple[PointID, ...] = ()
    damper_points: tuple[PointID, PointID] | None = None

    def validate(self, actuation: Actuation) -> None:
        """Require an actuation mechanism with a defined torsion axis."""
        if actuation.torsion_axis is None:
            raise ValueError(
                "Corner torsion bar is not supported by direct actuation yet"
            )

    def constraints(
        self,
        initial: SuspensionState,
        actuation: Actuation,
    ) -> list[Constraint]:
        """A torsion spring adds no positional constraint."""
        return []

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare torsion-bar twist relative to hub vertical travel."""
        if not isinstance(actuation, ActuationPushrodRocker):
            raise ValueError("Corner torsion-bar derivatives require rocker actuation")
        return (
            actuation.rotation_derivative(
                initial,
                side,
                "torsion_bar_twist",
                "Torsion Bar Twist",
            ),
        )

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> "MetricRow":
        """Return torsion-bar twist from the design state."""
        if not isinstance(actuation, ActuationPushrodRocker):
            raise ValueError("Corner torsion-bar metrics require rocker actuation")
        return OrderedDict(
            [("torsion_bar_twist", actuation.rocker_angle(state, initial, side))]
        )

    def elements(self, actuation: Actuation) -> tuple[SuspensionElement, ...]:
        """Return a torsion member on the actuation rotation axis."""
        if actuation.torsion_axis is None:
            raise ValueError("Corner torsion bar requires a rotation axis")
        return (
            TorsionElement(
                label="Torsion Bar",
                type=ElementType.TORSION_BAR,
                rotation_axis=cast(
                    "tuple[PointKey, PointKey]",
                    actuation.torsion_axis,
                ),
                attachments=(),
            ),
        )


type CornerSpring = CornerSpringNone | CornerSpringCoilover | CornerSpringTorsionBar
