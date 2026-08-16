"""Typed actuation, spring, and damper mechanisms for suspension corners."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
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
from kinematics.core.enums import Axis, PointID, Scope
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.registry import MetricKind, MetricSpec
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.definitions import get_point_along_line
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3, extract_array
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    rotate_point_about_axis,
    signed_angle_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    anchored_rigid_point_constraints,
    chiral_rigid_point_constraints,
    validate_rigid_anchor_points,
)

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow


PUSHROD_POINTS = frozenset({PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD})
# A -> B defines the axis direction; neither datum implies vehicle orientation.
ROCKER_AXIS_POINTS = frozenset({PointID.ROCKER_AXIS_A, PointID.ROCKER_AXIS_B})
COIL_SPRING_POINTS = frozenset({PointID.STRUT_TOP, PointID.STRUT_BOTTOM})
LINEAR_DAMPER_POINTS = frozenset({PointID.DAMPER_CHASSIS, PointID.DAMPER_ROCKER})

ROCKER_ANGLE_SPEC = MetricSpec(
    "rocker_angle",
    "Rocker Angle",
    MetricUnit.DEG,
    MetricKind.STATE,
    Scope.CORNER,
    "rocker",
)
TORSION_BAR_TWIST_SPEC = MetricSpec(
    "torsion_bar_twist",
    "Torsion Bar Twist",
    MetricUnit.DEG,
    MetricKind.STATE,
    Scope.CORNER,
    "torsion_bar",
)


@dataclass(frozen=True)
class ActuationDirect:
    """
    Direct connection between a corner member and its selected spring.

    The locating architecture supplies ``spring_pickup_body``. Ordinarily it
    contains at least three anchors on the rigid body carrying the pickup. A
    two-joint rod may instead carry a centreline pickup when
    ``derive_pickup_on_link`` is true; that pickup is a derived point rather
    than an independent solver variable.
    """

    spring_pickup_body: tuple[PointID, ...]
    derive_pickup_on_link: bool = False

    @property
    def moving_pickup_point(self) -> PointID:
        """Return the spring pickup carried by the selected corner body."""
        return PointID.STRUT_BOTTOM

    @property
    def moving_pickup_body(self) -> tuple[PointID, ...]:
        """Return the rigid body anchors that carry the spring pickup."""
        return self.spring_pickup_body

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
    def derived_pickup_points(self) -> frozenset[PointID]:
        """Return pickups supplied through the derived-point graph."""
        return (
            frozenset({self.moving_pickup_point})
            if self.derive_pickup_on_link
            else frozenset()
        )

    @property
    def torsion_axis(self) -> tuple[PointID, PointID] | None:
        """Direct torsion geometry is not yet defined."""
        return None

    def validate(self, hardpoints: Mapping[PointKey, Point3]) -> None:
        """Validate direct actuation geometry."""
        if self.derive_pickup_on_link:
            if len(self.spring_pickup_body) != 2:
                raise ValueError(
                    "On-link direct actuation requires exactly two link joints"
                )
            inboard, outboard = self.spring_pickup_body
            inboard_position = hardpoints[inboard]
            outboard_position = hardpoints[outboard]
            rod_length = compute_point_point_distance(
                inboard_position,
                outboard_position,
            )
            if rod_length <= EPS_GEOMETRIC:
                raise ValueError(
                    "On-link actuation requires distinct link joint positions"
                )
            pickup = hardpoints.get(self.moving_pickup_point)
            if pickup is None:
                return
            off_axis = compute_point_to_line_distance(
                pickup,
                inboard_position,
                (outboard_position - inboard_position).normalize(),
            )
            if off_axis > LINK_PICKUP_ALIGNMENT_TOLERANCE_MM:
                raise ValueError(
                    f"{self.moving_pickup_point.name} sits {off_axis:.3f} mm off "
                    f"the line from {inboard.name} to {outboard.name}. A "
                    "two-joint rod carries a pickup only on its own centreline."
                )
            axial = self.pickup_axial_offset(hardpoints)
            if axial <= EPS_GEOMETRIC or axial >= rod_length - EPS_GEOMETRIC:
                raise ValueError(
                    f"{self.moving_pickup_point.name} must lie between "
                    f"{inboard.name} and {outboard.name} along the link"
                )
            return
        validate_rigid_anchor_points(
            hardpoints, self.spring_pickup_body, "Direct spring actuation"
        )

    def pickup_axial_offset(self, hardpoints: Mapping[PointKey, Point3]) -> float:
        """Return the inboard-joint-to-pickup distance for an on-link pickup."""
        if not self.derive_pickup_on_link or len(self.spring_pickup_body) != 2:
            raise ValueError("Direct actuation pickup is not derived on a link")
        inboard, outboard = self.spring_pickup_body
        inboard_position = hardpoints[inboard]
        rod_axis = (hardpoints[outboard] - inboard_position).normalize()
        pickup_vector = hardpoints[self.moving_pickup_point] - inboard_position
        return float(pickup_vector.data.dot(rod_axis.data))

    def constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Direct actuation adds no constraint without a selected spring."""
        return []

    def linear_link_constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Attach a moving linear-link pickup rigidly to the supplied body."""
        if self.derive_pickup_on_link:
            return []
        return anchored_rigid_point_constraints(
            initial,
            PointID.STRUT_BOTTOM,
            self.spring_pickup_body,
        )

    def derived_spec(
        self,
        hardpoints: Mapping[PointKey, Point3],
    ) -> DerivedPointsSpec[PointID]:
        """Return the optional centreline pickup contribution."""
        if not self.derive_pickup_on_link:
            return DerivedPointsSpec({}, {})
        if len(self.spring_pickup_body) != 2:
            raise ValueError(
                "On-link direct actuation requires exactly two link joints"
            )
        inboard, outboard = self.spring_pickup_body
        pickup = self.moving_pickup_point
        return DerivedPointsSpec(
            functions={
                pickup: partial(
                    get_point_along_line,
                    start_point=inboard,
                    end_point=outboard,
                    distance_from_start=self.pickup_axial_offset(hardpoints),
                )
            },
            dependencies={pickup: {inboard, outboard}},
        )

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Direct actuation adds no derivative metrics itself."""
        return ()

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Declare no direct-actuation state metrics."""
        return ()

    def topology_metric_values(
        self,
        state: SuspensionState,
        initial: SuspensionState,
        side: Side,
    ) -> "MetricRow":
        """Direct actuation adds no state metrics itself."""
        return OrderedDict()

    def elements(
        self,
        additional_pickups: tuple[RockerPickup, ...] = (),
    ) -> tuple[SuspensionElement, ...]:
        """Direct actuation adds no physical element itself."""
        return ()


# How far an authored on-link pickup may sit off the rod centreline, in
# millimetres, before the coincident-with-the-link modelling choice is
# considered violated rather than an authoring rounding error.
LINK_PICKUP_ALIGNMENT_TOLERANCE_MM = 1.0


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
    def moving_pickup_point(self) -> PointID:
        """Return the outboard pushrod pickup carried by the corner body."""
        return PointID.PUSHROD_OUTBOARD

    @property
    def moving_pickup_body(self) -> tuple[PointID, ...]:
        """Return the rigid body anchors that carry the pushrod pickup."""
        return self.pushrod_outboard_body

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

    @property
    def derived_pickup_points(self) -> frozenset[PointID]:
        """Rocker actuation owns no derived pickups."""
        return frozenset()

    def validate(self, hardpoints: Mapping[PointKey, Point3]) -> None:
        """Validate the outboard anchors, rocker axis, and pickup radii."""
        validate_rigid_anchor_points(
            hardpoints, self.pushrod_outboard_body, "Pushrod actuation"
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

    def rotate_rocker_group(
        self,
        positions: dict[PointKey, Point3],
        angle_rad: float,
        additional_rocker_points: tuple[PointID, ...] = (),
    ) -> None:
        """Rotate all rocker-mounted pickups by a solved setup angle."""
        axis_a = positions[PointID.ROCKER_AXIS_A]
        axis_b = positions[PointID.ROCKER_AXIS_B]
        axis = (axis_b - axis_a).normalize()
        for point in dict.fromkeys(
            (*self.rocker_mounted_point_ids, *additional_rocker_points)
        ):
            positions[point] = rotate_point_about_axis(
                positions[point],
                axis_a,
                axis,
                angle_rad,
            )

    def constraints(self, initial: SuspensionState) -> list[Constraint]:
        """Build fixed pushrod and rigid rocker pickup constraints."""
        positions = initial.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        constraints: list[Constraint] = anchored_rigid_point_constraints(
            initial,
            PointID.PUSHROD_OUTBOARD,
            self.pushrod_outboard_body,
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

    def derived_spec(
        self,
        hardpoints: Mapping[PointKey, Point3],
    ) -> DerivedPointsSpec[PointID]:
        """Rocker actuation contributes no derived points."""
        return DerivedPointsSpec({}, {})

    def linear_link_constraints(
        self,
        initial: SuspensionState,
        moving_point: PointID = PointID.STRUT_BOTTOM,
    ) -> list[Constraint]:
        """Attach a moving linear-link pickup rigidly to the rocker."""
        return chiral_rigid_point_constraints(
            initial,
            moving_point,
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

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Declare rocker rotation from the design state."""
        return (ROCKER_ANGLE_SPEC,)

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
            driver=PointCoordinateResponse.from_chassis_axis(
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

    def elements(
        self,
        additional_pickups: tuple[RockerPickup, ...] = (),
    ) -> tuple[SuspensionElement, ...]:
        """Return the pushrod and rocker declarations with composed pickups."""
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
                    *additional_pickups,
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
    rocker_mounted_points: tuple[PointID, ...] = ()
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

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Declare no spring state metrics."""
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
    rocker_mounted_points: tuple[PointID, ...] = (PointID.STRUT_BOTTOM,)
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
        return actuation.linear_link_constraints(initial)

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
                driver=PointCoordinateResponse.from_chassis_axis(
                    PointID.WHEEL_CENTER,
                    Axis.Z,
                    name="hub_z",
                    unit=MetricUnit.MM,
                    label="Hub Z",
                ),
            ),
        )

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Declare no additional coilover state metrics."""
        return ()

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
    rocker_mounted_points: tuple[PointID, ...] = ()
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

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Declare torsion-bar twist from the design state."""
        return (TORSION_BAR_TWIST_SPEC,)

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


@dataclass(frozen=True)
class CornerDamperNone:
    """Explicit absence of an independent corner damper."""

    required_points: frozenset[PointID] = frozenset()
    free_points: tuple[PointID, ...] = ()
    output_points: tuple[PointID, ...] = ()
    rocker_mounted_points: tuple[PointID, ...] = ()
    rocker_pickups: tuple[RockerPickup, ...] = ()
    damper_points: tuple[PointID, PointID] | None = None

    def validate(
        self,
        actuation: Actuation,
        hardpoints: Mapping[PointKey, Point3],
    ) -> None:
        """Accept either actuation when no independent damper is fitted."""

    def constraints(
        self,
        initial: SuspensionState,
        actuation: Actuation,
    ) -> list[Constraint]:
        """Add no independent damper constraints."""
        return []

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Add no independent damper derivative metrics."""
        return ()

    def elements(self, actuation: Actuation) -> tuple[SuspensionElement, ...]:
        """Add no independent damper element."""
        return ()


@dataclass(frozen=True)
class CornerDamperLinear:
    """Independent chassis-to-rocker linear damper."""

    required_points: frozenset[PointID] = LINEAR_DAMPER_POINTS
    free_points: tuple[PointID, ...] = (PointID.DAMPER_ROCKER,)
    output_points: tuple[PointID, ...] = (
        PointID.DAMPER_CHASSIS,
        PointID.DAMPER_ROCKER,
    )
    rocker_mounted_points: tuple[PointID, ...] = (PointID.DAMPER_ROCKER,)
    rocker_pickups: tuple[RockerPickup, ...] = (
        RockerPickup(PointID.DAMPER_ROCKER, RockerPickupType.DAMPER),
    )
    damper_points: tuple[PointID, PointID] = (
        PointID.DAMPER_CHASSIS,
        PointID.DAMPER_ROCKER,
    )

    def validate(
        self,
        actuation: Actuation,
        hardpoints: Mapping[PointKey, Point3],
    ) -> None:
        """Require a distinct damper whose moving pickup lies off the rocker axis."""
        if not isinstance(actuation, ActuationPushrodRocker):
            raise ValueError(
                "A linear inboard damper requires pushrod-rocker actuation"
            )
        chassis = hardpoints[PointID.DAMPER_CHASSIS]
        rocker = hardpoints[PointID.DAMPER_ROCKER]
        if compute_point_point_distance(chassis, rocker) <= EPS_GEOMETRIC:
            raise ValueError("Independent damper endpoints must be distinct")
        axis_a = hardpoints[PointID.ROCKER_AXIS_A]
        axis = (hardpoints[PointID.ROCKER_AXIS_B] - axis_a).normalize()
        if compute_point_to_line_distance(rocker, axis_a, axis) <= EPS_GEOMETRIC:
            raise ValueError("DAMPER_ROCKER must not lie on the rocker axis")

    def constraints(
        self,
        initial: SuspensionState,
        actuation: Actuation,
    ) -> list[Constraint]:
        """Attach the moving damper pickup rigidly to the rocker."""
        if not isinstance(actuation, ActuationPushrodRocker):
            raise ValueError(
                "A linear inboard damper requires pushrod-rocker actuation"
            )
        return actuation.linear_link_constraints(initial, PointID.DAMPER_ROCKER)

    def derivative_metric_definitions(
        self,
        initial: SuspensionState,
        actuation: Actuation,
        side: Side,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare independent damper length relative to hub travel."""
        return (
            DerivativeMetricDefinition(
                response=PointDistanceResponse(
                    PointID.DAMPER_CHASSIS,
                    PointID.DAMPER_ROCKER,
                    name="damper_length",
                    unit=MetricUnit.MM,
                    label="Damper Length",
                ),
                driver=PointCoordinateResponse.from_chassis_axis(
                    PointID.WHEEL_CENTER,
                    Axis.Z,
                    name="hub_z",
                    unit=MetricUnit.MM,
                    label="Hub Z",
                ),
            ),
        )

    def elements(self, actuation: Actuation) -> tuple[SuspensionElement, ...]:
        """Return the physical independent damper link."""
        return (
            VariableLengthLinkElement(
                label="Damper",
                type=ElementType.DAMPER,
                point_a=PointID.DAMPER_CHASSIS,
                point_b=PointID.DAMPER_ROCKER,
            ),
        )


type CornerDamper = CornerDamperNone | CornerDamperLinear


def validate_composed_mechanisms(
    hardpoints: Mapping[PointKey, Point3],
    actuation: Actuation,
    spring: CornerSpring,
    damper: CornerDamper,
) -> None:
    """Validate installed mechanisms and their cross-mechanism compatibility."""
    actuation.validate(hardpoints)
    spring.validate(actuation)
    if spring.damper_points is not None and damper.damper_points is not None:
        raise ValueError("A separate linear damper cannot be combined with a coilover")
    damper.validate(actuation, hardpoints)


def composed_mechanism_free_points(
    actuation: Actuation,
    spring: CornerSpring,
    damper: CornerDamper,
) -> tuple[PointID, ...]:
    """Return mechanism solver points, excluding derived spring pickups."""
    return (
        *actuation.free_points,
        *(
            point
            for point in spring.free_points
            if point not in actuation.derived_pickup_points
        ),
        *damper.free_points,
    )


def composed_derivative_metric_definitions(
    initial: SuspensionState,
    side: Side,
    actuation: Actuation,
    spring: CornerSpring,
    damper: CornerDamper,
) -> tuple[DerivativeMetricDefinition, ...]:
    """Compose derivative declarations from the installed mechanisms."""
    return (
        *actuation.derivative_metric_definitions(initial, side),
        *spring.derivative_metric_definitions(initial, actuation, side),
        *damper.derivative_metric_definitions(initial, actuation, side),
    )


def composed_topology_metric_values(
    state: SuspensionState,
    initial: SuspensionState,
    side: Side,
    actuation: Actuation,
    spring: CornerSpring,
) -> "MetricRow":
    """Compose state metric values from the installed mechanisms."""
    row: MetricRow = OrderedDict()
    row.update(actuation.topology_metric_values(state, initial, side))
    row.update(spring.topology_metric_values(state, initial, actuation, side))
    return row


def composed_mechanism_elements(
    actuation: Actuation,
    spring: CornerSpring,
    damper: CornerDamper,
) -> tuple[SuspensionElement, ...]:
    """Return physical elements declared by the installed mechanisms."""
    return (
        *actuation.elements(damper.rocker_pickups),
        *spring.elements(actuation),
        *damper.elements(actuation),
    )


def composed_output_points(
    base_points: tuple[PointKey, ...],
    actuation: Actuation,
    spring: CornerSpring,
    damper: CornerDamper,
) -> tuple[PointKey, ...]:
    """Return base and installed mechanism output points, first-seen order."""
    return tuple(
        dict.fromkeys(
            (
                *base_points,
                *actuation.output_points,
                *spring.output_points,
                *damper.output_points,
            )
        )
    )


def composed_upright_attachments(
    base_attachments: tuple[PointID, ...],
    actuation: Actuation,
    upright_body: tuple[PointID, ...],
) -> tuple[PointID, ...]:
    """Return upright-carried points, including an upright-mounted pickup."""
    if actuation.moving_pickup_body == upright_body:
        return (*base_attachments, actuation.moving_pickup_point)
    return base_attachments
