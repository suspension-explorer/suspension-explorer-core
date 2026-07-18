"""
Double-wishbone corner suspension implementation.

This module defines the stable double-wishbone locating architecture. Installed
actuation and spring behavior is composed through typed mechanism fields.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Sequence, cast

from kinematics.core.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
)
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    UprightElement,
    WheelElement,
)
from kinematics.core.enums import (
    Axis,
    MountBody,
    PointID,
    ShimType,
    SteeringType,
    SuspensionType,
)
from kinematics.core.points.derived.definitions import build_wheel_derived_spec
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
    rotate_point_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.config.shims import (
    CamberShimRockerCoupling,
    solve_camber_shim_assembly,
)
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.suspensions.corner.mechanisms import (
    Actuation,
    ActuationDirect,
    ActuationPushrodRocker,
    CornerSpring,
    CornerSpringNone,
)
from kinematics.core.suspensions.corner.toe_link import ToeLink
from kinematics.core.suspensions.corner.track_rod import TrackRod

if TYPE_CHECKING:
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.metrics.registry import MetricSpec


@dataclass
class DoubleWishboneSuspension(CornerSuspension):
    """Double-wishbone locating geometry with composed corner mechanisms."""

    TYPE_KEY: ClassVar[SuspensionType] = SuspensionType.DOUBLE_WISHBONE
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            PointID.UPPER_WISHBONE_INBOARD_REAR,
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )

    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()

    # Rigid bodies that composed mechanisms may attach to. The architecture
    # owns these; mechanisms receive them at construction and stay free of
    # double-wishbone point names. MOUNT_BODIES maps the user-selectable mount
    # identifiers to those bodies so the geometry spec can choose one.
    LOWER_WISHBONE_BODY: ClassVar[tuple[PointID, PointID, PointID]] = (
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
    )
    UPRIGHT_BODY: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    UPRIGHT_ATTACHMENTS: ClassVar[tuple[PointID, ...]] = (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    MOUNT_BODIES: ClassVar[dict[MountBody, tuple[PointID, ...]]] = {
        MountBody.LOWER_WISHBONE: LOWER_WISHBONE_BODY,
        MountBody.UPRIGHT: UPRIGHT_BODY,
    }

    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset(
        {ShimType.OUTBOARD_CAMBER}
    )

    # Points included in solver output (CSV/Parquet), in column order.
    # Hardpoints first, then derived points.
    LOCATING_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_INBOARD_REAR,
        PointID.UPPER_WISHBONE_OUTBOARD,
    )
    WHEEL_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.AXLE_MIDPOINT,
        PointID.WHEEL_CENTER,
        PointID.WHEEL_INBOARD,
        PointID.WHEEL_OUTBOARD,
        PointID.CONTACT_PATCH_CENTER,
    )
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        *LOCATING_OUTPUT_POINTS,
        *WHEEL_OUTPUT_POINTS,
    )

    # Free points that move during solving.
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )

    wheel_heading_link: TrackRod | ToeLink = field(init=False)
    actuation: Actuation = field(
        default_factory=lambda: ActuationDirect(
            spring_pickup_body=DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        ),
        kw_only=True,
    )
    spring: CornerSpring = field(default_factory=CornerSpringNone, kw_only=True)

    def __post_init__(self) -> None:
        """Install a track rod or fixed toe link for wheel-heading control."""
        if self.config is None:
            raise ValueError("Double-wishbone suspension requires configuration")
        # The four upright anchors already overdetermine this attachment, while
        # the upright angle constraint preserves the authored assembly branch.
        if self.config.steering.type is SteeringType.RACK:
            self.wheel_heading_link = TrackRod(
                self.UPRIGHT_BODY,
                preserve_attachment_handedness=False,
            )
        else:
            self.wheel_heading_link = ToeLink(
                self.UPRIGHT_BODY,
                preserve_attachment_handedness=False,
            )
        super().__post_init__()

    def required_points(self) -> frozenset[PointID]:
        """Return base and selected mechanism point requirements."""
        return (
            self.REQUIRED_POINTS
            | self.wheel_heading_link.REQUIRED_POINTS
            | self.actuation.required_points
            | self.spring.required_points
        )

    def validate_hardpoints(self) -> None:
        """Validate base geometry and selected mechanism compatibility."""
        super().validate_hardpoints()
        self.wheel_heading_link.validate(self.hardpoints)
        self.actuation.validate(self.hardpoints)
        self.spring.validate(self.actuation)

    def free_points(self) -> Sequence[PointID]:
        """Return base and selected mechanism moving points."""
        return (
            *self.FREE_POINTS,
            *self.wheel_heading_link.free_points,
            *self.actuation.free_points,
            *self.spring.free_points,
        )

    def output_points(self) -> tuple[PointKey, ...]:
        """Return base and selected mechanism output points."""
        return tuple(
            dict.fromkeys(
                (
                    *self.LOCATING_OUTPUT_POINTS,
                    *self.wheel_heading_link.OUTPUT_POINTS,
                    *self.WHEEL_OUTPUT_POINTS,
                    *self.actuation.output_points,
                    *self.spring.output_points,
                )
            )
        )

    def damper_points(self) -> tuple[PointKey, PointKey] | None:
        """Return selected linear spring/damper endpoints."""
        return self.spring.damper_points

    def steering_axis_points(self) -> tuple[PointID, PointID]:
        """The steering axis runs between the two outboard ball joints."""
        return (PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD)

    def rack_attachment_point(self) -> PointID | None:
        """Return the track-rod rack pickup for a steered corner."""
        if isinstance(self.wheel_heading_link, TrackRod):
            return self.wheel_heading_link.inboard_point
        return None

    def initial_state(self) -> SuspensionState:
        """Build initial state from hardpoints, applying shims if configured."""
        if self._initial_state is not None:
            return self._initial_state

        positions = self.get_hardpoints_copy()

        # Get camber shim point positions.

        # Apply camber shim if configured.
        if self.config is not None and self.config.camber_shim is not None:
            self.apply_camber_shim(positions)

        # Compute derived points.
        derived_spec = self.derived_spec()
        derived_manager = DerivedPointsManager(derived_spec)
        derived_manager.update_in_place(positions)

        # get_hardpoints_copy widens keys to PointKey, so key the state on
        # PointKey too even though a corner only ever holds PointID keys.
        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set[PointKey](self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build geometric constraints for double wishbone."""
        initial_state = self.initial_state()
        constraints: list[Constraint] = []

        # Distance constraints (link lengths).
        length_pairs = [
            (PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
        ]

        for p1, p2 in length_pairs:
            target_distance = compute_point_point_distance(
                initial_state.positions[p1], initial_state.positions[p2]
            )
            constraints.append(DistanceConstraint(p1, p2, target_distance))

        # Angle constraint for upright rigidity.
        v1 = (
            initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]
            - initial_state.positions[PointID.UPPER_WISHBONE_OUTBOARD]
        )
        v2 = (
            initial_state.positions[PointID.AXLE_OUTBOARD]
            - initial_state.positions[PointID.AXLE_INBOARD]
        )
        target_angle = compute_vector_vector_angle(v1, v2)

        constraints.append(
            AngleConstraint(
                v1_start=PointID.UPPER_WISHBONE_OUTBOARD,
                v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
                v2_start=PointID.AXLE_INBOARD,
                v2_end=PointID.AXLE_OUTBOARD,
                target_angle=target_angle,
            )
        )

        constraints.extend(self.wheel_heading_link.constraints(initial_state))
        constraints.extend(self.actuation.constraints(initial_state))
        constraints.extend(self.spring.constraints(initial_state, self.actuation))
        return constraints

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Compose derivative declarations from actuation and spring mechanisms."""
        initial = self.initial_state()
        return (
            *self.actuation.derivative_metric_definitions(initial, self.side),
            *self.spring.derivative_metric_definitions(
                initial,
                self.actuation,
                self.side,
            ),
        )

    def topology_metric_values(self, state: SuspensionState) -> MetricRow:
        """Compose state metrics from actuation and spring mechanisms."""
        initial = self.initial_state()
        row: MetricRow = OrderedDict()
        row.update(self.actuation.topology_metric_values(state, initial, self.side))
        row.update(
            self.spring.topology_metric_values(
                state,
                initial,
                self.actuation,
                self.side,
            )
        )
        return row

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Compose state metric metadata from installed corner mechanisms."""
        return (
            *self.actuation.topology_metric_specs(),
            *self.spring.topology_metric_specs(),
        )

    def derived_spec(self) -> DerivedPointsSpec:
        """Standard wheel derived points from the axle pair."""
        if self.config is None:
            raise ValueError("Cannot compute derived spec without config")
        return build_wheel_derived_spec(self.config.wheel)

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """
        Compute side view instant center from wishbone planes.

        The SVIC is found by intersecting the 3D instant axis with the
        vertical plane that passes through the wheel center's Y position.
        Returns None when the instant axis is undefined or runs parallel
        to that side-view plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            raise

        wheel_center_y = float(state.positions[PointID.WHEEL_CENTER][1])

        if instant_axis is None:
            return None

        axis_point, axis_direction = instant_axis
        svic = intersect_line_with_vertical_plane(
            axis_point, axis_direction, wheel_center_y
        )

        return svic

    def compute_instant_axis(
        self, state: SuspensionState
    ) -> tuple[Point3, Direction3] | None:
        """Compute 3D instant axis from wishbone planes intersection."""
        upper_front = state.positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
        upper_rear = state.positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
        upper_outboard = state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

        lower_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        lower_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        lower_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        upper_plane = plane_from_three_points(upper_front, upper_rear, upper_outboard)
        lower_plane = plane_from_three_points(lower_front, lower_rear, lower_outboard)

        if upper_plane is None or lower_plane is None:
            raise ValueError(
                "Degenerate wishbone geometry. Cannot compute instant axis."
            )

        return intersect_two_planes(
            n1=upper_plane[0],
            d1=upper_plane[1],
            n2=lower_plane[0],
            d2=lower_plane[1],
        )

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """
        Compute front view instant center from wishbone planes.

        The FVIC is found by intersecting the 3D instant axis (the line
        where the upper and lower wishbone planes meet) with the front-view
        plane at the wheel center's X station. Using the wheel station keeps
        the front-view geometry invariant to rigid longitudinal translations
        of the entire corner. Returns None if the links are parallel or the
        instant axis runs parallel to that front-view plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            raise

        if instant_axis is None:
            return None

        axis_point, axis_direction = instant_axis
        wheel_center_x = float(state.positions[PointID.WHEEL_CENTER][Axis.X])

        return intersect_line_with_axis_aligned_plane(
            axis_point, axis_direction, Axis.X, wheel_center_x
        )

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the physical elements in this corner."""
        heading_link_outboard = self.wheel_heading_link.outboard_point
        base_elements: tuple[SuspensionElement, ...] = (
            RigidLinkElement(
                label="Upper Wishbone Front Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.UPPER_WISHBONE_INBOARD_FRONT,
                point_b=PointID.UPPER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Upper Wishbone Rear Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.UPPER_WISHBONE_INBOARD_REAR,
                point_b=PointID.UPPER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Lower Wishbone Front Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_FRONT,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Lower Wishbone Rear Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_REAR,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            UprightElement(
                label="Upright",
                hardpoints=(
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    heading_link_outboard,
                ),
                attachments=(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
                segments=(
                    (heading_link_outboard, PointID.UPPER_WISHBONE_OUTBOARD),
                    (
                        PointID.UPPER_WISHBONE_OUTBOARD,
                        PointID.LOWER_WISHBONE_OUTBOARD,
                    ),
                    (PointID.LOWER_WISHBONE_OUTBOARD, heading_link_outboard),
                ),
            ),
            RigidLinkElement(
                label="Axle",
                type=ElementType.AXLE,
                point_a=PointID.AXLE_INBOARD,
                point_b=PointID.AXLE_OUTBOARD,
            ),
            WheelElement(
                label="Wheel",
                center=PointID.WHEEL_CENTER,
                inboard=PointID.WHEEL_INBOARD,
                outboard=PointID.WHEEL_OUTBOARD,
                axle_inboard=PointID.AXLE_INBOARD,
                axle_outboard=PointID.AXLE_OUTBOARD,
                contact_patch=PointID.CONTACT_PATCH_CENTER,
            ),
        )
        return (
            *base_elements,
            *self.wheel_heading_link.elements(),
            *self.actuation.elements(),
            *self.spring.elements(self.actuation),
        )

    def apply_camber_shim(self, positions: dict[PointKey, Point3]) -> None:
        """
        Apply camber shim transformation to the suspension geometry.

        Solves the local split-body shim assembly to find how the camber block
        and upright body rotate when the shim thickness changes. Then:
        1. Writes the solved UBJ position back (it moves along the upper wishbone arc).
        2. Rotates upright attachments about the fixed LBJ using the solved
           upright-body rotation.
        3. Rotates the rocker group by the solved angle when the pushrod is
           upright-mounted.
        4. Leaves all chassis-mounted points unchanged.
        """
        if self.config is None or self.config.camber_shim is None:
            return

        shim_config = self.config.camber_shim
        rocker_actuation = (
            self.actuation
            if isinstance(self.actuation, ActuationPushrodRocker)
            and self.actuation.moving_pickup_body == self.UPRIGHT_BODY
            else None
        )
        rocker_coupling = (
            CamberShimRockerCoupling(
                axis_a=PointID.ROCKER_AXIS_A,
                axis_b=PointID.ROCKER_AXIS_B,
                pushrod_inboard=PointID.PUSHROD_INBOARD,
                pushrod_outboard=PointID.PUSHROD_OUTBOARD,
            )
            if rocker_actuation is not None
            else None
        )

        # Shim face geometry is read directly from shim_config by the solver,
        # so the positions dict only needs the kinematic hardpoints.
        # A corner keys strictly on PointID; the solver only reads hardpoints.
        assembly_solution = solve_camber_shim_assembly(
            positions=cast("dict[PointID, Point3]", positions),
            shim_config=shim_config,
            heading_link_inboard=self.wheel_heading_link.inboard_point,
            heading_link_outboard=self.wheel_heading_link.outboard_point,
            rocker_coupling=rocker_coupling,
        )

        # Write the solved UBJ position back. The upper wishbone arc constraint
        # means UBJ may shift slightly to accommodate the new shim thickness.
        positions[PointID.UPPER_WISHBONE_OUTBOARD] = Point3(
            assembly_solution.ubj_position
        )

        # Rotate each upright attachment about LBJ using the solved upright-body
        # rotation axis and angle.
        if assembly_solution.upright_body_rot_angle_rad > EPS_GEOMETRIC:
            lbj = positions[PointID.LOWER_WISHBONE_OUTBOARD]
            rot_axis = Direction3(assembly_solution.upright_body_rot_axis)
            for point_id in self.upright_attachment_points():
                if point_id in positions:
                    positions[point_id] = rotate_point_about_axis(
                        positions[point_id],
                        lbj,
                        rot_axis,
                        assembly_solution.upright_body_rot_angle_rad,
                    )

        if rocker_actuation is not None:
            rocker_actuation.rotate_rocker_group(
                positions,
                assembly_solution.rocker_angle_rad,
                self.spring.rocker_mounted_points,
            )

    def upright_attachment_points(self) -> tuple[PointID, ...]:
        """Return points carried by the upright during camber-shim setup."""
        base_attachments = (
            *self.UPRIGHT_ATTACHMENTS,
            self.wheel_heading_link.outboard_point,
        )
        if self.actuation.moving_pickup_body == self.UPRIGHT_BODY:
            return (*base_attachments, self.actuation.moving_pickup_point)
        return base_attachments
