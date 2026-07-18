"""
MacPherson strut corner suspension implementation.

The strut is modelled without a dedicated prismatic constraint, mirroring the
rigid-upright approach used by the double wishbone:

- The lower arm locates the lower ball joint on its arc.
- Modelling choice: the strut axis is taken to be coincident with the
  steering axis, which runs from the moving lower ball joint to the fixed
  strut top mount. Real struts often sit a few millimetres off the steering
  axis to reduce spring side load; this model deliberately ignores that
  offset. The authored strut clamp point (STRUT_BOTTOM) must therefore lie
  on the ball-joint-to-top-mount line at design, with only small authoring
  deviations tolerated. Its authored distance from the ball joint is retained.
- The strut clamp is derived at that fixed distance along the line from the
  lower ball joint to the fixed top mount. The upright is held rigidly to the
  derived clamp, while the unconstrained clamp-to-top distance is the
  telescoping strut degree of freedom, presented as one variable-length link.
- The installed track rod or toe link resolves rotation about the steering axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Sequence

from kinematics.core.constraints import (
    Constraint,
    DistanceConstraint,
)
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    UprightElement,
    VariableLengthLinkElement,
    WheelElement,
)
from kinematics.core.enums import Axis, PointID, SteeringType, SuspensionType
from kinematics.core.metrics.derivatives import (
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.definitions import (
    build_wheel_derived_spec,
    get_point_along_line,
)
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    chiral_rigid_point_constraints,
)
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.suspensions.corner.toe_link import ToeLink
from kinematics.core.suspensions.corner.track_rod import TrackRod

# How far the authored strut clamp may sit off the design steering axis, in
# millimetres, before the coincident-axis modelling choice is considered
# violated rather than an authoring rounding error. Within the tolerance the
# clamp is projected exactly onto the axis; beyond it we refuse the geometry
# instead of silently reshaping an intentionally offset strut.
STRUT_AXIS_ALIGNMENT_TOLERANCE_MM = 1.0


@dataclass
class MacPhersonSuspension(CornerSuspension):
    """MacPherson strut with a selected track rod or fixed toe link."""

    TYPE_KEY: ClassVar[SuspensionType] = SuspensionType.MACPHERSON
    UPRIGHT_BODY: ClassVar[tuple[PointID, PointID, PointID]] = (
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.STRUT_TOP,
            PointID.STRUT_BOTTOM,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )

    # Points included in solver output, hardpoints first, then derived.
    LOCATING_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
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

    # Free points that move during solving. The strut top is a fixed chassis
    # mount and the strut clamp is derived from the lower ball joint and top.
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )

    wheel_heading_link: TrackRod | ToeLink = field(init=False)

    def __post_init__(self) -> None:
        """Install a track rod or fixed toe link for wheel-heading control."""
        if self.config is None:
            raise ValueError("MacPherson suspension requires configuration")
        if self.config.steering.type is SteeringType.RACK:
            self.wheel_heading_link = TrackRod(self.UPRIGHT_BODY)
        else:
            self.wheel_heading_link = ToeLink(self.UPRIGHT_BODY)
        super().__post_init__()

    def required_points(self) -> frozenset[PointID]:
        """Return the strut and selected heading-link point requirements."""
        return self.REQUIRED_POINTS | self.wheel_heading_link.REQUIRED_POINTS

    def output_points(self) -> tuple[PointKey, ...]:
        """Return authored, heading-link, and derived strut points."""
        return tuple(
            dict.fromkeys(
                (
                    *self.LOCATING_OUTPUT_POINTS,
                    *self.wheel_heading_link.OUTPUT_POINTS,
                    *self.WHEEL_OUTPUT_POINTS,
                )
            )
        )

    def validate_hardpoints(self) -> None:
        """Validate the point set and the strut axis definition."""
        super().validate_hardpoints()
        self.wheel_heading_link.validate(self.hardpoints)
        ball_joint = self.hardpoints[PointID.LOWER_WISHBONE_OUTBOARD]
        strut_top = self.hardpoints[PointID.STRUT_TOP]
        axis_length = compute_point_point_distance(ball_joint, strut_top)
        if axis_length <= EPS_GEOMETRIC:
            raise ValueError(
                "STRUT_TOP must not coincide with LOWER_WISHBONE_OUTBOARD; "
                "the steering axis would be undefined."
            )
        # This model chooses to treat the strut axis as coincident with the
        # steering axis, so the authored clamp must sit on the
        # ball-joint-to-top line to within authoring tolerance.
        clamp_offset = compute_point_to_line_distance(
            self.hardpoints[PointID.STRUT_BOTTOM],
            ball_joint,
            (strut_top - ball_joint).normalize(),
        )
        if clamp_offset > STRUT_AXIS_ALIGNMENT_TOLERANCE_MM:
            raise ValueError(
                f"STRUT_BOTTOM sits {clamp_offset:.3f} mm off the line from "
                "LOWER_WISHBONE_OUTBOARD to STRUT_TOP. This model treats the "
                "strut axis as coincident with the steering axis; an "
                "intentionally offset strut is not supported."
            )

        axial_offset = self._strut_clamp_offset()
        if axial_offset <= EPS_GEOMETRIC or axial_offset >= (
            axis_length - EPS_GEOMETRIC
        ):
            raise ValueError(
                "STRUT_BOTTOM must lie between LOWER_WISHBONE_OUTBOARD and "
                "STRUT_TOP along the strut axis"
            )

    def _strut_clamp_offset(self) -> float:
        """Return the authored ball-joint-to-clamp distance along the strut axis."""
        ball_joint = self.hardpoints[PointID.LOWER_WISHBONE_OUTBOARD]
        strut_axis = (self.hardpoints[PointID.STRUT_TOP] - ball_joint).normalize()
        clamp_vector = self.hardpoints[PointID.STRUT_BOTTOM] - ball_joint
        return float(clamp_vector.data.dot(strut_axis.data))

    def free_points(self) -> Sequence[PointID]:
        """Return the moving upright group and heading-link points."""
        return (*self.FREE_POINTS, *self.wheel_heading_link.free_points)

    def steering_axis_points(self) -> tuple[PointID, PointID]:
        """The steering axis runs from the lower ball joint to the strut top."""
        return (PointID.LOWER_WISHBONE_OUTBOARD, PointID.STRUT_TOP)

    def rack_attachment_point(self) -> PointID | None:
        """Return the track-rod rack pickup for a steered corner."""
        if isinstance(self.wheel_heading_link, TrackRod):
            return self.wheel_heading_link.inboard_point
        return None

    def damper_points(self) -> tuple[PointKey, PointKey] | None:
        """The strut is the spring/damper: top mount to upright clamp."""
        return (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare strut length relative to hub vertical travel."""
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

    def initial_state(self) -> SuspensionState:
        """Build the initial state from hardpoints plus derived points."""
        if self._initial_state is not None:
            return self._initial_state

        positions = self.get_hardpoints_copy()
        derived_manager = DerivedPointsManager(self.derived_spec())
        derived_manager.update_in_place(positions)

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set[PointKey](self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build lower-arm, upright, strut, and heading-link constraints."""
        initial_state = self.initial_state()
        positions = initial_state.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        # Lower arm legs locate the ball joint, and the rigid upright group
        # (ball joint, axle pair, heading-link outboard) mirrors the
        # double-wishbone upright treatment.
        length_pairs = [
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
        ]
        constraints: list[Constraint] = [
            distance(point_a, point_b) for point_a, point_b in length_pairs
        ]

        # The derived strut clamp lies on the lower-balljoint-to-top line at a
        # fixed offset from the ball joint. Holding the upright to that datum
        # leaves the clamp-to-top distance free to telescope.
        constraints.extend(
            chiral_rigid_point_constraints(
                initial_state,
                PointID.STRUT_BOTTOM,
                self.UPRIGHT_BODY,
            )
        )

        constraints.extend(self.wheel_heading_link.constraints(initial_state))
        return constraints

    def derived_spec(self) -> DerivedPointsSpec:
        """Derived strut clamp and standard wheel points."""
        if self.config is None:
            raise ValueError("Cannot compute derived spec without config")
        wheel_spec = build_wheel_derived_spec(self.config.wheel)
        functions = {
            PointID.STRUT_BOTTOM: partial(
                get_point_along_line,
                start_point=PointID.LOWER_WISHBONE_OUTBOARD,
                end_point=PointID.STRUT_TOP,
                distance_from_start=self._strut_clamp_offset(),
            ),
            **wheel_spec.functions,
        }
        dependencies = {
            PointID.STRUT_BOTTOM: {
                PointID.LOWER_WISHBONE_OUTBOARD,
                PointID.STRUT_TOP,
            },
            **wheel_spec.dependencies,
        }
        return DerivedPointsSpec(functions, dependencies)

    def compute_instant_axis(
        self, state: SuspensionState
    ) -> tuple[Point3, Direction3] | None:
        """
        Compute the 3D instant axis of the upright.

        The lower ball joint moves in the lower arm plane, and the upright
        point at the strut top mount can only translate along the strut
        axis, so the instant axis lies in the plane through the strut top
        perpendicular to the strut axis. The axis is the intersection of
        those two planes.
        """
        arm_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        arm_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        ball_joint = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]
        strut_top = state.positions[PointID.STRUT_TOP]

        arm_plane = plane_from_three_points(arm_front, arm_rear, ball_joint)
        if arm_plane is None:
            raise ValueError("Degenerate lower arm geometry. Cannot compute axis.")

        strut_axis = (strut_top - ball_joint).normalize()
        # Plane in the n.x + d = 0 form through the strut top.
        strut_plane_offset = -float(strut_axis.data.dot(strut_top.data))

        return intersect_two_planes(
            n1=arm_plane[0],
            d1=arm_plane[1],
            n2=strut_axis,
            d2=strut_plane_offset,
        )

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """Intersect the instant axis with the wheel center's side-view plane."""
        instant_axis = self.compute_instant_axis(state)
        if instant_axis is None:
            return None
        axis_point, axis_direction = instant_axis
        wheel_center_y = float(state.positions[PointID.WHEEL_CENTER][Axis.Y])
        return intersect_line_with_vertical_plane(
            axis_point, axis_direction, wheel_center_y
        )

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """Intersect the instant axis with the wheel center's front-view plane."""
        instant_axis = self.compute_instant_axis(state)
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
        return (
            RigidLinkElement(
                label="Lower Arm Front Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_FRONT,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Lower Arm Rear Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_REAR,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            VariableLengthLinkElement(
                label="Strut",
                type=ElementType.SPRING_DAMPER,
                point_a=PointID.STRUT_TOP,
                point_b=PointID.STRUT_BOTTOM,
            ),
            UprightElement(
                label="Upright",
                hardpoints=(
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    heading_link_outboard,
                    PointID.STRUT_BOTTOM,
                ),
                attachments=(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
                segments=(
                    (PointID.LOWER_WISHBONE_OUTBOARD, heading_link_outboard),
                    (PointID.LOWER_WISHBONE_OUTBOARD, PointID.STRUT_BOTTOM),
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
            *self.wheel_heading_link.elements(),
        )
