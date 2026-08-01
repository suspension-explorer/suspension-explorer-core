"""
Metric computation context.

Provides a single per-state object that resolves and caches shared geometry
needed by multiple metric functions. Solved points and directions are stored
in chassis coordinates. The road datum is an ISO 8855 style local or
equivalent road plane expressed in those coordinates; it is not a world-space
vehicle-pose result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from kinematics.core.enums import PointID
from kinematics.core.metrics.steering_axis_geometry import SteeringAxis
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.road import RoadPlane
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.suspensions.corner.base import CornerSuspension


@dataclass(init=False)
class MetricContext:
    """
    Shared context for computing metrics on a single solved corner state.

    Caches expensive geometry (ICs, wheel axis, etc.) so that multiple
    metric functions can share the same intermediate results. Point roles
    (wheel axis, steering axis) are resolved through the corner's role hooks
    rather than assumed from any one architecture's point naming. All points
    and axes are expressed in chassis coordinates unless a property explicitly
    says that it is resolved against the local road plane.
    """

    state: SuspensionState
    suspension: "CornerSuspension"
    config: SuspensionConfig
    road: RoadPlane
    virtual_steering_axis: SteeringAxis | None

    def __init__(
        self,
        state: SuspensionState,
        suspension: "CornerSuspension",
        config: SuspensionConfig,
        road: RoadPlane | None = None,
        virtual_steering_axis: SteeringAxis | None = None,
    ) -> None:
        """Resolve one axle-local road datum in chassis coordinates."""
        self.state = state
        self.suspension = suspension
        self.config = config
        self.virtual_steering_axis = virtual_steering_axis
        self.road = (
            road
            if road is not None
            else RoadPlane.horizontal_at(state.get(PointID.WHEEL_CONTACT_CENTRE))
        )

    @cached_property
    def design_state(self) -> SuspensionState:
        """Return the as-authored chassis-space state used as a reference."""
        return self.suspension.initial_state()

    @cached_property
    def design_wheel_center(self) -> Point3:
        """Return the design wheel-centre position in chassis coordinates."""
        return self.design_state.get(PointID.WHEEL_CENTER)

    @cached_property
    def design_wheel_contact_centre(self) -> Point3:
        """Return the design wheel contact centre in chassis coordinates."""
        return self.design_state.get(PointID.WHEEL_CONTACT_CENTRE)

    @cached_property
    def side_view_ic(self) -> Point3 | None:
        """Return the side-view instant centre in chassis coordinates."""
        return self.suspension.compute_side_view_instant_center(self.state)

    @cached_property
    def front_view_ic(self) -> Point3 | None:
        """Return the front-view instant centre in chassis coordinates."""
        return self.suspension.compute_front_view_instant_center(self.state)

    @cached_property
    def wheel_center(self) -> Point3:
        """Return the current wheel-centre position in chassis coordinates."""
        return self.state.get(PointID.WHEEL_CENTER)

    @cached_property
    def wheel_contact_centre(self) -> Point3:
        """Return the current wheel contact centre in chassis coordinates."""
        return self.state.get(PointID.WHEEL_CONTACT_CENTRE)

    @cached_property
    def wheel_axis(self) -> Direction3:
        """
        Wheel spin direction from inboard to outboard in chassis coordinates.

        This is a wheel-defined axis represented in the chassis basis; it has
        not been projected into the road plane.
        """
        inboard_id, outboard_id = self.suspension.wheel_axis_points()
        axle_in = self.state.get(inboard_id)
        axle_out = self.state.get(outboard_id)
        return (axle_out - axle_in).normalize()

    @cached_property
    def steering_axis_pivots(self) -> tuple[Point3, Point3]:
        """Return lower and upper steering pivots in chassis coordinates."""
        lower_id, upper_id = self.suspension.steering_axis_points()
        return (self.state.get(lower_id), self.state.get(upper_id))

    @cached_property
    def steering_axis(self) -> Direction3:
        """Return lower-to-upper physical steering direction in chassis space."""
        return self.physical_steering_axis.direction

    @cached_property
    def physical_steering_axis(self) -> SteeringAxis:
        """Establish the physical steering axis from its resolved pivots."""
        lower, upper = self.steering_axis_pivots
        return SteeringAxis.from_pivots(lower, upper)

    @cached_property
    def steering_axis_ground_intersection(self) -> Point3 | None:
        """
        Intersect the chassis-space steering axis with the local road plane.

        Parameterises the line from the lower steering pivot through the upper
        pivot and solves ``n · (lower + t * direction) + c = 0`` against the
        ISO-style road datum, all expressed in chassis coordinates. This does
        not require world space or inferred chassis pitch. Returns None if the
        steering axis is parallel to the road plane.
        """
        return self.physical_steering_axis.intersect_road(self.road)

    @cached_property
    def side_sign(self) -> float:
        """Return the ISO vehicle-side sign: +1 left and -1 right."""
        return self.suspension.side.lateral_sign

    @cached_property
    def tire_radius(self) -> float:
        """Return the nominal tyre radius, independent of reference system."""
        return self.config.wheel.tire.nominal_radius

    @cached_property
    def wheelbase(self) -> float:
        """Return the configured longitudinal vehicle wheelbase."""
        return self.config.wheelbase

    @cached_property
    def cg_position(self) -> Point3:
        """Return the configured centre of gravity in chassis coordinates."""
        return self.config.cg_position.copy()
