"""
Metric computation context.

Provides a single per-state object that resolves and caches shared geometry
needed by multiple metric functions (wheel axis, wheel-ground tangent, ICs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from kinematics.core.enums import PointID
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
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
    rather than assumed from any one architecture's point naming.
    """

    state: SuspensionState
    suspension: "CornerSuspension"
    config: SuspensionConfig
    ground: GroundDatum

    def __init__(
        self,
        state: SuspensionState,
        suspension: "CornerSuspension",
        config: SuspensionConfig,
        ground: GroundDatum | None = None,
    ) -> None:
        """Resolve one ground datum when the metric context is created."""
        self.state = state
        self.suspension = suspension
        self.config = config
        self.ground = (
            ground
            if ground is not None
            else GroundDatum.horizontal_at(
                state.get(PointID.WHEEL_GROUND_TANGENT)
            )
        )

    @cached_property
    def design_state(self) -> SuspensionState:
        """Return the as-authored state used as the travel reference."""
        return self.suspension.initial_state()

    @cached_property
    def design_wheel_center(self) -> Point3:
        """Wheel-center position at the design condition."""
        return self.design_state.get(PointID.WHEEL_CENTER)

    @cached_property
    def design_wheel_ground_tangent(self) -> Point3:
        """Wheel-ground tangent position at the design condition."""
        return self.design_state.get(PointID.WHEEL_GROUND_TANGENT)

    @cached_property
    def side_view_ic(self) -> Point3 | None:
        """
        Side-view instant center from the suspension.
        """
        return self.suspension.compute_side_view_instant_center(self.state)

    @cached_property
    def front_view_ic(self) -> Point3 | None:
        """
        Front-view instant center from the suspension.
        """
        return self.suspension.compute_front_view_instant_center(self.state)

    @cached_property
    def wheel_center(self) -> Point3:
        """
        Wheel center position.
        """
        return self.state.get(PointID.WHEEL_CENTER)

    @cached_property
    def wheel_ground_tangent(self) -> Point3:
        """Current wheel-ground tangent position."""
        return self.state.get(PointID.WHEEL_GROUND_TANGENT)

    @cached_property
    def wheel_axis(self) -> Direction3:
        """
        Unit vector along the wheel spin axis from inboard to outboard.
        """
        inboard_id, outboard_id = self.suspension.wheel_axis_points()
        axle_in = self.state.get(inboard_id)
        axle_out = self.state.get(outboard_id)
        return (axle_out - axle_in).normalize()

    @cached_property
    def steering_axis_pivots(self) -> tuple[Point3, Point3]:
        """
        Steering-axis pivot positions as (lower, upper).
        """
        lower_id, upper_id = self.suspension.steering_axis_points()
        return (self.state.get(lower_id), self.state.get(upper_id))

    @cached_property
    def steering_axis(self) -> Direction3:
        """
        Unit vector along the steering axis from lower to upper pivot.
        """
        lower, upper = self.steering_axis_pivots
        return (upper - lower).normalize()

    @cached_property
    def steering_axis_ground_intersection(self) -> Point3 | None:
        """
        Point where the steering axis intersects the ground plane.

        Parameterises the line from the lower steering pivot through the upper
        pivot and solves ``n · (lower + t * direction) + c = 0`` against the
        actual ground plane.
        Returns None if the steering axis is parallel to the ground plane.
        """
        lower, upper = self.steering_axis_pivots
        direction = upper - lower
        normal = self.ground.normal
        denominator = normal.dot(direction)
        if abs(denominator) < EPS_GEOMETRIC:
            return None
        t = -self.ground.signed_distance(lower) / denominator
        return lower + t * direction

    @cached_property
    def side_sign(self) -> float:
        """
        Explicit vehicle-side sign: 1.0 for left, -1.0 for right.
        """
        return self.suspension.side.lateral_sign

    @cached_property
    def tire_radius(self) -> float:
        """
        Nominal tire radius from configuration.
        """
        return self.config.wheel.tire.nominal_radius

    @cached_property
    def wheelbase(self) -> float:
        """
        Vehicle wheelbase from configuration.
        """
        return self.config.wheelbase

    @cached_property
    def cg_position(self) -> Point3:
        """
        Center of gravity position from configuration.
        """
        return self.config.cg_position.copy()
