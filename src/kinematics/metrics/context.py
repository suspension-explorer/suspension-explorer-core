"""
Metric computation context.

Provides a single per-state object that resolves and caches shared geometry
needed by multiple metric functions (wheel axis, contact patch, ICs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Direction3, Point3
from kinematics.schema.config import SuspensionConfig
from kinematics.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.suspensions.base import Suspension


@dataclass
class MetricContext:
    """
    Shared context for computing metrics on a single solved state.

    Caches expensive geometry (ICs, wheel axis, etc.) so that multiple
    metric functions can share the same intermediate results.
    """

    state: SuspensionState
    suspension: "Suspension"
    config: SuspensionConfig

    @cached_property
    def design_state(self) -> SuspensionState:
        """Return the as-authored state used as the travel reference."""
        return self.suspension.initial_state()

    @cached_property
    def design_wheel_center(self) -> Point3:
        """Wheel-center position at the design condition."""
        return self.design_state.get(PointID.WHEEL_CENTER)

    @cached_property
    def design_contact_patch_center(self) -> Point3:
        """Contact-patch position at the design condition."""
        return self.design_state.get(PointID.CONTACT_PATCH_CENTER)

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
    def contact_patch_center(self) -> Point3:
        """
        Contact patch center position.
        """
        return self.state.get(PointID.CONTACT_PATCH_CENTER)

    @cached_property
    def wheel_axis(self) -> Direction3:
        """
        Unit vector along the axle from inboard to outboard.
        """
        axle_in = self.state.get(PointID.AXLE_INBOARD)
        axle_out = self.state.get(PointID.AXLE_OUTBOARD)
        return (axle_out - axle_in).normalize()

    @cached_property
    def steering_axis(self) -> Direction3:
        """
        Unit vector along the steering axis from lower to upper pivot.
        """
        lower = self.state.get(PointID.LOWER_WISHBONE_OUTBOARD)
        upper = self.state.get(PointID.UPPER_WISHBONE_OUTBOARD)
        return (upper - lower).normalize()

    @cached_property
    def ground_z(self) -> float:
        """
        Ground plane Z-height in the chassis-fixed frame.

        In a chassis-fixed reference frame the ground is not at Z=0; it
        follows the tire. We define ground level as the contact patch
        centre Z so that all ground-plane intersections (steering axis,
        instant centres, etc.) are evaluated at the actual tire-road
        interface.
        """
        return float(self.contact_patch_center[Axis.Z])

    @cached_property
    def steering_axis_ground_intersection(self) -> Point3 | None:
        """
        Point where the steering axis intersects the ground plane.

        Parameterises the line from the lower ball joint through the upper
        ball joint and solves for the parameter t where Z = ground_z.
        Returns None if the steering axis is parallel to the ground plane.
        """
        lower = self.state.get(PointID.LOWER_WISHBONE_OUTBOARD)
        upper = self.state.get(PointID.UPPER_WISHBONE_OUTBOARD)
        direction = upper - lower
        dz = direction[Axis.Z]
        if abs(dz) < EPS_GEOMETRIC:
            return None
        # t such that lower + t * direction has Z = ground_z
        t = (self.ground_z - lower[Axis.Z]) / dz
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
