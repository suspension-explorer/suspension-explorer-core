"""
Steering axis geometry metrics.

Scrub radius and mechanical trail are measured from the point where the
steering axis (kingpin axis) intersects the local ground reference plane
through the wheel-ground tangent point to that point itself.

Coordinate System Assumption: ISO 8855 (X-Forward, Y-Left, Z-Up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis
from kinematics.core.primitives.constants import EPS_GEOMETRIC

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_scrub_radius(ctx: MetricContext) -> float | None:
    """
    Scrub radius in mm.

    The distance from the steering axis ground intersection to the
    wheel-ground tangent point, measured along the wheel's lateral direction
    in the ground plane. Projecting the wheel axis into the ground
    plane keeps the measurement correct when the wheel is both steered
    and cambered.

    Positive scrub radius means the steering axis meets the ground
    inboard of the ground tangent (the common case for a double-
    wishbone layout with positive KPI).

    The steering-axis intersection and lateral direction are evaluated on the
    actual ground plane.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    ground_tangent = ctx.wheel_ground_tangent
    # Scrub radius lives in the ground plane, so remove the wheel axis component
    # normal to that plane before measuring the lateral offset.
    # Negate so that positive scrub means the ground intersection is
    # inboard of the ground tangent. The projected wheel axis already
    # encodes left/right handedness, so no explicit side_sign is needed.
    displacement = ground_pt - ground_tangent
    ground_normal, _ = ctx.ground_plane
    projected_axis = (
        ctx.wheel_axis.vector()
        - ground_normal * ctx.wheel_axis.dot(ground_normal)
    )
    if projected_axis.norm() < EPS_GEOMETRIC:
        return None
    wheel_lateral_ground = projected_axis.normalize()
    return -float(displacement.dot(wheel_lateral_ground))


def calculate_mechanical_trail(ctx: MetricContext) -> float | None:
    """
    Mechanical trail (caster trail) in mm.

    The longitudinal (X-axis) distance from the steering axis ground
    intersection to the wheel-ground tangent point. Positive mechanical
    trail means the ground tangent is behind (rearward of) the steering
    axis ground intersection, which produces a self-centring moment.

    The steering-axis intersection is evaluated on the actual ground plane.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    ground_tangent = ctx.wheel_ground_tangent
    # Positive when the ground tangent is behind the ground intersection.
    return float(ground_pt[Axis.X] - ground_tangent[Axis.X])
