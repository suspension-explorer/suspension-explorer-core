"""
Steering axis geometry metrics.

Scrub radius and mechanical trail are measured from the point where the
steering axis (kingpin axis) intersects the local ground reference plane
through the contact patch center to the contact patch center itself.

Coordinate System Assumption: ISO 8855 (X-Forward, Y-Left, Z-Up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis
from kinematics.core.geometry import Vector3

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_scrub_radius(ctx: MetricContext) -> float | None:
    """
    Scrub radius in mm.

    The distance from the steering axis ground intersection to the
    contact patch center, measured along the wheel's lateral direction
    in the ground plane. Projecting the wheel axis into the ground
    plane keeps the measurement correct when the wheel is both steered
    and cambered.

    Positive scrub radius means the steering axis meets the ground
    inboard of the contact patch (the common case for a double-
    wishbone layout with positive KPI).

    The steering-axis intersection is evaluated on the horizontal plane
    at the contact patch Z-height, not at world Z = 0.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    cp = ctx.contact_patch_center
    # Scrub radius lives in the road plane, so remove the camber-driven
    # Z component from the wheel axis before measuring the lateral offset.
    # Negate so that positive scrub means the ground intersection is
    # inboard of the contact patch. The projected wheel axis already
    # encodes left/right handedness, so no explicit side_sign is needed.
    displacement = ground_pt - cp
    wheel_lateral_ground = Vector3(
        [ctx.wheel_axis[Axis.X], ctx.wheel_axis[Axis.Y], 0.0]
    ).normalize()
    return -float(displacement.dot(wheel_lateral_ground))


def calculate_mechanical_trail(ctx: MetricContext) -> float | None:
    """
    Mechanical trail (caster trail) in mm.

    The longitudinal (X-axis) distance from the steering axis ground
    intersection to the contact patch center. Positive mechanical
    trail means the contact patch is behind (rearward of) the steering
    axis ground intersection, which produces a self-centring moment.

    The steering-axis intersection is evaluated on the horizontal plane
    at the contact patch Z-height, not at world Z = 0.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    cp = ctx.contact_patch_center
    # Positive when the contact patch is behind the ground intersection.
    return float(ground_pt[Axis.X] - cp[Axis.X])
