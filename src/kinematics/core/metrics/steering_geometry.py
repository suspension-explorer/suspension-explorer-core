"""ISO-aligned steering geometry on the axle-local road plane.

The road datum is the ISO 8855 local or equivalent road plane expressed in
chassis coordinates. The supported environment is straight and level, so road
space and world space are aligned; these calculations nevertheless use the
local road datum directly and do not require the inferred vehicle pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kinematics.core.metrics.steering_axis_geometry as axis_geometry

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_scrub_radius(ctx: MetricContext) -> float | None:
    """Return ISO scrub radius in millimetres.

    Following ISO 8855:2011 scrub radius (§7.2.10), this is the
    distance in the axle-local road plane from the tyre contact centre to the
    steering-axis intersection with that plane. Both points are represented
    in chassis coordinates. The value is an unsigned road-plane magnitude,
    not the commonly conflated signed lateral steering-axis offset, and does
    not use world space or an inferred chassis pitch.

    Returns ``None`` when the steering axis is parallel to the road plane.
    """
    return axis_geometry.calculate_scrub_radius(
        ctx.physical_steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
    )


def calculate_steering_axis_offset_ground(ctx: MetricContext) -> float | None:
    """Return ISO steering-axis offset at ground in millimetres.

    Following ISO 8855:2011 steering-axis offset at ground (§7.2.6), this is
    the signed lateral
    component along tyre ``Y_T`` from the tyre contact centre to the
    steering-axis intersection with the axle-local road plane. The tyre axis
    and both points are expressed in chassis coordinates. Positive means that
    the intersection is inboard of the contact centre. World space and inferred
    chassis pitch are not used.

    Returns ``None`` when the steering axis is parallel to the road plane or
    the wheel spin axis has no lateral projection into it.
    """
    return axis_geometry.calculate_steering_axis_offset_at_ground(
        ctx.physical_steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
        ctx.wheel_axis,
        ctx.side_sign,
    )


def calculate_mechanical_trail(ctx: MetricContext) -> float | None:
    """Return wheel-relative mechanical trail in millimetres.

    This is ISO 8855:2011 castor offset at ground (§7.2.3, also called castor
    trail or kinematic trail): the signed longitudinal component along tyre ``X_T``
    from the tyre contact centre to the steering-axis intersection with the
    axle-local road plane. The tyre axis and both points are expressed in
    chassis coordinates. Positive means that the intersection is ahead of the
    contact centre.

    It is deliberately not a raw chassis-X difference. At non-zero steer the
    tyre longitudinal axis rotates away from chassis X, and mechanical trail
    remains a wheel-relative steering-force lever arm. World space and inferred
    chassis pitch are not used.

    Returns ``None`` when the steering axis is parallel to the road plane or
    the wheel spin axis has no lateral projection into it.
    """
    return axis_geometry.calculate_mechanical_trail(
        ctx.physical_steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
        ctx.wheel_axis,
        ctx.side_sign,
    )
