"""ISO-aligned steering-axis geometry on the current ground plane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.primitives.constants import EPS_GEOMETRIC

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext
    from kinematics.core.primitives.geometry import Direction3, Vector3


def _ground_displacement(ctx: MetricContext) -> Vector3 | None:
    """Return steering-axis ground intersection minus tyre contact centre."""
    ground_intersection = ctx.steering_axis_ground_intersection
    if ground_intersection is None:
        return None
    return ground_intersection - ctx.wheel_ground_tangent


def _wheel_ground_axes(
    ctx: MetricContext,
) -> tuple[Direction3, Direction3] | None:
    """Return outward-lateral and forward-longitudinal tyre axes on ground.

    These are the ISO tyre ``Y_T`` and ``X_T`` directions expressed in chassis
    coordinates. ``Y_T`` follows the wheel spin axis toward the wheel outboard
    face; ``X_T`` is oriented forward for both vehicle sides.
    """
    ground_normal = ctx.ground.normal
    projected_axis = ctx.wheel_axis.vector() - ground_normal * ctx.wheel_axis.dot(
        ground_normal
    )
    if projected_axis.norm() < EPS_GEOMETRIC:
        return None
    lateral = projected_axis.normalize()
    longitudinal = (ctx.side_sign * lateral.cross(ground_normal)).normalize()
    return lateral, longitudinal


def calculate_scrub_radius(ctx: MetricContext) -> float | None:
    """Return ISO scrub radius in millimetres.

    ISO 8855 defines scrub radius as the distance in the ground plane from the
    tyre contact centre to the steering-axis intersection with that plane.
    This is an unsigned magnitude, not the commonly conflated signed lateral
    steering-axis offset.

    Returns ``None`` when the steering axis is parallel to the ground plane.
    """
    displacement = _ground_displacement(ctx)
    return None if displacement is None else displacement.norm()


def calculate_steering_axis_offset_ground(ctx: MetricContext) -> float | None:
    """Return ISO steering-axis offset at ground in millimetres.

    This is the signed lateral component, along the tyre ``Y_T`` axis, from the
    tyre contact centre to the steering-axis ground intersection. Positive
    means that the intersection is inboard of the contact centre.

    Returns ``None`` when the steering axis is parallel to the ground plane or
    the wheel spin axis has no lateral projection into it.
    """
    displacement = _ground_displacement(ctx)
    axes = _wheel_ground_axes(ctx)
    if displacement is None or axes is None:
        return None
    lateral, _ = axes
    return -float(displacement.dot(lateral))


def calculate_mechanical_trail(ctx: MetricContext) -> float | None:
    """Return wheel-relative mechanical trail in millimetres.

    This is ISO 8855 *castor offset at ground* (also called castor trail or
    kinematic trail): the signed longitudinal component, along the tyre
    ``X_T`` axis, from the tyre contact centre to the steering-axis
    intersection with the ground plane. Positive means that the intersection
    is ahead of the contact centre.

    It is deliberately not a raw chassis-X difference. At non-zero steer the
    tyre longitudinal axis rotates away from chassis X, and mechanical trail
    remains a wheel-relative steering-force lever arm.

    Returns ``None`` when the steering axis is parallel to the ground plane or
    the wheel spin axis has no lateral projection into it.
    """
    displacement = _ground_displacement(ctx)
    axes = _wheel_ground_axes(ctx)
    if displacement is None or axes is None:
        return None
    _, longitudinal = axes
    return float(displacement.dot(longitudinal))
