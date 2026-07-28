"""
Side-view anti-geometry metrics.

Quantifies how the side-view suspension geometry resists chassis pitch under
longitudinal load transfer: anti-dive (front braking), anti-lift (rear
braking), and anti-squat (driven axle under acceleration). All are expressed
as a percentage where 100 percent means the geometry fully reacts the
load-transfer pitch moment and 0 percent means it reacts none of it.

The anti percentages model brake and tractive forces relative to the
axle-local road plane. Following the ISO 8855 vehicle and road-plane
conventions, its normal and forward direction are expressed in chassis
coordinates. In the supported straight, level environment road space and
world space are aligned, but the single-axle metric does not infer or predict
whole-vehicle pitch. The governing line runs from a reaction point
(wheel-ground tangent for outboard brakes, wheel center for inboard-sprung
drive) to the side-view instant center (SVIC). Its inclination, expressed as
tan(theta), together with the wheelbase L and CG height above ground h, sets
the anti percentage.

Sign conventions follow the ISO 8855 vehicle-axis orientation (X forward,
Y left, Z up). Every metric returns None when the SVIC is undefined, a
denominator is within EPS_GEOMETRIC of zero, the CG is not above the road
plane, or a required configuration field is unset.
"""

from __future__ import annotations

from math import atan, degrees
from typing import TYPE_CHECKING

from kinematics.core.enums import Axis, AxlePosition
from kinematics.core.primitives.constants import EPS_GEOMETRIC

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_svsa_angle(ctx: "MetricContext") -> float | None:
    """
    Side-view swing-arm line inclination in degrees.

    This is deliberately a chassis-space construction: the line from the wheel
    contact tangent to the SVIC is resolved in the chassis XZ plane, using the
    ISO 8855 vehicle-axis orientation. It does not use the road plane, world
    space, or inferred chassis attitude.

    A side-view line has a 180-degree ambiguity (it is a line, not a ray), so
    the angle is taken as the atan of the slope rather than atan2, giving a
    range of (-90, 90) degrees:

        svsa_angle = degrees(atan((SVIC_z - T_z) / (SVIC_x - T_x)))

    Positive means the line rises toward +X (toward the vehicle front).
    Returns None if the SVIC is undefined or the line is vertical (the
    horizontal run is within EPS_GEOMETRIC of zero).
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    tangent = ctx.wheel_ground_tangent

    # Deliberately a chassis-frame side-view construction: this angle describes
    # the linkage's XZ swing-arm line, not a ground-relative force line. The anti
    # percentages resolve their rise along the ground normal instead.
    run = float(svic[Axis.X]) - float(tangent[Axis.X])
    if abs(run) < EPS_GEOMETRIC:
        # Vertical side-view line: slope undefined.
        return None
    rise = float(svic[Axis.Z]) - float(tangent[Axis.Z])
    return degrees(atan(rise / run))


def _cg_height_above_road(ctx: "MetricContext") -> float | None:
    """
    CG height above the road plane in mm, or None if not strictly positive.

    The CG point and road plane are represented in chassis coordinates. The
    load-transfer moment arm in the anti formulas is the CG's perpendicular
    distance to the ISO-style local road plane because the braking or tractive
    force acts parallel to that plane. This is not a world-space CG height or a
    prediction of whole-vehicle pitch. A non-positive height would put the CG
    at or below the road, which is non-physical here.
    """
    height = ctx.road.signed_distance(ctx.cg_position)
    if height <= EPS_GEOMETRIC:
        return None
    return height


def calculate_anti_dive_pct(ctx: "MetricContext") -> float | None:
    """
    Front-axle anti-dive percentage under braking.

    This is an axle-local geometric anti percentage, not an actual chassis
    pitch prediction. Following the ISO 8855 road-plane concept, the reaction
    line is resolved along the local road forward direction and normal, both
    expressed in chassis coordinates. World-space presentation is not used.

    Only defined for a front axle with a known front brake bias. With outboard
    brakes the front suspension reacts the brake force along the wheel contact
    tangent -> SVIC line. With L the wheelbase, h the CG height above road, and
    n the road plane's upward normal, the line inclination is taken about the
    tangent as:

        tan_theta = n . (SVIC - T) / (T_x - SVIC_x)

    The tangent T lies on the road plane, so the rise n . (SVIC - T) is the
    SVIC's signed distance to that plane; on a level road plane it reduces
    exactly to SVIC_z - T_z. tan_theta is positive in the classic anti-dive
    layout (SVIC above and BEHIND the front wheel-ground tangent). Then:

        anti_dive_pct = 100 * front_brake_bias * (L / h) * tan_theta

    Returns None when the axle is not the front, the front brake bias is unset,
    the SVIC is undefined, the CG is not above ground, or the run is degenerate.
    """
    config = ctx.config
    if config.axle_position is not AxlePosition.FRONT or (
        config.front_brake_bias is None
    ):
        return None

    svic = ctx.side_view_ic
    if svic is None:
        return None
    tangent = ctx.wheel_ground_tangent

    # Run measured from SVIC to the tangent so tan_theta is positive when the
    # SVIC sits behind (-X of) the front tangent: the anti-dive geometry.
    run = float(ctx.road.forward.dot(tangent - svic))
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_road(ctx)
    if height is None:
        return None

    # The tangent lies on the road plane, so the tangent -> SVIC rise along
    # the road normal is the SVIC's signed distance to that plane.
    tan_theta = ctx.road.signed_distance(svic) / run
    return 100.0 * config.front_brake_bias * (ctx.wheelbase / height) * tan_theta


def calculate_anti_lift_pct(ctx: "MetricContext") -> float | None:
    """
    Rear-axle anti-lift percentage under braking.

    This is an axle-local geometric anti percentage, not an actual chassis
    pitch prediction. Following the ISO 8855 road-plane concept, the reaction
    line is resolved along the local road forward direction and normal, both
    expressed in chassis coordinates. World-space presentation is not used.

    Only defined for a rear axle with a known front brake bias; the rear share
    of the braking force is (1 - front_brake_bias). With outboard brakes the
    rear suspension reacts the brake force along the wheel contact tangent
    -> SVIC line. With n the road plane's upward normal, the line inclination
    is taken about the tangent as:

        tan_theta = n . (SVIC - T) / (SVIC_x - T_x)

    The tangent T lies on the road plane, so the rise n . (SVIC - T) is the
    SVIC's signed distance to that plane; on a level road plane it reduces
    exactly to SVIC_z - T_z. tan_theta is positive when the SVIC sits above
    and AHEAD (+X) of the rear wheel-ground tangent. Then:

        anti_lift_pct = 100 * (1 - front_brake_bias) * (L / h) * tan_theta

    Returns None when the axle is not the rear, the front brake bias is unset,
    the SVIC is undefined, the CG is not above ground, or the run is degenerate.
    """
    config = ctx.config
    if config.axle_position is not AxlePosition.REAR or (
        config.front_brake_bias is None
    ):
        return None

    svic = ctx.side_view_ic
    if svic is None:
        return None
    tangent = ctx.wheel_ground_tangent

    run = float(ctx.road.forward.dot(svic - tangent))
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_road(ctx)
    if height is None:
        return None

    rear_brake_bias = 1.0 - config.front_brake_bias
    # The tangent lies on the road plane, so the tangent -> SVIC rise along
    # the road normal is the SVIC's signed distance to that plane.
    tan_theta = ctx.road.signed_distance(svic) / run
    return 100.0 * rear_brake_bias * (ctx.wheelbase / height) * tan_theta


def calculate_anti_squat_pct(ctx: "MetricContext") -> float | None:
    """
    Anti-squat (rear) / anti-lift (front) percentage under acceleration.

    This is an axle-local geometric anti percentage, not an actual chassis
    pitch prediction. Following the ISO 8855 road-plane concept, the reaction
    line is resolved along the local road forward direction and normal, both
    expressed in chassis coordinates. World-space presentation is not used.

    Only defined when a driven axle is configured AND it is this axle
    (driven_axle == axle_position, both non-None). With inboard-sprung drive
    (halfshafts) the tractive force reacts along the WHEEL-CENTER -> SVIC line,
    not the wheel-ground tangent line. The full drive torque is carried by
    the driven axle. With L the wheelbase, h the CG height above ground, and
    n the road plane's upward normal:

        rear axle:  tan_theta = n . (SVIC - WC) / (SVIC_x - WC_x)
        front axle: tan_theta = n . (SVIC - WC) / (WC_x - SVIC_x)

    Neither endpoint lies on the road plane, so the rise n . (SVIC - WC) is
    the difference of their signed distances to it; on a level road plane it
    reduces exactly to SVIC_z - WC_z. tan_theta is positive when the geometry
    resists the acceleration pitch (squat at the rear, lift at the front).
    Then:

        anti_squat_pct = 100 * (L / h) * tan_theta

    Returns None when no driven axle matches this axle, the SVIC is undefined,
    the CG is not above ground, or the run is degenerate.
    """
    config = ctx.config
    if config.driven_axle is None or config.axle_position is None:
        return None
    if config.driven_axle != config.axle_position:
        return None

    svic = ctx.side_view_ic
    if svic is None:
        return None
    wc = ctx.wheel_center

    # Front (FWD) and rear axles flip the sense of the horizontal run so that
    # positive tan_theta always means "resists the acceleration pitch".
    if config.axle_position is AxlePosition.FRONT:
        run = float(ctx.road.forward.dot(wc - svic))
    else:
        run = float(ctx.road.forward.dot(svic - wc))
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_road(ctx)
    if height is None:
        return None

    # Neither endpoint lies on the road plane, so the rise along the road
    # normal is the signed-distance difference (the plane offset cancels).
    rise = ctx.road.signed_distance(svic) - ctx.road.signed_distance(wc)
    tan_theta = rise / run
    return 100.0 * (ctx.wheelbase / height) * tan_theta
