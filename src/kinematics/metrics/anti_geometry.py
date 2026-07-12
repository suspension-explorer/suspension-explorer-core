"""
Side-view anti-geometry metrics.

Quantifies how the side-view suspension geometry resists chassis pitch under
longitudinal load transfer: anti-dive (front braking), anti-lift (rear
braking), and anti-squat (driven axle under acceleration). All are expressed
as a percentage where 100 percent means the geometry fully reacts the
load-transfer pitch moment and 0 percent means it reacts none of it.

All quantities are evaluated in the side view (the XZ plane). The governing
line is the line from a reaction point (contact patch for outboard brakes,
wheel center for inboard-sprung drive) to the side-view instant center
(SVIC). Its inclination, expressed as tan(theta), together with the wheelbase
L and CG height above ground h, sets the anti percentage.

Sign conventions follow ISO 8855 (X forward, Z up). Every metric returns None
when the SVIC is undefined, a denominator is within EPS_GEOMETRIC of zero, the
CG is not above the ground plane, or a required configuration field is unset.
"""

from __future__ import annotations

from math import atan, degrees
from typing import TYPE_CHECKING

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_svsa_angle(ctx: "MetricContext") -> float | None:
    """
    Side-view swing-arm line inclination in degrees.

    This is the inclination of the side-view line from the contact patch to
    the SVIC. A side-view line has a 180-degree ambiguity (it is a line, not
    a ray), so the angle is taken as the atan of the slope rather than atan2,
    giving a range of (-90, 90) degrees:

        svsa_angle = degrees(atan((SVIC_z - CP_z) / (SVIC_x - CP_x)))

    Positive means the line rises toward +X (toward the vehicle front).
    Returns None if the SVIC is undefined or the line is vertical (the
    horizontal run is within EPS_GEOMETRIC of zero).
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    cp = ctx.contact_patch_center

    run = float(svic[Axis.X]) - float(cp[Axis.X])
    if abs(run) < EPS_GEOMETRIC:
        # Vertical side-view line: slope undefined.
        return None
    rise = float(svic[Axis.Z]) - float(cp[Axis.Z])
    return degrees(atan(rise / run))


def _cg_height_above_ground(ctx: "MetricContext") -> float | None:
    """
    CG height above the ground plane in mm, or None if not strictly positive.

    In the chassis-fixed frame the ground follows the tire, so ground level is
    the contact-patch Z. A non-positive height would put the CG at or below the
    road, which is non-physical for these anti formulas, so guard it to None.
    """
    cp = ctx.contact_patch_center
    height = float(ctx.cg_position[Axis.Z]) - float(cp[Axis.Z])
    if height <= EPS_GEOMETRIC:
        return None
    return height


def calculate_anti_dive_pct(ctx: "MetricContext") -> float | None:
    """
    Front-axle anti-dive percentage under braking.

    Only defined for a front axle with a known front brake bias. With outboard
    brakes the front suspension reacts the brake force along the contact-patch
    -> SVIC line. With L the wheelbase and h the CG height above ground, the
    line inclination is taken about the contact patch as:

        tan_theta = (SVIC_z - CP_z) / (CP_x - SVIC_x)

    which is positive in the classic anti-dive layout (SVIC above and BEHIND
    the front contact patch). Then:

        anti_dive_pct = 100 * front_brake_bias * (L / h) * tan_theta

    Returns None when the axle is not the front, the front brake bias is unset,
    the SVIC is undefined, the CG is not above ground, or the run is degenerate.
    """
    config = ctx.config
    if config.axle_position != "front" or config.front_brake_bias is None:
        return None

    svic = ctx.side_view_ic
    if svic is None:
        return None
    cp = ctx.contact_patch_center

    # Run measured from SVIC to contact patch so tan_theta is positive when the
    # SVIC sits behind (-X of) the front contact patch: the anti-dive geometry.
    run = float(cp[Axis.X]) - float(svic[Axis.X])
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_ground(ctx)
    if height is None:
        return None

    tan_theta = (float(svic[Axis.Z]) - float(cp[Axis.Z])) / run
    return 100.0 * config.front_brake_bias * (ctx.wheelbase / height) * tan_theta


def calculate_anti_lift_pct(ctx: "MetricContext") -> float | None:
    """
    Rear-axle anti-lift percentage under braking.

    Only defined for a rear axle with a known front brake bias; the rear share
    of the braking force is (1 - front_brake_bias). With outboard brakes the
    rear suspension reacts the brake force along the contact-patch -> SVIC line.
    The line inclination is taken about the contact patch as:

        tan_theta = (SVIC_z - CP_z) / (SVIC_x - CP_x)

    which is positive when the SVIC sits above and AHEAD (+X) of the rear
    contact patch. Then:

        anti_lift_pct = 100 * (1 - front_brake_bias) * (L / h) * tan_theta

    Returns None when the axle is not the rear, the front brake bias is unset,
    the SVIC is undefined, the CG is not above ground, or the run is degenerate.
    """
    config = ctx.config
    if config.axle_position != "rear" or config.front_brake_bias is None:
        return None

    svic = ctx.side_view_ic
    if svic is None:
        return None
    cp = ctx.contact_patch_center

    run = float(svic[Axis.X]) - float(cp[Axis.X])
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_ground(ctx)
    if height is None:
        return None

    rear_brake_bias = 1.0 - config.front_brake_bias
    tan_theta = (float(svic[Axis.Z]) - float(cp[Axis.Z])) / run
    return 100.0 * rear_brake_bias * (ctx.wheelbase / height) * tan_theta


def calculate_anti_squat_pct(ctx: "MetricContext") -> float | None:
    """
    Anti-squat (rear) / anti-lift (front) percentage under acceleration.

    Only defined when a driven axle is configured AND it is this axle
    (driven_axle == axle_position, both non-None). With inboard-sprung drive
    (halfshafts) the tractive force reacts along the WHEEL-CENTER -> SVIC line,
    not the contact-patch line. The full drive torque is carried by the driven
    axle. With L the wheelbase and h the CG height above ground:

        rear axle:  tan_theta = (SVIC_z - WC_z) / (SVIC_x - WC_x)
        front axle: tan_theta = (SVIC_z - WC_z) / (WC_x - SVIC_x)

    tan_theta is positive when the geometry resists the acceleration pitch
    (squat at the rear, lift at the front). Then:

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
    if config.axle_position == "front":
        run = float(wc[Axis.X]) - float(svic[Axis.X])
    else:
        run = float(svic[Axis.X]) - float(wc[Axis.X])
    if abs(run) < EPS_GEOMETRIC:
        return None
    height = _cg_height_above_ground(ctx)
    if height is None:
        return None

    tan_theta = (float(svic[Axis.Z]) - float(wc[Axis.Z])) / run
    return 100.0 * (ctx.wheelbase / height) * tan_theta
