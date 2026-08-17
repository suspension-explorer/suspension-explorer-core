"""
Wheel alignment angle metrics.

All functions accept a MetricContext and return angles in degrees.

The calculations use the ISO 8855 vehicle-axis orientation (X forward,
Y left, Z up), expressed here in chassis space. They are kinematic alignment
angles relative to the chassis axes; they do not depend on world space or the
road plane. Steering-axis metrics are documented separately in ``steering``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kinematics.core.coordinates import ChassisAxisSystem
from kinematics.core.enums import Axis

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_camber(ctx: MetricContext) -> float:
    """
    Camber angle in degrees.

    This is ISO 8855:2011 vehicle-relative camber (§7.1.17). The wheel spin
    axis is expressed in chassis space and resolved in the chassis YZ plane
    against chassis +Z. It is not the road-relative inclination angle of
    §7.1.16.

    Negative camber means the top of the wheel is tilted inwards, towards the
    vehicle centerline.
    """
    side = ctx.side_sign
    axle = ctx.wheel_axis

    # Wheel's "up" vector is perpendicular to both the axle and the
    # vehicle's longitudinal axis (X-axis).
    # Multiply by -side so the vector points roughly +Z (Up) for both
    # sides.
    wheel_up = axle.cross(ChassisAxisSystem.X) * -side

    # Project onto the front view plane (YZ plane).
    proj_y = wheel_up[Axis.Y]
    proj_z = wheel_up[Axis.Z]

    # Angle with the global Z-axis.
    angle = np.arctan2(proj_y, proj_z)

    # For right side, inward tilt is +Y which gives positive angle.
    # Invert to match convention.
    camber_rad = angle if side > 0 else -angle
    return float(np.rad2deg(camber_rad))


def calculate_toe(ctx: MetricContext) -> float:
    """
    Toe angle in degrees.

    The wheel spin axis is expressed in chassis space and resolved in the
    chassis XY plane relative to chassis +X. It is independent of the road
    plane and world space. Positive toe means toe-in: the front of the wheel
    points inwards.
    """
    side = ctx.side_sign
    axle = ctx.wheel_axis

    # Project axle vector onto the top view plane (XY plane).
    proj_x = axle[Axis.X]
    proj_y = axle[Axis.Y]

    # Toe-in results in the axle vector pointing slightly forward (+X).
    if side > 0:  # Left side
        toe_rad = np.arctan2(proj_x, proj_y)
    else:  # Right side: measure relative to -Y axis
        toe_rad = np.arctan2(proj_x, -proj_y)

    return float(np.rad2deg(toe_rad))


def calculate_steer(ctx: MetricContext) -> float:
    """Return ISO 8855:2011 steer angle (§7.1.1) in degrees.

    Steer is the vehicle-fixed, right-hand-rule heading of the wheel forward
    direction about chassis +Z. Thus a left turn is positive for *both*
    corners. It deliberately differs from :func:`calculate_toe`, whose
    side-folded convention makes positive mean toe-in.

    The result is resolved in the chassis XY plane and is independent of the
    road plane and world space.
    """
    axle = ctx.wheel_axis
    side = ctx.side_sign

    # The wheel axis is inboard-to-outboard. ``side * (axle × +Z)`` is the
    # wheel's forward direction on either side of the vehicle.
    forward_x = side * axle[Axis.Y]
    forward_y = -side * axle[Axis.X]
    return float(np.rad2deg(np.arctan2(forward_y, forward_x)))
