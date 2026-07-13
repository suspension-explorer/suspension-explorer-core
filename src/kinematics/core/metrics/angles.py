"""
Wheel alignment angle metrics.

All functions accept a MetricContext and return angles in degrees.

Coordinate System Assumption: ISO 8855 (X-Forward, Y-Left, Z-Up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kinematics.core.primitives.enums import Axis
from kinematics.core.targeting import WorldAxisSystem

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_camber(ctx: MetricContext) -> float:
    """
    Camber angle in degrees.

    Camber is the angle of the wheel's vertical centerline with respect
    to the vehicle's vertical axis (Z-axis), viewed from the front
    (YZ plane). Negative camber means the top of the wheel is tilted
    inwards (towards vehicle centerline).
    """
    side = ctx.side_sign
    axle = ctx.wheel_axis

    # Wheel's "up" vector is perpendicular to both the axle and the
    # vehicle's longitudinal axis (X-axis).
    # Multiply by -side so the vector points roughly +Z (Up) for both
    # sides.
    wheel_up = axle.cross(WorldAxisSystem.X) * -side

    # Project onto the front view plane (YZ plane).
    proj_y = wheel_up[Axis.Y]
    proj_z = wheel_up[Axis.Z]

    # Angle with the global Z-axis.
    angle = np.arctan2(proj_y, proj_z)

    # For right side, inward tilt is +Y which gives positive angle.
    # Invert to match convention.
    camber_rad = angle if side > 0 else -angle
    return float(np.rad2deg(camber_rad))


def calculate_caster(ctx: MetricContext) -> float:
    """
    Caster angle in degrees.

    Caster is the angle of the steering axis with respect to the
    vehicle's vertical axis (Z-axis), viewed from the side (XZ plane).
    Positive caster means the top of the steering axis is tilted
    rearward.
    """
    steering = ctx.steering_axis

    # Project onto the side view plane (XZ plane).
    proj_x = steering[Axis.X]
    proj_z = steering[Axis.Z]

    # Positive caster = top tilted rearward (negative X relative to
    # bottom). Negate x so rearward tilt gives a positive angle.
    caster_rad = np.arctan2(-proj_x, proj_z)
    return float(np.rad2deg(caster_rad))


def calculate_kpi(ctx: MetricContext) -> float:
    """
    Kingpin inclination (KPI) angle in degrees.

    KPI is the angle of the steering axis with respect to the
    vehicle's vertical axis (Z-axis), viewed from the front
    (YZ plane). Positive KPI means the top of the steering axis
    is tilted inward (towards vehicle centerline).
    """
    side = ctx.side_sign
    steering = ctx.steering_axis

    # Project onto the front view plane (YZ plane).
    proj_y = steering[Axis.Y]
    proj_z = steering[Axis.Z]

    # Positive KPI = top tilted inward. For the left side (Y > 0),
    # inward tilt means negative Y component relative to bottom,
    # so negate Y. For the right side, inward tilt is positive Y.
    kpi_rad = np.arctan2(-side * proj_y, proj_z)
    return float(np.rad2deg(kpi_rad))


def calculate_roadwheel_angle(ctx: MetricContext) -> float:
    """
    Roadwheel angle in degrees.

    The angle of the wheel's longitudinal axis relative to the
    vehicle X-axis, viewed from the top (XY plane). Positive means
    the front of the wheel is turned towards the vehicle centerline.
    This is the same measurement as toe but uses the clearer
    vehicle-dynamics-facing name.
    """
    return calculate_toe(ctx)


def calculate_toe(ctx: MetricContext) -> float:
    """
    Toe angle in degrees.

    Toe is the angle of the wheel's longitudinal axis with respect to
    the vehicle's longitudinal axis (X-axis), viewed from the top
    (XY plane). Positive toe (toe-in) means the front of the wheel
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
