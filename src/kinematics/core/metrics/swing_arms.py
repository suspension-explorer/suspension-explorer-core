"""
Virtual swing arm length metrics.

Computes front-view and side-view swing arm lengths from the instant
center positions and the wheel-ground tangent point.

Definitions and sign conventions:

SVSA (Side-View Swing Arm):
    The horizontal (X-axis) distance from the wheel-ground tangent point to
    the side-view instant center (SVIC).

        SVSA = SVIC_X - GroundTangent_X

    Positive when the SVIC is ahead of (+X relative to) the wheel-plane ground
    tangent, which is the typical case for a conventional double-wishbone
    layout. Negative values indicate the SVIC is behind that tangent.

FVSA (Front-View Swing Arm):
    The Euclidean distance in the YZ plane from the wheel-ground tangent point
    to the front-view instant center (FVIC), with a sign that encodes
    whether the FVIC is inboard or outboard of the ground tangent.

        FVSA = +/- sqrt((FVIC_Y - T_Y)^2 + (FVIC_Z - T_Z)^2)

    Positive when the FVIC is inboard (closer to vehicle centerline) of the
    ground tangent, where inboard/outboard is judged by the component of
    (FVIC - T) along the ground line rather than along chassis Y. For a
    left-side corner (Y > 0) "inboard" means a negative along-ground
    component; for a right-side corner (Y < 0) it means a positive one. On a
    level ground line the tangent is +Y, so this is the sign of FVIC_Y - T_Y.
    Negative values indicate the FVIC is outboard of the ground tangent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_svsa_length(ctx: MetricContext) -> float | None:
    """
    Side-view swing arm length in mm.

    The SVSA length is the forward distance in the ground plane from the
    wheel-ground tangent to the side-view instant center. A positive value
    means the IC is ahead of the ground tangent; negative means behind.

    Returns None if the SVIC is undefined.
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    displacement = svic - ctx.wheel_ground_tangent
    return float(ctx.ground.forward.dot(displacement))


def calculate_fvsa_length(ctx: MetricContext) -> float | None:
    """
    Front-view swing arm length in mm.

    The FVSA length is the YZ distance from the wheel-plane ground-tangent
    point to the front-view instant center. The magnitude is a plain
    point-to-point distance, while the sign is taken from the displacement
    resolved along the ground line, so "inboard" is measured across the road
    surface rather than along chassis Y.

    Returns None if the FVIC is undefined.
    """
    fvic = ctx.front_view_ic
    if fvic is None:
        return None
    displacement = fvic - ctx.wheel_ground_tangent
    along_ground = float(ctx.ground.lateral.dot(displacement))
    normal_height = float(ctx.ground.normal.dot(displacement))
    length = np.sqrt(along_ground * along_ground + normal_height * normal_height)

    # Sign convention: positive when FVIC is on the vehicle-center side
    # of the ground tangent. The ground-line tangent points from vehicle
    # right to left, so this along-ground component reduces to dy on a level
    # ground line. For the left side (Y > 0), inboard means a negative
    # component, so we negate. For the right side the sign is already correct.
    return float(length * (-ctx.side_sign * np.sign(along_ground)))
