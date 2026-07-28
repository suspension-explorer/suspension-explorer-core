"""
Virtual swing arm length metrics.

Computes front-view and side-view swing arm lengths from the instant-center
positions and the wheel contact tangent. The ISO 8855 vehicle-axis orientation
defines forward and left, while the ISO local road-plane concept supplies the
road normal. The road basis is expressed in chassis coordinates; world space
and inferred whole-vehicle pitch are not used.

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
    (FVIC - T) along the road plane rather than along chassis Y. For a
    left-side corner (Y > 0) "inboard" means a negative along-ground
    component; for a right-side corner (Y < 0) it means a positive one. On a
    level road plane the lateral direction is +Y, so this is the sign of
    FVIC_Y - T_Y.
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

    The displacement from the wheel contact tangent to the SVIC is expressed
    in chassis coordinates and resolved along the axle-local road plane's
    forward direction. A positive value means the IC is ahead of the contact
    tangent; negative means behind. World-space presentation is not used.

    Returns None if the SVIC is undefined.
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    displacement = svic - ctx.wheel_ground_tangent
    return float(ctx.road.forward.dot(displacement))


def calculate_fvsa_length(ctx: MetricContext) -> float | None:
    """
    Front-view swing arm length in mm.

    The displacement from the wheel contact tangent to the FVIC is expressed
    in chassis coordinates, then resolved along the axle-local road plane's
    lateral direction and normal. Its magnitude is therefore a front-view
    distance relative to the road datum. The sign encodes whether the FVIC is
    inboard or outboard along the road surface, rather than simply along
    chassis Y. World space and inferred chassis pitch are not used.

    Returns None if the FVIC is undefined.
    """
    fvic = ctx.front_view_ic
    if fvic is None:
        return None
    displacement = fvic - ctx.wheel_ground_tangent
    along_ground = float(ctx.road.lateral.dot(displacement))
    normal_height = float(ctx.road.normal.dot(displacement))
    length = np.sqrt(along_ground * along_ground + normal_height * normal_height)

    # Sign convention: positive when FVIC is on the vehicle-center side
    # of the contact tangent. The road lateral direction points from vehicle
    # right to left, so this component reduces to dy on a level road plane.
    # For the left side (Y > 0), inboard means a negative
    # component, so we negate. For the right side the sign is already correct.
    return float(length * (-ctx.side_sign * np.sign(along_ground)))
