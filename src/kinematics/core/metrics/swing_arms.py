"""
Virtual swing arm length metrics.

Computes front-view and side-view swing arm lengths from the instant
center positions and the wheel-plane road-tangent point.

Definitions and sign conventions:

SVSA (Side-View Swing Arm):
    The horizontal (X-axis) distance from the wheel-plane road-tangent point to
    the side-view instant center (SVIC).

        SVSA = SVIC_X - RoadTangent_X

    Positive when the SVIC is ahead of (+X relative to) the wheel-plane road
    tangent, which is the typical case for a conventional double-wishbone
    layout. Negative values indicate the SVIC is behind that tangent.

FVSA (Front-View Swing Arm):
    The Euclidean distance in the YZ plane from the wheel-plane road-tangent point
    to the front-view instant center (FVIC), with a sign that encodes
    whether the FVIC is inboard or outboard of the road tangent.

        FVSA = +/- sqrt((FVIC_Y - T_Y)^2 + (FVIC_Z - T_Z)^2)

    Positive when the FVIC is inboard (closer to vehicle centerline)
    of the road tangent. For a left-side corner (Y > 0) "inboard"
    means FVIC_Y < T_Y; for a right-side corner (Y < 0) "inboard"
    means FVIC_Y > T_Y. Negative values indicate the FVIC is
    outboard of the road tangent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kinematics.core.enums import Axis

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_svsa_length(ctx: MetricContext) -> float | None:
    """
    Side-view swing arm length in mm.

    The SVSA length is the horizontal distance (in X) from the wheel-plane
    road-tangent point to the side-view instant center. A positive value means
    the IC is ahead of the road tangent; negative means behind.

    Returns None if the SVIC is undefined.
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    road_tangent = ctx.wheel_plane_road_tangent
    return float(svic[Axis.X] - road_tangent[Axis.X])


def calculate_fvsa_length(ctx: MetricContext) -> float | None:
    """
    Front-view swing arm length in mm.

    The FVSA length is the lateral distance (in Y) from the wheel-plane
    road-tangent point to the front-view instant center. The sign follows
    the vehicle Y axis (positive = towards vehicle left).

    Returns None if the FVIC is undefined.
    """
    fvic = ctx.front_view_ic
    if fvic is None:
        return None
    road_tangent = ctx.wheel_plane_road_tangent

    # Lateral distance from road tangent to FVIC, preserving sign.
    dy = float(fvic[Axis.Y] - road_tangent[Axis.Y])
    dz = float(fvic[Axis.Z] - road_tangent[Axis.Z])
    # Signed length: positive when IC is inboard of the road tangent.
    length = np.sqrt(dy * dy + dz * dz)

    # Sign convention: positive when FVIC is on the vehicle-center side
    # of the road tangent. For the left side (Y > 0), inboard means
    # FVIC_Y < T_Y, so we negate. For right side, FVIC_Y > T_Y is
    # inboard, so the sign is already correct.
    return float(length * (-ctx.side_sign * np.sign(dy)))
