"""
Virtual swing arm length metrics.

Computes front-view and side-view swing arm lengths from the instant
center positions and the contact patch center.

Definitions and sign conventions:

SVSA (Side-View Swing Arm):
    The horizontal (X-axis) distance from the contact patch center to
    the side-view instant center (SVIC).

        SVSA = SVIC_X - ContactPatch_X

    Positive when the SVIC is ahead of (+X relative to) the contact
    patch, which is the typical case for a conventional double-wishbone
    layout.  Negative values indicate the SVIC is behind the contact
    patch.

FVSA (Front-View Swing Arm):
    The Euclidean distance in the YZ plane from the contact patch center
    to the front-view instant center (FVIC), with a sign that encodes
    whether the FVIC is inboard or outboard of the contact patch.

        FVSA = +/- sqrt((FVIC_Y - CP_Y)^2 + (FVIC_Z - CP_Z)^2)

    Positive when the FVIC is inboard (closer to vehicle centerline)
    of the contact patch.  For a left-side corner (Y > 0) "inboard"
    means FVIC_Y < CP_Y; for a right-side corner (Y < 0) "inboard"
    means FVIC_Y > CP_Y.  Negative values indicate the FVIC is
    outboard of the contact patch.
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

    The SVSA length is the horizontal distance (in X) from the contact
    patch center to the side-view instant center. A positive value means
    the IC is ahead of the contact patch; negative means behind.

    Returns None if the SVIC is undefined.
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    cp = ctx.contact_patch_center
    return float(svic[Axis.X] - cp[Axis.X])


def calculate_fvsa_length(ctx: MetricContext) -> float | None:
    """
    Front-view swing arm length in mm.

    The FVSA length is the lateral distance (in Y) from the contact
    patch center to the front-view instant center. The sign follows
    the vehicle Y axis (positive = towards vehicle left).

    Returns None if the FVIC is undefined.
    """
    fvic = ctx.front_view_ic
    if fvic is None:
        return None
    cp = ctx.contact_patch_center

    # Lateral distance from contact patch to FVIC, preserving sign.
    dy = float(fvic[Axis.Y] - cp[Axis.Y])
    dz = float(fvic[Axis.Z] - cp[Axis.Z])
    # Signed length: positive when IC is inboard of the contact patch.
    length = np.sqrt(dy * dy + dz * dz)

    # Sign convention: positive when FVIC is on the vehicle-center side
    # of the contact patch. For the left side (Y > 0), inboard means
    # FVIC_Y < CP_Y, so we negate. For right side, FVIC_Y > CP_Y is
    # inboard, so the sign is already correct.
    return float(length * (-ctx.side_sign * np.sign(dy)))
