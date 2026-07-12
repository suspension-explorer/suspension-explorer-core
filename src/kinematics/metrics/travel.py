"""
Per-state suspension travel metrics.

These metrics report wheel travel, half-track, and installed damper length.
Every value is a scalar in millimeters. Sign conventions follow ISO 8855
(X forward, Y left, Z up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis, PointID

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_wheel_travel(ctx: "MetricContext") -> float | None:
    """
    Vertical wheel travel in mm relative to the design condition.

    Defined as the current wheel-center Z minus the design wheel-center Z.
    Positive means the wheel has moved up in the chassis-fixed frame (bump);
    negative means droop.

        wheel_travel = WC_z(current) - WC_z(design)
    """
    current_z = float(ctx.wheel_center[Axis.Z])
    design_z = float(ctx.design_wheel_center[Axis.Z])
    return current_z - design_z


def calculate_half_track(ctx: "MetricContext") -> float | None:
    """
    Half-track at this corner in mm.

    Half-track is the lateral distance of the contact patch from the vehicle
    centerline, i.e. the magnitude of the contact-patch Y.

        half_track = |CP_y(current)|
    """
    return abs(float(ctx.contact_patch_center[Axis.Y]))


def calculate_damper_length(ctx: "MetricContext") -> float | None:
    """
    Installed spring/damper (coilover) length in mm.

    The length is the straight-line distance between the strut top mount
    (chassis-fixed) and the strut bottom (body-mounted foot). Only defined
    when the suspension actually carries a strut group; otherwise None.

        damper_length = |STRUT_TOP - STRUT_BOTTOM|
    """
    if not ctx.suspension.has_strut:
        return None
    top = ctx.state.get(PointID.STRUT_TOP)
    bottom = ctx.state.get(PointID.STRUT_BOTTOM)
    # Euclidean distance between the two mounts (a Point3 - Point3 -> Vector3).
    return float((top - bottom).norm())
