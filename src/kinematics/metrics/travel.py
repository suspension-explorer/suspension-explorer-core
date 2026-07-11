"""
Per-state suspension travel metrics.

These metrics measure how the current solved corner has displaced relative
to its design (as-authored) condition: wheel travel, half-track change,
wheel recession, and the installed damper length. Every value is a scalar
in millimetres. Sign conventions follow ISO 8855 (X forward, Y left, Z up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis, PointID

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_wheel_travel(ctx: "MetricContext") -> float | None:
    """
    Vertical wheel travel in mm relative to the design condition.

    Defined as the current wheel-centre Z minus the design wheel-centre Z.
    Positive means the wheel has moved up in the chassis-fixed frame (bump);
    negative means droop.

        wheel_travel = WC_z(current) - WC_z(design)
    """
    current_z = float(ctx.wheel_center[Axis.Z])
    design_z = float(ctx.design_wheel_center[Axis.Z])
    return current_z - design_z


def calculate_half_track_change(ctx: "MetricContext") -> float | None:
    """
    Change in half-track at this corner in mm relative to design.

    Half-track is the lateral distance of the contact patch from the vehicle
    centreline, i.e. the magnitude of the contact-patch Y. Using magnitudes
    makes the sign convention identical for left and right corners:
    positive means the track has widened at this corner, negative means it
    has narrowed.

        half_track_change = |CP_y(current)| - |CP_y(design)|
    """
    current_y = abs(float(ctx.contact_patch_center[Axis.Y]))
    design_y = abs(float(ctx.design_contact_patch_center[Axis.Y]))
    return current_y - design_y


def calculate_wheel_recession(ctx: "MetricContext") -> float | None:
    """
    Longitudinal recession of the contact patch in mm relative to design.

    In ISO 8855 +X points forward, so a rearward move is a decrease in X.
    We negate the X displacement so that positive recession means the wheel
    has moved rearward (toward the vehicle rear), which is the intuitive
    reading of "recession".

        wheel_recession = -(CP_x(current) - CP_x(design))
    """
    current_x = float(ctx.contact_patch_center[Axis.X])
    design_x = float(ctx.design_contact_patch_center[Axis.X])
    return -(current_x - design_x)


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
