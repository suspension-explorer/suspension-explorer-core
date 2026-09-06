"""
Per-state suspension travel metrics.

These metrics report wheel travel, design-relative migration, half-track, and
installed spring and damper lengths.
Every value is a scalar in millimeters. Coordinates follow the ISO 8855
vehicle-axis orientation (X forward, Y left, Z up), expressed in chassis
space. These metrics do not use the local road plane or world space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_wheel_travel(ctx: MetricContext) -> float | None:
    """
    Vertical wheel travel in mm relative to the design condition.

    Defined as the current wheel-center chassis Z minus its design chassis Z.
    Positive means the wheel has moved up relative to the chassis (bump);
    negative means droop. This is not displacement normal to the road or world
    vertical.

        wheel_travel = WC_z(current) - WC_z(design)
    """
    current_z = float(ctx.wheel_center[Axis.Z])
    design_z = float(ctx.design_wheel_center[Axis.Z])
    return current_z - design_z


def calculate_half_track(ctx: MetricContext) -> float | None:
    """
    Half-track at this corner in mm.

    Half-track is the lateral distance of the wheel contact centre from the
    vehicle centerline, measured as the magnitude of its chassis Y coordinate.
    Unlike the axle-level track metric, it is not resolved along the current
    local road plane.

        half_track = |CP_y(current)|
    """
    return abs(float(ctx.wheel_contact_centre[Axis.Y]))


def calculate_wheel_center_recession(ctx: MetricContext) -> float:
    """Return rearward wheel-centre travel from design in mm.

    Chassis X is forward, so recession is design X minus current X. Positive
    means the wheel centre moved rearward.
    """
    return float(ctx.design_wheel_center[Axis.X] - ctx.wheel_center[Axis.X])


def calculate_contact_patch_lateral_migration(ctx: MetricContext) -> float:
    """Return inboard contact-patch travel from design in mm.

    The lateral displacement is folded by side so positive always means
    movement toward the vehicle centreline.
    """
    design_y = float(ctx.design_wheel_contact_centre[Axis.Y])
    current_y = float(ctx.wheel_contact_centre[Axis.Y])
    return ctx.side_sign * (design_y - current_y)


def calculate_damper_length(ctx: MetricContext) -> float | None:
    """
    Installed spring/damper (coilover) length in mm.

    Both mounts are represented in chassis coordinates and the length is their
    Euclidean separation, which is invariant under a rigid change to world
    coordinates. It does not reference the road plane. Only defined when the
    suspension carries a strut group; otherwise None.

        damper_length = |STRUT_TOP - STRUT_BOTTOM|
    """
    damper_points = ctx.suspension.damper_points()
    if damper_points is None:
        return None
    top = ctx.state.get(damper_points[0])
    bottom = ctx.state.get(damper_points[1])
    # Euclidean distance between the two mounts (a Point3 - Point3 -> Vector3).
    return float((top - bottom).norm())


def calculate_spring_length(ctx: MetricContext) -> float | None:
    """Return installed linear spring length in mm, when present.

    A coilover shares endpoints with its damper but remains a distinct physical
    response. Torsion springs have no linear installed length and return None.
    """
    spring_points = ctx.suspension.spring_points()
    if spring_points is None:
        return None
    top = ctx.state.get(spring_points[0])
    bottom = ctx.state.get(spring_points[1])
    return float((top - bottom).norm())
