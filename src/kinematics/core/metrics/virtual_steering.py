"""Metric adapters for a motion-derived virtual steering axis.

The metric context stores the common geometric axis established from the
instantaneous rack-partial screw axis. These adapters deliberately sit beside,
rather than replace, the established physical-pivot steering metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kinematics.core.metrics.steering_axis_geometry as axis_geometry

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext


def calculate_virtual_caster(ctx: MetricContext) -> float | None:
    """Return chassis-relative caster from the virtual steering axis."""
    axis = ctx.virtual_steering_axis
    return None if axis is None else axis_geometry.calculate_caster(axis)


def calculate_virtual_kpi(ctx: MetricContext) -> float | None:
    """Return inward-positive KPI from the virtual steering axis."""
    axis = ctx.virtual_steering_axis
    return None if axis is None else axis_geometry.calculate_kpi(axis, ctx.side_sign)


def calculate_virtual_steering_axis_offset_ground(
    ctx: MetricContext,
) -> float | None:
    """Return inward-positive road offset from the virtual steering axis."""
    axis = ctx.virtual_steering_axis
    return (
        None
        if axis is None
        else axis_geometry.calculate_steering_axis_offset_at_ground(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
            ctx.wheel_axis,
            ctx.side_sign,
        )
    )


def calculate_virtual_scrub_radius(ctx: MetricContext) -> float | None:
    """Return unsigned road-plane scrub radius from the virtual axis."""
    axis = ctx.virtual_steering_axis
    return (
        None
        if axis is None
        else axis_geometry.calculate_scrub_radius(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
        )
    )


def calculate_virtual_mechanical_trail(ctx: MetricContext) -> float | None:
    """Return wheel-relative mechanical trail from the virtual axis."""
    axis = ctx.virtual_steering_axis
    return (
        None
        if axis is None
        else axis_geometry.calculate_mechanical_trail(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
            ctx.wheel_axis,
            ctx.side_sign,
        )
    )
