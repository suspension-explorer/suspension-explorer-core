"""
Metric catalog.

Defines the ordered set of corner-level metrics and their export column names.
This is the single place to add, remove, or reorder exported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


@dataclass(frozen=True)
class MetricDefinition:
    """
    A single metric: its export column name, computation function, and units.

    Attributes:
        column_name: Stable export/column identifier (e.g. "camber_deg").
        compute: Function mapping a MetricContext to the metric value.
        label: Human-readable display name (e.g. "Camber").
        unit: Physical unit symbol for the value (e.g. "deg", "mm").
    """

    column_name: str
    compute: Callable[["MetricContext"], float | None]
    label: str
    unit: str


def _build_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Build the default corner metric catalog.

    Imports are deferred to avoid circular dependencies at module level.
    """
    from kinematics.core.enums import Axis
    from kinematics.metrics.angles import (
        calculate_camber,
        calculate_caster,
        calculate_kpi,
        calculate_roadwheel_angle,
    )
    from kinematics.metrics.anti_geometry import (
        calculate_anti_dive_pct,
        calculate_anti_lift_pct,
        calculate_anti_squat_pct,
        calculate_svsa_angle,
    )
    from kinematics.metrics.steering_geometry import (
        calculate_mechanical_trail,
        calculate_scrub_radius,
    )
    from kinematics.metrics.swing_arms import (
        calculate_fvsa_length,
        calculate_svsa_length,
    )
    from kinematics.metrics.travel import (
        calculate_damper_length,
        calculate_half_track_change,
        calculate_wheel_recession,
        calculate_wheel_travel,
    )

    def _ic_coord(attr: str, axis: Axis) -> Callable[["MetricContext"], float | None]:
        def extract(ctx: "MetricContext") -> float | None:
            ic = getattr(ctx, attr)
            return None if ic is None else float(ic[axis])

        return extract

    return (
        MetricDefinition("camber_deg", calculate_camber, "Camber", "deg"),
        MetricDefinition("caster_deg", calculate_caster, "Caster", "deg"),
        MetricDefinition("kpi_deg", calculate_kpi, "KPI", "deg"),
        MetricDefinition(
            "scrub_radius_mm", calculate_scrub_radius, "Scrub Radius", "mm"
        ),
        MetricDefinition(
            "mechanical_trail_mm", calculate_mechanical_trail, "Mechanical Trail", "mm"
        ),
        MetricDefinition(
            "roadwheel_angle_deg", calculate_roadwheel_angle, "Roadwheel Angle", "deg"
        ),
        MetricDefinition(
            "svic_x_mm", _ic_coord("side_view_ic", Axis.X), "SVIC X", "mm"
        ),
        MetricDefinition(
            "svic_z_mm", _ic_coord("side_view_ic", Axis.Z), "SVIC Z", "mm"
        ),
        MetricDefinition("svsa_length_mm", calculate_svsa_length, "SVSA Length", "mm"),
        MetricDefinition(
            "fvic_y_mm", _ic_coord("front_view_ic", Axis.Y), "FVIC Y", "mm"
        ),
        MetricDefinition(
            "fvic_z_mm", _ic_coord("front_view_ic", Axis.Z), "FVIC Z", "mm"
        ),
        MetricDefinition("fvsa_length_mm", calculate_fvsa_length, "FVSA Length", "mm"),
        MetricDefinition(
            "wheel_travel_mm", calculate_wheel_travel, "Wheel Travel", "mm"
        ),
        MetricDefinition(
            "half_track_change_mm",
            calculate_half_track_change,
            "Half-Track Change",
            "mm",
        ),
        MetricDefinition(
            "wheel_recession_mm",
            calculate_wheel_recession,
            "Wheel Recession",
            "mm",
        ),
        MetricDefinition(
            "damper_length_mm", calculate_damper_length, "Damper Length", "mm"
        ),
        MetricDefinition("svsa_angle_deg", calculate_svsa_angle, "SVSA Angle", "deg"),
        MetricDefinition("anti_dive_pct", calculate_anti_dive_pct, "Anti-Dive", "%"),
        MetricDefinition("anti_lift_pct", calculate_anti_lift_pct, "Anti-Lift", "%"),
        MetricDefinition("anti_squat_pct", calculate_anti_squat_pct, "Anti-Squat", "%"),
    )


def get_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Return the default ordered corner metric catalog.
    """
    return _build_default_corner_metrics()
