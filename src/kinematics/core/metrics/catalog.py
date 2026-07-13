"""
Metric catalog.

Defines the ordered set of corner-level metrics and their export column names.
This is the single place to add, remove, or reorder exported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
from kinematics.core.metrics.units import MetricUnit

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext
    from kinematics.core.suspensions.base import Suspension


@dataclass(frozen=True)
class MetricDefinition:
    """
    A single metric: its export column name, computation function, and units.

    Attributes:
        column_name: Stable unit-independent metric identity (e.g. "camber").
        compute: Function mapping a MetricContext to the metric value.
        label: Human-readable display name (e.g. "Camber").
        unit: Structured physical unit for the value.
    """

    column_name: str
    compute: Callable[["MetricContext"], float | None]
    label: str
    unit: MetricUnit


def _build_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Build the default corner metric catalog.

    Imports are deferred to avoid circular dependencies at module level.
    """
    from kinematics.core.metrics.angles import (
        calculate_camber,
        calculate_caster,
        calculate_kpi,
        calculate_roadwheel_angle,
    )
    from kinematics.core.metrics.anti_geometry import (
        calculate_anti_dive_pct,
        calculate_anti_lift_pct,
        calculate_anti_squat_pct,
        calculate_svsa_angle,
    )
    from kinematics.core.metrics.steering_geometry import (
        calculate_mechanical_trail,
        calculate_scrub_radius,
    )
    from kinematics.core.metrics.swing_arms import (
        calculate_fvsa_length,
        calculate_svsa_length,
    )
    from kinematics.core.metrics.travel import (
        calculate_damper_length,
        calculate_half_track,
        calculate_wheel_travel,
    )
    def _ic_coord(attr: str, axis: Axis) -> Callable[["MetricContext"], float | None]:
        def extract(ctx: "MetricContext") -> float | None:
            ic = getattr(ctx, attr)
            return None if ic is None else float(ic[axis])

        return extract

    return (
        MetricDefinition("camber", calculate_camber, "Camber", MetricUnit.DEG),
        MetricDefinition("caster", calculate_caster, "Caster", MetricUnit.DEG),
        MetricDefinition("kpi", calculate_kpi, "KPI", MetricUnit.DEG),
        MetricDefinition(
            "scrub_radius", calculate_scrub_radius, "Scrub Radius", MetricUnit.MM
        ),
        MetricDefinition(
            "mechanical_trail",
            calculate_mechanical_trail,
            "Mechanical Trail",
            MetricUnit.MM,
        ),
        MetricDefinition(
            "roadwheel_angle",
            calculate_roadwheel_angle,
            "Roadwheel Angle",
            MetricUnit.DEG,
        ),
        MetricDefinition(
            "svic_x", _ic_coord("side_view_ic", Axis.X), "SVIC X", MetricUnit.MM
        ),
        MetricDefinition(
            "svic_z", _ic_coord("side_view_ic", Axis.Z), "SVIC Z", MetricUnit.MM
        ),
        MetricDefinition(
            "svsa_length", calculate_svsa_length, "SVSA Length", MetricUnit.MM
        ),
        MetricDefinition(
            "fvic_y", _ic_coord("front_view_ic", Axis.Y), "FVIC Y", MetricUnit.MM
        ),
        MetricDefinition(
            "fvic_z", _ic_coord("front_view_ic", Axis.Z), "FVIC Z", MetricUnit.MM
        ),
        MetricDefinition(
            "fvsa_length", calculate_fvsa_length, "FVSA Length", MetricUnit.MM
        ),
        MetricDefinition(
            "wheel_travel", calculate_wheel_travel, "Wheel Travel", MetricUnit.MM
        ),
        MetricDefinition(
            "half_track",
            calculate_half_track,
            "Half-Track",
            MetricUnit.MM,
        ),
        MetricDefinition(
            "damper_length",
            calculate_damper_length,
            "Damper Length",
            MetricUnit.MM,
        ),
        MetricDefinition(
            "svsa_angle", calculate_svsa_angle, "SVSA Angle", MetricUnit.DEG
        ),
        MetricDefinition(
            "anti_dive",
            calculate_anti_dive_pct,
            "Anti-Dive",
            MetricUnit.PERCENT,
        ),
        MetricDefinition(
            "anti_lift",
            calculate_anti_lift_pct,
            "Anti-Lift",
            MetricUnit.PERCENT,
        ),
        MetricDefinition(
            "anti_squat",
            calculate_anti_squat_pct,
            "Anti-Squat",
            MetricUnit.PERCENT,
        ),
    )


def get_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Return the default ordered corner metric catalog.
    """
    return _build_default_corner_metrics()


def get_default_corner_derivative_metrics(
    suspension: "Suspension",
) -> tuple[DerivativeMetricDefinition, ...]:
    """Declare derivative metrics common to every supported corner."""
    from kinematics.core.metrics import kernels
    from kinematics.core.metrics.derivatives import (
        CallableScalarResponse,
        DerivativeMetricDefinition,
        DualPositions,
        PointCoordinateResponse,
    )
    from kinematics.core.primitives.dual import DualScalar
    side_sign = suspension.side.lateral_sign
    hub_z_driver = PointCoordinateResponse.from_world_axis(
        PointID.WHEEL_CENTER,
        Axis.Z,
        name="hub_z",
        unit=MetricUnit.MM,
    )
    trackrod_inboard_y_driver = PointCoordinateResponse.from_world_axis(
        PointID.TRACKROD_INBOARD,
        Axis.Y,
        name="trackrod_inboard_y",
        unit=MetricUnit.MM,
    )

    def response(
        function: Callable[[DualPositions], object],
        name: str,
        unit: MetricUnit,
    ) -> CallableScalarResponse:
        def evaluate(positions: DualPositions) -> DualScalar:
            result = function(positions)
            assert isinstance(result, DualScalar)
            return result

        return CallableScalarResponse(evaluate, name=name, unit=unit)

    return (
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.camber_deg(positions, side_sign),
                "camber",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.toe_deg(positions, side_sign),
                "roadwheel_angle",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(kernels.caster_deg, "caster", MetricUnit.DEG),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.kpi_deg(positions, side_sign),
                "kpi",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=PointCoordinateResponse.from_axis(
                PointID.CONTACT_PATCH_CENTER,
                (0.0, side_sign, 0.0),
                name="half_track",
                unit=MetricUnit.MM,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=PointCoordinateResponse.from_axis(
                PointID.WHEEL_CENTER,
                (1.0, 0.0, 0.0),
                name="wheel_center_x",
                unit=MetricUnit.MM,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.toe_deg(positions, side_sign),
                "roadwheel_angle",
                MetricUnit.DEG,
            ),
            driver=trackrod_inboard_y_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.camber_deg(positions, side_sign),
                "camber",
                MetricUnit.DEG,
            ),
            driver=trackrod_inboard_y_driver,
        ),
    )
