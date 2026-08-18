"""
Metric catalog.

Defines the ordered set of corner-level metrics and their export column names.
This is the single place to add, remove, or reorder exported metrics.

Reference systems belong to each calculation's docstring rather than the
catalog metadata. Instant-centre coordinate extractors below report principal
chassis coordinates using the ISO 8855 vehicle-axis orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    DualPositions,
    PointCoordinateResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.dual import DualScalar

if TYPE_CHECKING:
    from kinematics.core.metrics.context import MetricContext
    from kinematics.core.suspensions.corner.base import CornerSuspension


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


def _build_steering_metrics(
    *,
    column_suffix: str = "",
    label_suffix: str = "",
) -> tuple[MetricDefinition, ...]:
    """Build one catalog family around the axis selected by the context."""
    from kinematics.core.metrics.steering import (
        calculate_caster,
        calculate_kpi,
        calculate_mechanical_trail,
        calculate_scrub_radius,
        calculate_steering_axis_offset_at_ground,
    )

    def caster(ctx: "MetricContext") -> float | None:
        axis = ctx.steering_axis
        if axis is None:
            return None
        return calculate_caster(axis)

    def kpi(ctx: "MetricContext") -> float | None:
        axis = ctx.steering_axis
        if axis is None:
            return None
        return calculate_kpi(axis, ctx.side_sign)

    def steering_axis_offset_at_ground(ctx: "MetricContext") -> float | None:
        axis = ctx.steering_axis
        if axis is None:
            return None
        return calculate_steering_axis_offset_at_ground(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
            ctx.wheel_axis,
            ctx.side_sign,
        )

    def scrub_radius(ctx: "MetricContext") -> float | None:
        axis = ctx.steering_axis
        if axis is None:
            return None
        return calculate_scrub_radius(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
        )

    def mechanical_trail(ctx: "MetricContext") -> float | None:
        axis = ctx.steering_axis
        if axis is None:
            return None
        return calculate_mechanical_trail(
            axis,
            ctx.road,
            ctx.wheel_contact_centre,
            ctx.wheel_axis,
            ctx.side_sign,
        )

    def definition(
        name: str,
        compute: Callable[["MetricContext"], float | None],
        label: str,
        unit: MetricUnit,
    ) -> MetricDefinition:
        return MetricDefinition(
            f"{name}{column_suffix}",
            compute,
            f"{label}{label_suffix}",
            unit,
        )

    return (
        definition("caster", caster, "Caster", MetricUnit.DEG),
        definition("kpi", kpi, "KPI", MetricUnit.DEG),
        definition(
            "steering_axis_offset_ground",
            steering_axis_offset_at_ground,
            "Steering-Axis Offset at Ground",
            MetricUnit.MM,
        ),
        definition("scrub_radius", scrub_radius, "Scrub Radius", MetricUnit.MM),
        definition(
            "mechanical_trail",
            mechanical_trail,
            "Mechanical Trail",
            MetricUnit.MM,
        ),
    )


def _build_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Build the default corner metric catalog.

    Imports are deferred to avoid circular dependencies at module level.
    """
    from kinematics.core.metrics.angles import (
        calculate_camber,
        calculate_steer,
        calculate_toe,
    )
    from kinematics.core.metrics.anti_geometry import (
        calculate_anti_dive_pct,
        calculate_anti_lift_pct,
        calculate_anti_squat_pct,
        calculate_svsa_angle,
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
            """Extract one chassis-axis coordinate from an instant centre."""
            ic = getattr(ctx, attr)
            return None if ic is None else float(ic[axis])

        return extract

    return (
        MetricDefinition("camber", calculate_camber, "Camber", MetricUnit.DEG),
        *_build_steering_metrics(),
        MetricDefinition(
            "toe_angle",
            calculate_toe,
            "Toe Angle",
            MetricUnit.DEG,
        ),
        MetricDefinition(
            "steer_angle",
            calculate_steer,
            "Steer Angle",
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


def get_virtual_steering_metrics() -> tuple[MetricDefinition, ...]:
    """Return additive labels for the common metrics on the virtual-axis context."""
    return _build_steering_metrics(
        column_suffix="_virtual",
        label_suffix=", Virtual",
    )


def physical_steering_metric_keys() -> frozenset[str]:
    """Column names of the steering family evaluated on the physical axis.

    Architectures without a physical steering axis omit exactly these columns
    while keeping their ``*_virtual`` counterparts.
    """
    return frozenset(metric.column_name for metric in _build_steering_metrics())


def get_default_corner_derivative_metrics(
    suspension: "CornerSuspension",
) -> tuple[DerivativeMetricDefinition, ...]:
    """
    Declare derivative metrics common to every supported corner.

    Point roles are resolved through the corner's role hooks. The common
    alignment responses use chassis-referenced angles, while their hub and rack
    drivers use chassis-axis coordinates. Differentiation does not introduce a
    road or world reference. Wheel-travel derivatives apply to every corner;
    rack-driven derivatives are omitted when no steering rack is installed.
    """
    side_sign = suspension.side.lateral_sign
    axle_inboard, axle_outboard = suspension.wheel_axis_points()
    steering_axis_points = suspension.steering_axis_points()
    rack_attachment = suspension.rack_attachment_point()
    hub_z_driver = PointCoordinateResponse.from_chassis_axis(
        PointID.WHEEL_CENTER,
        Axis.Z,
        name="hub_z",
        unit=MetricUnit.MM,
        label="Hub Z",
    )

    def response(
        function: Callable[[DualPositions], object],
        name: str,
        label: str,
        unit: MetricUnit,
    ) -> CallableScalarResponse:
        def evaluate(positions: DualPositions) -> DualScalar:
            result = function(positions)
            assert isinstance(result, DualScalar)
            return result

        return CallableScalarResponse(evaluate, name=name, unit=unit, label=label)

    definitions = [
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.camber_deg(
                    positions, side_sign, axle_inboard, axle_outboard
                ),
                "camber",
                "Camber",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.toe_deg(
                    positions, side_sign, axle_inboard, axle_outboard
                ),
                "toe_angle",
                "Toe Angle",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
        DerivativeMetricDefinition(
            response=response(
                lambda positions: kernels.steer_deg(
                    positions, side_sign, axle_inboard, axle_outboard
                ),
                "steer_angle",
                "Steer Angle",
                MetricUnit.DEG,
            ),
            driver=hub_z_driver,
        ),
    ]

    if steering_axis_points is not None:
        # Physical caster/KPI responses only exist for architectures with a
        # joint-to-joint steering axis; motion-derived axes report virtually.
        lower_pivot, upper_pivot = steering_axis_points
        definitions.extend(
            (
                DerivativeMetricDefinition(
                    response=response(
                        lambda positions: kernels.caster_deg(
                            positions, lower_pivot, upper_pivot
                        ),
                        "caster",
                        "Caster",
                        MetricUnit.DEG,
                    ),
                    driver=hub_z_driver,
                ),
                DerivativeMetricDefinition(
                    response=response(
                        lambda positions: kernels.kpi_deg(
                            positions, side_sign, lower_pivot, upper_pivot
                        ),
                        "kpi",
                        "KPI",
                        MetricUnit.DEG,
                    ),
                    driver=hub_z_driver,
                ),
            )
        )

    definitions.extend(
        (
            DerivativeMetricDefinition(
                response=PointCoordinateResponse.from_axis(
                    PointID.WHEEL_CONTACT_CENTRE,
                    (0.0, side_sign, 0.0),
                    name="half_track",
                    unit=MetricUnit.MM,
                    label="Half-Track",
                ),
                driver=hub_z_driver,
            ),
            DerivativeMetricDefinition(
                response=PointCoordinateResponse.from_axis(
                    PointID.WHEEL_CENTER,
                    (1.0, 0.0, 0.0),
                    name="wheel_center_x",
                    unit=MetricUnit.MM,
                    label="Wheel Center X",
                ),
                driver=hub_z_driver,
            ),
        )
    )

    if rack_attachment is not None:
        # Rack displacement is the rack attachment point chassis Y offset;
        # the corner constrains that point to translate along chassis Y.
        rack_displacement_driver = PointCoordinateResponse.from_chassis_axis(
            rack_attachment,
            Axis.Y,
            name="rack_displacement",
            unit=MetricUnit.MM,
            label="Rack Displacement",
        )
        definitions.extend(
            (
                DerivativeMetricDefinition(
                    response=response(
                        lambda positions: kernels.toe_deg(
                            positions, side_sign, axle_inboard, axle_outboard
                        ),
                        "toe_angle",
                        "Toe Angle",
                        MetricUnit.DEG,
                    ),
                    driver=rack_displacement_driver,
                ),
                DerivativeMetricDefinition(
                    response=response(
                        lambda positions: kernels.steer_deg(
                            positions, side_sign, axle_inboard, axle_outboard
                        ),
                        "steer_angle",
                        "Steer Angle",
                        MetricUnit.DEG,
                    ),
                    driver=rack_displacement_driver,
                ),
                DerivativeMetricDefinition(
                    response=response(
                        lambda positions: kernels.camber_deg(
                            positions, side_sign, axle_inboard, axle_outboard
                        ),
                        "camber",
                        "Camber",
                        MetricUnit.DEG,
                    ),
                    driver=rack_displacement_driver,
                ),
            )
        )

    return tuple(definitions)
