"""Regression coverage for shared physical and virtual steering metrics."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.metrics.catalog import (
    get_default_corner_metrics,
    get_virtual_steering_metrics,
)
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.steering import (
    SteeringAxis,
    calculate_caster,
    calculate_kpi,
    calculate_mechanical_trail,
    calculate_scrub_radius,
    calculate_steering_axis_offset_at_ground,
)
from kinematics.core.primitives.geometry import Direction3
from kinematics.core.road import RoadPlane
from kinematics.core.suspensions.corner.base import CornerSuspension

TEST_DATA = Path(__file__).parent / "data"
STEERING_KEYS = (
    "caster",
    "kpi",
    "steering_axis_offset_ground",
    "scrub_radius",
    "mechanical_trail",
)


def _context(*, steering_axis: SteeringAxis | None = None) -> MetricContext:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert isinstance(suspension, CornerSuspension)
    assert suspension.config is not None
    state = suspension.initial_state()
    road = RoadPlane.through(
        Direction3([0.0, -0.15, 1.0]),
        state.get(PointID.WHEEL_CONTACT_CENTRE),
    )
    return MetricContext(
        state,
        suspension,
        suspension.config,
        road=road,
        steering_axis=steering_axis,
    )


def _common_outputs(ctx: MetricContext) -> tuple[float, ...]:
    caster = calculate_caster(ctx.steering_axis)
    kpi = calculate_kpi(ctx.steering_axis, ctx.side_sign)
    offset = calculate_steering_axis_offset_at_ground(
        ctx.steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
        ctx.wheel_axis,
        ctx.side_sign,
    )
    scrub = calculate_scrub_radius(
        ctx.steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
    )
    trail = calculate_mechanical_trail(
        ctx.steering_axis,
        ctx.road,
        ctx.wheel_contact_centre,
        ctx.wheel_axis,
        ctx.side_sign,
    )
    assert caster is not None
    assert kpi is not None
    assert offset is not None
    assert scrub is not None
    assert trail is not None
    return caster, kpi, offset, scrub, trail


def test_metric_context_establishes_physical_axis_by_default() -> None:
    ctx = _context()
    lower_id, upper_id = ctx.suspension.steering_axis_points()
    lower = ctx.state.get(lower_id)
    upper = ctx.state.get(upper_id)

    axis = ctx.steering_axis
    assert axis.point.data == pytest.approx(lower.data)
    assert axis.direction.data == pytest.approx((upper - lower).normalize().data)

    pivots = ctx.steering_axis_pivots
    assert pivots[0].data == pytest.approx(lower.data)
    assert pivots[1].data == pytest.approx(upper.data)
    expected_intersection = axis.intersect_road(ctx.road)
    actual_intersection = ctx.steering_axis_ground_intersection
    assert expected_intersection is not None
    assert actual_intersection is not None
    assert actual_intersection.data == pytest.approx(expected_intersection.data)


def test_same_metric_definitions_accept_physical_or_virtual_axis_context() -> None:
    physical_ctx = _context()
    physical_axis = physical_ctx.steering_axis
    equivalent_point = physical_axis.point + physical_axis.direction.vector() * 125.0
    virtual_axis = SteeringAxis.from_unoriented_line(
        equivalent_point,
        -physical_axis.direction,
    )
    virtual_ctx = _context(steering_axis=virtual_axis)

    physical_definitions = {
        metric.column_name: metric for metric in get_default_corner_metrics()
    }
    virtual_definitions = {
        metric.column_name: metric for metric in get_virtual_steering_metrics()
    }
    physical_outputs = tuple(
        physical_definitions[key].compute(physical_ctx) for key in STEERING_KEYS
    )
    virtual_outputs = tuple(
        virtual_definitions[f"{key}_virtual"].compute(virtual_ctx)
        for key in STEERING_KEYS
    )

    assert virtual_ctx.steering_axis is virtual_axis
    assert physical_outputs == pytest.approx(_common_outputs(physical_ctx))
    assert virtual_outputs == pytest.approx(_common_outputs(virtual_ctx))
    assert virtual_outputs == pytest.approx(physical_outputs)
