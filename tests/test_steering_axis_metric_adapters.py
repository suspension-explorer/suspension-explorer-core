"""Regression coverage for physical and virtual steering-axis adapters."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.metrics import steering_axis_geometry as common
from kinematics.core.metrics.angles import (
    calculate_caster as calculate_physical_caster,
)
from kinematics.core.metrics.angles import calculate_kpi as calculate_physical_kpi
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.steering_geometry import (
    calculate_mechanical_trail as calculate_physical_mechanical_trail,
)
from kinematics.core.metrics.steering_geometry import (
    calculate_scrub_radius as calculate_physical_scrub_radius,
)
from kinematics.core.metrics.steering_geometry import (
    calculate_steering_axis_offset_ground as calculate_physical_offset,
)
from kinematics.core.metrics.virtual_steering import (
    calculate_virtual_caster,
    calculate_virtual_kpi,
    calculate_virtual_mechanical_trail,
    calculate_virtual_scrub_radius,
    calculate_virtual_steering_axis_offset_ground,
)
from kinematics.core.primitives.geometry import Direction3
from kinematics.core.road import RoadPlane
from kinematics.core.suspensions.corner.base import CornerSuspension

TEST_DATA = Path(__file__).parent / "data"


def _context(*, with_equivalent_virtual_axis: bool = False) -> MetricContext:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert isinstance(suspension, CornerSuspension)
    assert suspension.config is not None
    state = suspension.initial_state()
    road = RoadPlane.through(
        Direction3([0.0, -0.15, 1.0]),
        state.get(PointID.WHEEL_CONTACT_CENTRE),
    )
    context = MetricContext(state, suspension, suspension.config, road=road)
    if not with_equivalent_virtual_axis:
        return context

    physical = context.physical_steering_axis
    equivalent_point = physical.point + physical.direction.vector() * 125.0
    virtual = common.SteeringAxis.from_unoriented_line(
        equivalent_point,
        -physical.direction,
    )
    return MetricContext(
        state,
        suspension,
        suspension.config,
        road=road,
        virtual_steering_axis=virtual,
    )


def _common_outputs(ctx: MetricContext, axis: common.SteeringAxis) -> tuple[float, ...]:
    caster = common.calculate_caster(axis)
    kpi = common.calculate_kpi(axis, ctx.side_sign)
    offset = common.calculate_steering_axis_offset_at_ground(
        axis,
        ctx.road,
        ctx.wheel_contact_centre,
        ctx.wheel_axis,
        ctx.side_sign,
    )
    scrub = common.calculate_scrub_radius(
        axis,
        ctx.road,
        ctx.wheel_contact_centre,
    )
    trail = common.calculate_mechanical_trail(
        axis,
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


def test_metric_context_builds_physical_axis_and_preserves_legacy_properties() -> None:
    ctx = _context()
    lower_id, upper_id = ctx.suspension.steering_axis_points()
    lower = ctx.state.get(lower_id)
    upper = ctx.state.get(upper_id)

    axis = ctx.physical_steering_axis
    assert axis.point.data == pytest.approx(lower.data)
    assert axis.direction.data == pytest.approx((upper - lower).normalize().data)
    assert ctx.steering_axis is axis.direction

    pivots = ctx.steering_axis_pivots
    assert pivots[0].data == pytest.approx(lower.data)
    assert pivots[1].data == pytest.approx(upper.data)
    expected_intersection = axis.intersect_road(ctx.road)
    actual_intersection = ctx.steering_axis_ground_intersection
    assert expected_intersection is not None
    assert actual_intersection is not None
    assert actual_intersection.data == pytest.approx(expected_intersection.data)


def test_physical_and_virtual_adapters_match_their_common_geometry_outputs() -> None:
    ctx = _context(with_equivalent_virtual_axis=True)
    virtual_axis = ctx.virtual_steering_axis
    assert virtual_axis is not None

    physical_outputs = (
        calculate_physical_caster(ctx),
        calculate_physical_kpi(ctx),
        calculate_physical_offset(ctx),
        calculate_physical_scrub_radius(ctx),
        calculate_physical_mechanical_trail(ctx),
    )
    virtual_outputs = (
        calculate_virtual_caster(ctx),
        calculate_virtual_kpi(ctx),
        calculate_virtual_steering_axis_offset_ground(ctx),
        calculate_virtual_scrub_radius(ctx),
        calculate_virtual_mechanical_trail(ctx),
    )

    assert physical_outputs == pytest.approx(
        _common_outputs(ctx, ctx.physical_steering_axis)
    )
    assert virtual_outputs == pytest.approx(_common_outputs(ctx, virtual_axis))
    assert virtual_outputs == pytest.approx(physical_outputs)
