"""Additional rigid-kinematic metric definitions and sign conventions."""

from math import atan, atan2, degrees
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import AxlePosition, PointID
from kinematics.core.metrics.anti_geometry import (
    calculate_anti_dive_pct,
    calculate_braking_anti_angle,
    calculate_braking_anti_ratio,
    calculate_traction_anti_angle,
    calculate_traction_anti_ratio,
)
from kinematics.core.metrics.axle_metrics import _ackermann_from_angles
from kinematics.core.metrics.catalog import get_default_corner_derivative_metrics
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.main import (
    AxleMetricRows,
    compute_metrics_for_state,
    compute_metrics_for_state_from_suspension,
)
from kinematics.core.metrics.steering import (
    SteeringAxis,
    calculate_steering_axis_lateral_offset_at_wheel_center,
    calculate_steering_axis_longitudinal_offset_at_wheel_center,
)
from kinematics.core.metrics.travel import (
    calculate_contact_patch_lateral_migration,
    calculate_wheel_center_recession,
)
from kinematics.core.primitives.geometry import Direction3, Point3, Vector3
from kinematics.core.primitives.point_ref import Side
from kinematics.core.road import RoadPlane
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner.base import CornerSuspension

TEST_DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("side_sign", "wheel_y", "axis_y", "wheel_axis_y"),
    [(1.0, 100.0, 80.0, 1.0), (-1.0, -100.0, -80.0, -1.0)],
)
def test_wheel_center_steering_axis_offsets_use_documented_signs(
    side_sign: float,
    wheel_y: float,
    axis_y: float,
    wheel_axis_y: float,
) -> None:
    wheel_center = Point3((0.0, wheel_y, 300.0))
    road = RoadPlane.horizontal_at(Point3((0.0, wheel_y, 0.0)))
    axis = SteeringAxis.from_point_direction(
        Point3((10.0, axis_y, 0.0)),
        Direction3((0.0, 0.0, 1.0)),
    )
    wheel_axis = Direction3((0.0, wheel_axis_y, 0.0))

    longitudinal = calculate_steering_axis_longitudinal_offset_at_wheel_center(
        axis, road, wheel_center, wheel_axis, side_sign
    )
    lateral = calculate_steering_axis_lateral_offset_at_wheel_center(
        axis, road, wheel_center, wheel_axis, side_sign
    )

    assert longitudinal == pytest.approx(-10.0)
    assert lateral == pytest.approx(20.0)


def test_ackermann_percentage_matches_perfect_and_parallel_steer() -> None:
    wheelbase = 2500.0
    track = 1500.0
    inner_radius = 5000.0
    inner_angle = degrees(atan2(wheelbase, inner_radius))
    outer_angle = degrees(atan2(wheelbase, inner_radius + track))

    assert _ackermann_from_angles(
        inner_angle, outer_angle, track, wheelbase
    ) == pytest.approx(100.0)
    assert _ackermann_from_angles(20.0, 20.0, track, wheelbase) == pytest.approx(0.0)
    assert _ackermann_from_angles(0.0, 0.0, track, wheelbase) is None


def test_recession_and_lateral_migration_are_design_relative_and_inward_positive() -> (
    None
):
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert isinstance(suspension, CornerSuspension)
    assert suspension.config is not None
    state = suspension.initial_state().copy()
    state[PointID.WHEEL_CENTER] = state.get(PointID.WHEEL_CENTER) + Vector3(
        (-5.0, 0.0, 0.0)
    )
    state[PointID.WHEEL_CONTACT_CENTRE] = state.get(
        PointID.WHEEL_CONTACT_CENTRE
    ) + Vector3((0.0, -3.0, 0.0))
    context = MetricContext(state, suspension, suspension.config)

    assert calculate_wheel_center_recession(context) == pytest.approx(5.0)
    assert calculate_contact_patch_lateral_migration(context) == pytest.approx(3.0)


def test_coilover_exposes_distinct_spring_length_and_ratio_response() -> None:
    suspension = load_geometry(TEST_DATA / "multi_link_geometry.yaml")
    assert isinstance(suspension, CornerSuspension)
    assert suspension.config is not None
    state = suspension.initial_state()

    metrics = compute_metrics_for_state(state, suspension, suspension.config)
    definitions = {
        definition.column_name
        for definition in get_default_corner_derivative_metrics(suspension)
    }

    assert metrics["spring_length"] == pytest.approx(metrics["damper_length"])
    assert "deriv_spring_length_wrt_hub_z" in definitions


def test_anti_geometry_exposes_force_ratio_and_angle_forms() -> None:
    contact = Point3((0.0, 800.0, 0.0))
    context = cast(
        MetricContext,
        SimpleNamespace(
            config=SimpleNamespace(
                axle_position=AxlePosition.FRONT,
                front_brake_bias=0.6,
                driven_axle=AxlePosition.FRONT,
            ),
            road=RoadPlane.horizontal_at(contact),
            side_view_ic=Point3((-500.0, 800.0, 300.0)),
            wheel_contact_centre=contact,
            wheel_center=Point3((0.0, 800.0, 250.0)),
            cg_position=Point3((1250.0, 0.0, 450.0)),
            wheelbase=2500.0,
        ),
    )

    anti_dive_pct = calculate_anti_dive_pct(context)
    braking_ratio = calculate_braking_anti_ratio(context)
    braking_angle = calculate_braking_anti_angle(context)
    traction_ratio = calculate_traction_anti_ratio(context)
    traction_angle = calculate_traction_anti_angle(context)

    assert anti_dive_pct is not None
    assert braking_ratio is not None
    assert braking_ratio == pytest.approx(anti_dive_pct / 100.0)
    assert braking_angle == pytest.approx(degrees(atan(braking_ratio)))
    assert traction_ratio is not None
    assert traction_angle == pytest.approx(degrees(atan(traction_ratio)))


def test_u_bar_exposes_end_displacement_and_motion_ratio_response() -> None:
    axle = load_geometry(TEST_DATA / "axle_geometry_rocker.yaml")
    assert isinstance(axle, AxleSuspension)

    metrics = compute_metrics_for_state_from_suspension(axle.initial_state(), axle)
    definitions = {
        definition.column_name for definition in axle.derivative_metric_definitions()
    }

    assert isinstance(metrics, AxleMetricRows)
    for side in (Side.LEFT, Side.RIGHT):
        assert metrics.corners[side]["arb_end_displacement"] == pytest.approx(0.0)
        assert f"deriv_arb_end_displacement_wrt_hub_z_{side.name.lower()}" in (
            definitions
        )


def test_new_state_metrics_are_exported_with_units() -> None:
    suspension = load_geometry(TEST_DATA / "axle_geometry.yaml")
    assert isinstance(suspension, AxleSuspension)
    metrics = compute_metrics_for_state_from_suspension(
        suspension.initial_state(), suspension
    )

    assert isinstance(metrics, AxleMetricRows)
    assert "ackermann_percentage" in metrics.axle
    for side in (Side.LEFT, Side.RIGHT):
        row = metrics.corners[side]
        assert "steering_axis_longitudinal_offset_wheel_center" in row
        assert "steering_axis_lateral_offset_wheel_center" in row
        assert row["wheel_center_recession"] == pytest.approx(0.0)
        assert row["contact_patch_lateral_migration"] == pytest.approx(0.0)
