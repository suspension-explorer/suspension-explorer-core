from math import atan2, degrees, sqrt

import numpy as np
import pytest

from kinematics.core.metrics.steering import (
    SteeringAxis,
    calculate_caster,
    calculate_kpi,
    calculate_mechanical_trail,
    calculate_scrub_radius,
    calculate_steering_axis_offset_at_ground,
)
from kinematics.core.primitives.geometry import Direction3, Point3, Vector3
from kinematics.core.road import RoadPlane


def test_axis_construction_preserves_source_orientation_and_owns_data() -> None:
    point = Point3([1.0, 2.0, 3.0])
    direction = Vector3([-1.0, -2.0, 4.0])

    axis = SteeringAxis.from_point_direction(point, direction)
    from_pivots = SteeringAxis.from_pivots(point, point + direction)

    assert axis.direction.data == pytest.approx(Direction3(direction).data)
    assert from_pivots.direction.data == pytest.approx(axis.direction.data)
    with pytest.raises(ValueError, match="read-only"):
        axis.point.data[0] = 10.0
    with pytest.raises(ValueError, match="read-only"):
        axis.direction.data[0] = 10.0


def test_axis_angles_are_invariant_to_reversed_input_direction() -> None:
    point = Point3([0.0, 0.0, 0.0])
    direction = Vector3([-1.0, -2.0, 4.0])
    forward = SteeringAxis.from_unoriented_line(point, direction)
    reverse = SteeringAxis.from_unoriented_line(point, -direction)

    expected_caster = degrees(atan2(1.0, 4.0))
    expected_left_kpi = degrees(atan2(2.0, 4.0))
    assert calculate_caster(forward) == pytest.approx(expected_caster)
    assert calculate_caster(reverse) == pytest.approx(expected_caster)
    assert calculate_kpi(forward, 1.0) == pytest.approx(expected_left_kpi)
    assert calculate_kpi(reverse, 1.0) == pytest.approx(expected_left_kpi)


def test_physical_pivots_preserve_lower_to_upper_direction() -> None:
    lower = Point3([0.0, 0.0, 10.0])
    upper = Point3([1.0, 2.0, 6.0])

    axis = SteeringAxis.from_pivots(lower, upper)

    assert axis.direction.data == pytest.approx((upper - lower).normalize().data)


def test_horizontal_axis_has_no_resolvable_caster_or_kpi() -> None:
    axis = SteeringAxis.from_unoriented_line(
        Point3([0.0, 0.0, 0.0]),
        Vector3([1.0, 2.0, 0.0]),
    )

    assert calculate_caster(axis) is None
    assert calculate_kpi(axis, 1.0) is None


def test_axis_intersects_road_independently_of_input_direction_sign() -> None:
    road = RoadPlane.horizontal_at(Point3([0.0, 0.0, 0.0]))
    point = Point3([10.0, 20.0, 30.0])
    direction = Vector3([1.0, -2.0, -3.0])

    forward = SteeringAxis.from_point_direction(point, direction).intersect_road(road)
    reverse = SteeringAxis.from_point_direction(point, -direction).intersect_road(road)

    assert forward is not None
    assert reverse is not None
    assert forward.data == pytest.approx([20.0, 0.0, 0.0])
    assert reverse.data == pytest.approx(forward.data)


def test_axis_road_metrics_follow_iso_sign_conventions_on_both_sides() -> None:
    road = RoadPlane.horizontal_at(Point3([0.0, 0.0, 0.0]))
    left_axis = SteeringAxis.from_point_direction(
        Point3([10.0, 95.0, 20.0]),
        Vector3([0.0, 0.0, 1.0]),
    )
    left_contact = Point3([0.0, 100.0, 0.0])

    assert calculate_scrub_radius(left_axis, road, left_contact) == pytest.approx(
        sqrt(125.0)
    )
    assert calculate_steering_axis_offset_at_ground(
        left_axis,
        road,
        left_contact,
        Direction3([0.0, 1.0, 0.0]),
        1.0,
    ) == pytest.approx(5.0)
    assert calculate_mechanical_trail(
        left_axis,
        road,
        left_contact,
        Direction3([0.0, 1.0, 0.0]),
        1.0,
    ) == pytest.approx(10.0)

    right_axis = SteeringAxis.from_point_direction(
        Point3([10.0, -95.0, 20.0]),
        Vector3([0.0, 0.0, -1.0]),
    )
    right_contact = Point3([0.0, -100.0, 0.0])
    assert calculate_steering_axis_offset_at_ground(
        right_axis,
        road,
        right_contact,
        Direction3([0.0, -1.0, 0.0]),
        -1.0,
    ) == pytest.approx(5.0)
    assert calculate_mechanical_trail(
        right_axis,
        road,
        right_contact,
        Direction3([0.0, -1.0, 0.0]),
        -1.0,
    ) == pytest.approx(10.0)


def test_parallel_or_unresolvable_axis_inputs_are_rejected_or_unavailable() -> None:
    road = RoadPlane.horizontal_at(Point3([0.0, 0.0, 0.0]))
    parallel = SteeringAxis.from_point_direction(
        Point3([10.0, 20.0, 30.0]),
        Vector3([1.0, 0.0, 0.0]),
    )
    contact = Point3([0.0, 0.0, 0.0])

    assert parallel.intersect_road(road) is None
    assert calculate_scrub_radius(parallel, road, contact) is None
    assert (
        calculate_steering_axis_offset_at_ground(
            SteeringAxis.from_point_direction(
                Point3([0.0, 0.0, 1.0]), Vector3([0.0, 0.0, 1.0])
            ),
            road,
            contact,
            Vector3([0.0, 0.0, 1.0]),
            1.0,
        )
        is None
    )
    assert (
        calculate_mechanical_trail(
            SteeringAxis.from_point_direction(
                Point3([0.0, 0.0, 1.0]), Vector3([0.0, 0.0, 1.0])
            ),
            road,
            contact,
            Vector3([0.0, 0.0, 1.0]),
            1.0,
        )
        is None
    )
    with pytest.raises(ValueError, match="non-zero"):
        SteeringAxis.from_point_direction(contact, Vector3([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        SteeringAxis.from_point_direction(
            contact,
            Vector3(np.array([np.nan, 0.0, 1.0])),
        )
