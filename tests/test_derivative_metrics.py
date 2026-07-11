"""Unit tests for the declarative scalar derivative engine."""

import numpy as np
import pytest

from kinematics.core.dual import DualScalar, dot
from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.geometry import Point3
from kinematics.core.types import PointTarget, PointTargetAxis
from kinematics.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDisplacementMagnitudeResponse,
    PointDistanceResponse,
    evaluate_derivative_metrics,
)
from kinematics.sensitivity import TangentField
from kinematics.state import SuspensionState


def _state_and_tangent() -> tuple[SuspensionState, TangentField]:
    point_a = PointID.AXLE_INBOARD
    point_b = PointID.AXLE_OUTBOARD
    state = SuspensionState(
        positions={
            point_a: Point3([3.0, 4.0, 0.0]),
            point_b: Point3([0.0, 0.0, 0.0]),
        },
        free_points={point_a, point_b},
    )
    target = PointTarget(
        point_id=point_a,
        direction=PointTargetAxis(Axis.X),
        value=3.0,
        mode=TargetPositionMode.ABSOLUTE,
    )
    tangent = TangentField(
        target_index=0,
        target=target,
        velocities={
            point_a: np.array([2.0, 1.0, 0.0]),
            point_b: np.array([0.5, 0.0, 0.0]),
        },
    )
    return state, tangent


def test_distance_response_per_custom_axis_driver() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        column_name="distance_per_x",
        unit="mm/mm",
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (2.0, 0.0, 0.0),
        ),
    )

    # Distance derivative is dot([3,4,0], [1.5,1,0]) / 5 = 1.7.
    # The normalized X driver changes at 2.0, so d(distance)/d(driver) = 0.85.
    assert definition.evaluate(state, tangent) == pytest.approx(0.85)


def test_displacement_magnitude_response() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        column_name="displacement_per_y",
        unit="mm/mm",
        response=PointDisplacementMagnitudeResponse.from_reference(
            PointID.AXLE_INBOARD,
            Point3([0.0, 0.0, 0.0]),
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.Y,
        ),
    )

    # d(|[3,4,0]|) = dot([3,4,0], [2,1,0]) / 5 = 2.
    assert definition.evaluate(state, tangent) == pytest.approx(2.0)


def test_callable_response() -> None:
    state, tangent = _state_and_tangent()

    def response(positions) -> DualScalar:
        result = dot(
            positions[PointID.AXLE_INBOARD],
            np.array([1.0, 1.0, 0.0]),
        )
        assert isinstance(result, DualScalar)
        return result

    definition = DerivativeMetricDefinition(
        column_name="callable_per_x",
        unit="1",
        response=CallableScalarResponse(response),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (1.0, 0.0, 0.0),
        ),
    )

    assert definition.evaluate(state, tangent) == pytest.approx(1.5)


def test_zero_rate_driver_is_rejected() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        column_name="distance_per_z",
        unit="mm/mm",
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (0.0, 0.0, 1.0),
        ),
    )

    with pytest.raises(ValueError, match="zero-rate driver"):
        definition.evaluate(state, tangent)


def test_zero_displacement_magnitude_is_rejected() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        column_name="displacement_per_x",
        unit="mm/mm",
        response=PointDisplacementMagnitudeResponse.from_reference(
            PointID.AXLE_INBOARD,
            state.positions[PointID.AXLE_INBOARD],
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (1.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="undefined at zero displacement"):
        definition.evaluate(state, tangent)


def test_zero_point_distance_is_rejected() -> None:
    state, tangent = _state_and_tangent()
    state.positions[PointID.AXLE_OUTBOARD] = state.positions[
        PointID.AXLE_INBOARD
    ].copy()
    definition = DerivativeMetricDefinition(
        column_name="distance_per_x",
        unit="mm/mm",
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
        ),
    )

    with pytest.raises(ValueError, match="undefined at zero length"):
        definition.evaluate(state, tangent)


def test_displacement_reference_is_immutable() -> None:
    response = PointDisplacementMagnitudeResponse.from_reference(
        PointID.AXLE_INBOARD,
        Point3([0.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="read-only"):
        response.reference[0] = 1.0


def test_multi_tangent_selection_prefers_strongest_driver_rate() -> None:
    state, tangent = _state_and_tangent()
    weaker = TangentField(
        target_index=1,
        target=tangent.target,
        velocities={PointID.AXLE_INBOARD: np.array([0.5, 10.0, 0.0])},
    )
    unrelated = TangentField(
        target_index=2,
        target=PointTarget(
            point_id=PointID.AXLE_OUTBOARD,
            direction=PointTargetAxis(Axis.X),
            value=0.0,
            mode=TargetPositionMode.ABSOLUTE,
        ),
        velocities={PointID.AXLE_INBOARD: np.array([100.0, 100.0, 0.0])},
    )
    definition = DerivativeMetricDefinition(
        column_name="scaled_response",
        unit="mm/mm",
        scale=2.0,
        response=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (0.0, 1.0, 0.0),
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
        ),
    )

    row = evaluate_derivative_metrics(
        [definition],
        state,
        [weaker, unrelated, tangent],
    )

    # The matching tangent with X rate 2 wins over the matching tangent at 0.5.
    assert list(row) == ["scaled_response"]
    assert row["scaled_response"] == pytest.approx(1.0)


def test_distance_driver_requires_explicit_driving_point() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        column_name="coordinate_per_distance",
        unit="mm/mm",
        response=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
        ),
        driver=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ),
    )

    with pytest.raises(ValueError, match="explicit driving point"):
        definition.evaluate_from_tangents(state, [tangent])


def test_equal_strength_matching_tangents_are_rejected() -> None:
    state, tangent = _state_and_tangent()
    tied = TangentField(
        target_index=1,
        target=tangent.target,
        velocities={PointID.AXLE_INBOARD: np.array([-2.0, 3.0, 0.0])},
    )
    definition = DerivativeMetricDefinition(
        column_name="coordinate_per_x",
        unit="mm/mm",
        response=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.Y,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
        ),
    )

    with pytest.raises(ValueError, match="Ambiguous derivative driver"):
        definition.evaluate_from_tangents(state, [tangent, tied])
