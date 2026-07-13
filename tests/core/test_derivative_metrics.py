"""Unit tests for the declarative scalar derivative engine."""

from typing import cast

import numpy as np
import pytest

from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDisplacementMagnitudeResponse,
    PointDistanceResponse,
    evaluate_derivative_metrics,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.dual import DualScalar, dot
from kinematics.core.primitives.enums import Axis, PointID, TargetPositionMode
from kinematics.core.primitives.geometry import Point3
from kinematics.core.sensitivity import TangentField
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import PointTarget, PointTargetAxis


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
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            name="point_distance",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (2.0, 0.0, 0.0),
            name="wheel_x",
            unit=MetricUnit.MM,
        ),
    )

    # Distance derivative is dot([3,4,0], [1.5,1,0]) / 5 = 1.7.
    # The normalized X driver changes at 2.0, so d(distance)/d(driver) = 0.85.
    assert definition.evaluate(state, tangent) == pytest.approx(0.85)


def test_displacement_magnitude_response() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        response=PointDisplacementMagnitudeResponse.from_reference(
            PointID.AXLE_INBOARD,
            Point3([0.0, 0.0, 0.0]),
            name="point_displacement",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.Y,
            name="wheel_y",
            unit=MetricUnit.MM,
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
        response=CallableScalarResponse(
            response,
            name="coordinate_sum",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (1.0, 0.0, 0.0),
            name="wheel_x",
            unit=MetricUnit.MM,
        ),
    )

    assert definition.evaluate(state, tangent) == pytest.approx(1.5)


def test_zero_rate_driver_is_rejected() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            name="point_distance",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (0.0, 0.0, 1.0),
            name="wheel_z",
            unit=MetricUnit.MM,
        ),
    )

    with pytest.raises(ValueError, match="zero-rate driver"):
        definition.evaluate(state, tangent)


def test_zero_displacement_magnitude_is_rejected() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        response=PointDisplacementMagnitudeResponse.from_reference(
            PointID.AXLE_INBOARD,
            state.positions[PointID.AXLE_INBOARD],
            name="point_displacement",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (1.0, 0.0, 0.0),
            name="wheel_x",
            unit=MetricUnit.MM,
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
        response=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            name="point_distance",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name="wheel_x",
            unit=MetricUnit.MM,
        ),
    )

    with pytest.raises(ValueError, match="undefined at zero length"):
        definition.evaluate(state, tangent)


def test_displacement_reference_is_immutable() -> None:
    response = PointDisplacementMagnitudeResponse.from_reference(
        PointID.AXLE_INBOARD,
        Point3([0.0, 0.0, 0.0]),
        name="point_displacement",
        unit=MetricUnit.MM,
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
        scale=2.0,
        response=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            (0.0, 1.0, 0.0),
            name="response_y",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name="driver_x",
            unit=MetricUnit.MM,
        ),
    )

    row = evaluate_derivative_metrics(
        [definition],
        state,
        [weaker, unrelated, tangent],
    )

    # The matching tangent with X rate 2 wins over the matching tangent at 0.5.
    assert list(row) == ["deriv_response_y_wrt_driver_x"]
    assert row["deriv_response_y_wrt_driver_x"] == pytest.approx(1.0)
    assert definition.unit == MetricUnit.MM / MetricUnit.MM
    assert definition.unit.symbol == "mm/mm"


def test_distance_driver_requires_explicit_driving_point() -> None:
    state, tangent = _state_and_tangent()
    definition = DerivativeMetricDefinition(
        response=PointCoordinateResponse.from_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name="coordinate_x",
            unit=MetricUnit.MM,
        ),
        driver=PointDistanceResponse(
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            name="link_length",
            unit=MetricUnit.MM,
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
        response=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.Y,
            name="coordinate_y",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name="coordinate_x",
            unit=MetricUnit.MM,
        ),
    )

    with pytest.raises(ValueError, match="Ambiguous derivative driver"):
        definition.evaluate_from_tangents(state, [tangent, tied])


@pytest.mark.parametrize("name", ["UpperCase", "has-hyphen", "has space", "_x"])
def test_scalar_names_must_be_lowercase_snake_case(name: str) -> None:
    with pytest.raises(ValueError, match="lowercase snake-case"):
        PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name=name,
            unit=MetricUnit.MM,
        )


def test_scalar_units_must_be_supported_metric_units() -> None:
    with pytest.raises(TypeError, match="must be a MetricUnit"):
        PointCoordinateResponse.from_world_axis(
            PointID.AXLE_INBOARD,
            Axis.X,
            name="coordinate_x",
            unit=cast(MetricUnit, "  "),
        )
