"""Double-wishbone physical and isolated virtual steering-axis comparison."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.coordinates import actuator_coordinate_matches
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.screw_axis import InstantaneousScrewAxis
from kinematics.core.steering_axis import (
    SteeringResponseStatus,
    compute_steering_response_tangent,
)
from kinematics.core.sweep import compute_sweep_tangents, solve_evaluated_sweep

DATA_DIR = Path(__file__).parent / "data"
GEOMETRY = "axle_geometry_rocker_damper.yaml"
STEERING_METRIC_KEYS = (
    "caster",
    "kpi",
    "steering_axis_offset_ground",
    "scrub_radius",
    "mechanical_trail",
)


def _solve(sweep_name: str):
    """Solve the damper-equipped mirrored axle with one comparison sweep."""
    suspension = load_geometry(DATA_DIR / GEOMETRY)
    sweep = load_sweep(DATA_DIR / sweep_name, suspension)
    return suspension, sweep, solve_evaluated_sweep(suspension, sweep)


def _point_line_distance(
    point: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> float:
    """Return the perpendicular distance from a point to an infinite line."""
    return float(np.linalg.norm(np.cross(point - line_point, line_direction)))


def test_steering_response_axis_is_recomputed_per_frame() -> None:
    """Every stored response axis must follow its current solved configuration."""
    _suspension, _sweep, evaluated = _solve("axle_steer_moving_dampers_sweep.yaml")

    left_results = [frame[0] for frame in evaluated.steering_response_axes]
    assert all(result.status is SteeringResponseStatus.VALID for result in left_results)
    assert all(result.axis is not None for result in left_results)

    axes = [cast(InstantaneousScrewAxis, result.axis) for result in left_results]
    points = np.asarray([axis.point.data for axis in axes])
    directions = np.asarray([axis.direction.data for axis in axes])

    assert float(np.max(np.ptp(points, axis=0))) > 1.0
    assert float(np.max(np.ptp(directions, axis=0))) > 1e-4


def test_wheel_centre_height_control_does_not_contaminate_virtual_metrics() -> None:
    """Authored jacking remains real, but the suspension hold excludes it."""
    suspension, sweep, evaluated = _solve("axle_steer_sweep.yaml")
    authored_tangents = compute_sweep_tangents(suspension, sweep, evaluated.states)
    midpoint = len(evaluated.states) // 2
    steering_coordinate = suspension.steering_actuator_coordinate()
    assert steering_coordinate is not None
    authored_rack_tangent = next(
        field
        for field in authored_tangents.per_step[midpoint]
        if actuator_coordinate_matches(steering_coordinate, field.target)
    )

    lower_key = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    upper_key = PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD)
    wheel_centre_key = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
    lower_rate = authored_rack_tangent.rate(lower_key)
    upper_rate = authored_rack_tangent.rate(upper_key)

    assert np.linalg.norm(lower_rate) > 0.05
    assert lower_rate == pytest.approx(upper_rate, abs=2e-5)
    assert lower_rate[:2] == pytest.approx(np.zeros(2), abs=2e-8)
    assert authored_rack_tangent.rate(wheel_centre_key)[2] == pytest.approx(
        0.0,
        abs=2e-10,
    )

    isolated = compute_steering_response_tangent(
        suspension,
        evaluated.states[midpoint],
    )
    assert isolated.status is SteeringResponseStatus.VALID
    assert isolated.tangent is not None
    assert isolated.tangent.rate(lower_key) == pytest.approx(np.zeros(3), abs=2e-10)
    assert isolated.tangent.rate(upper_key) == pytest.approx(np.zeros(3), abs=2e-10)

    row = evaluated.metrics.rows[midpoint]
    assert isinstance(row, AxleMetricRows)
    for side in (Side.LEFT, Side.RIGHT):
        for key in STEERING_METRIC_KEYS:
            assert row.corners[side][f"{key}_virtual"] == pytest.approx(
                row.corners[side][key],
                abs=1e-8,
            )


def test_moving_dampers_recover_physical_axis_at_every_state() -> None:
    """The hold uses current lengths while the authored path moves them."""
    suspension, _sweep, evaluated = _solve("axle_steer_moving_dampers_sweep.yaml")

    left_damper_lengths = []
    for state, frame_axes, raw_row in zip(
        evaluated.states,
        evaluated.steering_response_axes,
        evaluated.metrics.rows,
        strict=True,
    ):
        assert isinstance(raw_row, AxleMetricRows)
        isolated = compute_steering_response_tangent(suspension, state)
        assert isolated.status is SteeringResponseStatus.VALID
        assert isolated.tangent is not None

        for side, result in zip((Side.LEFT, Side.RIGHT), frame_axes, strict=True):
            assert result.status is SteeringResponseStatus.VALID
            assert result.axis is not None
            axis = result.axis
            lower_key = PointRef(side, PointID.LOWER_WISHBONE_OUTBOARD)
            upper_key = PointRef(side, PointID.UPPER_WISHBONE_OUTBOARD)
            lower = state.get(lower_key).data
            upper = state.get(upper_key).data
            physical_direction = upper - lower
            physical_direction /= np.linalg.norm(physical_direction)

            assert abs(float(np.dot(axis.direction.data, physical_direction))) == (
                pytest.approx(1.0, abs=1e-10)
            )
            assert (
                _point_line_distance(
                    lower,
                    axis.point.data,
                    axis.direction.data,
                )
                < 1e-8
            )
            assert (
                _point_line_distance(
                    upper,
                    axis.point.data,
                    axis.direction.data,
                )
                < 1e-8
            )
            assert axis.pitch == pytest.approx(0.0, abs=1e-8)
            assert isolated.tangent.rate(lower_key) == pytest.approx(
                np.zeros(3), abs=2e-10
            )
            assert isolated.tangent.rate(upper_key) == pytest.approx(
                np.zeros(3), abs=2e-10
            )

            for key in STEERING_METRIC_KEYS:
                assert raw_row.corners[side][f"{key}_virtual"] == pytest.approx(
                    raw_row.corners[side][key],
                    abs=1e-8,
                )

        damper_chassis = state.get(PointRef(Side.LEFT, PointID.DAMPER_CHASSIS))
        damper_rocker = state.get(PointRef(Side.LEFT, PointID.DAMPER_ROCKER))
        left_damper_lengths.append((damper_rocker - damper_chassis).norm())

    assert max(left_damper_lengths) - min(left_damper_lengths) > 10.0


def test_same_state_has_same_axis_under_different_authored_target_bases() -> None:
    """The design-state result depends on the hold, not authored sweep inputs."""
    _suspension_a, _sweep_a, wheel_held = _solve("axle_steer_sweep.yaml")
    _suspension_b, _sweep_b, damper_driven = _solve(
        "axle_steer_moving_dampers_sweep.yaml"
    )
    axes_a = wheel_held.steering_response_axes[len(wheel_held.states) // 2]
    axes_b = damper_driven.steering_response_axes[len(damper_driven.states) // 2]

    for result_a, result_b in zip(axes_a, axes_b, strict=True):
        assert result_a.axis is not None
        assert result_b.axis is not None
        assert result_a.axis.point.data == pytest.approx(
            result_b.axis.point.data,
            abs=2e-5,
        )
        assert abs(
            float(np.dot(result_a.axis.direction.data, result_b.axis.direction.data))
        ) == pytest.approx(1.0, abs=1e-12)
        assert result_a.axis.pitch == pytest.approx(result_b.axis.pitch, abs=1e-10)


def test_upper_wishbone_option_is_equivalent_to_canonical_fixed_travel() -> None:
    """Both wishbone angles span the same local jounce coordinate."""
    suspension = load_geometry(DATA_DIR / GEOMETRY)
    sweep = load_sweep(DATA_DIR / "axle_steer_sweep.yaml", suspension)
    canonical = solve_evaluated_sweep(suspension, sweep)
    upper = solve_evaluated_sweep(
        suspension,
        replace(
            sweep,
            suspension_hold_id="upper_wishbone_angle",
        ),
    )

    for canonical_row, upper_row in zip(
        canonical.metrics.rows,
        upper.metrics.rows,
        strict=True,
    ):
        assert isinstance(canonical_row, AxleMetricRows)
        assert isinstance(upper_row, AxleMetricRows)
        for side in (Side.LEFT, Side.RIGHT):
            for key in STEERING_METRIC_KEYS:
                assert upper_row.corners[side][f"{key}_virtual"] == pytest.approx(
                    canonical_row.corners[side][f"{key}_virtual"],
                    abs=1e-8,
                )


def test_upright_pushrod_damper_option_is_a_deliberately_different_probe() -> None:
    """Fixed damper length introduces wishbone travel for upright actuation."""
    suspension = load_geometry(DATA_DIR / GEOMETRY)
    sweep = load_sweep(DATA_DIR / "axle_steer_sweep.yaml", suspension)
    diagnostic = solve_evaluated_sweep(
        suspension,
        replace(sweep, suspension_hold_id="damper_length"),
    )

    deltas = []
    for row in diagnostic.metrics.rows:
        assert isinstance(row, AxleMetricRows)
        for side in (Side.LEFT, Side.RIGHT):
            for key in STEERING_METRIC_KEYS:
                virtual = row.corners[side][f"{key}_virtual"]
                physical = row.corners[side][key]
                assert virtual is not None and physical is not None
                deltas.append(abs(virtual - physical))

    assert max(deltas) > 1.0
