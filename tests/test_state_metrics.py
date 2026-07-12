from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.types import SweepConfig
from kinematics.io import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.metrics.anti_geometry import (
    calculate_anti_dive_pct,
    calculate_anti_lift_pct,
    calculate_anti_squat_pct,
)
from kinematics.metrics.context import MetricContext
from kinematics.metrics.main import AxleMetricRows, MetricRow, compute_metrics_for_state
from kinematics.schema.config import SuspensionConfig

TEST_DATA = Path(__file__).parent / "data"


def test_config_validates_anti_geometry_inputs() -> None:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert suspension.config is not None

    configured = suspension.config.model_copy(
        update={
            "axle_position": "front",
            "front_brake_bias": 0.6,
            "driven_axle": "rear",
        }
    )

    assert configured.axle_position == "front"
    assert configured.front_brake_bias == pytest.approx(0.6)
    assert configured.driven_axle == "rear"


@pytest.mark.parametrize("bias", [-0.01, 1.01])
def test_config_rejects_invalid_front_brake_bias(bias: float) -> None:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert suspension.config is not None
    data = suspension.config.model_dump()
    data["front_brake_bias"] = bias

    with pytest.raises(ValueError, match="front_brake_bias must be in"):
        SuspensionConfig.model_validate(data)


@pytest.mark.parametrize("field", ["axle_position", "driven_axle"])
def test_config_rejects_unknown_axle_selection(field: str) -> None:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert suspension.config is not None
    data = suspension.config.model_dump()
    data[field] = "middle"

    with pytest.raises(ValueError, match=field):
        SuspensionConfig.model_validate(data)


def test_design_state_travel_and_position_metrics() -> None:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert suspension.config is not None
    state = suspension.initial_state()

    metrics = compute_metrics_for_state(state, suspension, suspension.config)

    assert metrics["wheel_travel"] == pytest.approx(0.0)
    assert metrics["half_track"] == pytest.approx(
        abs(float(state.get(PointID.CONTACT_PATCH_CENTER)[1]))
    )
    assert metrics["damper_length"] is None
    assert metrics["anti_dive"] is None
    assert metrics["anti_lift"] is None
    assert metrics["anti_squat"] is None


def test_coilover_damper_length_matches_mount_distance() -> None:
    suspension = load_geometry(TEST_DATA / "corner_strut_geometry.yaml")
    assert suspension.config is not None
    state = suspension.initial_state()

    metrics = compute_metrics_for_state(state, suspension, suspension.config)
    expected = (state.get(PointID.STRUT_TOP) - state.get(PointID.STRUT_BOTTOM)).norm()

    assert metrics["damper_length"] == pytest.approx(expected)


def test_coilover_sweep_emits_corner_derivative_metrics() -> None:
    suspension = load_geometry(TEST_DATA / "corner_strut_geometry.yaml")
    assert suspension.config is not None
    sweep = parse_sweep_file(TEST_DATA / "sweep.yaml")
    states, _ = solve_sweep(suspension, sweep)

    result = compute_sweep_metrics(suspension, sweep, states)

    assert result.derivative_error is None
    assert len(result.rows) == len(states)
    assert result.tangent_solve_infos is not None
    for state, row in zip(states, result.rows):
        # A corner sweep yields plain metric rows, never axle row bundles.
        assert not isinstance(row, AxleMetricRows)
        assert "deriv_camber_wrt_hub_z" in row
        assert "deriv_damper_length_wrt_hub_z" in row
        assert row["deriv_damper_length_wrt_hub_z"] is not None
        non_derivative = compute_metrics_for_state(
            state,
            suspension,
            suspension.config,
        )
        assert list(row.items())[: len(non_derivative)] == list(non_derivative.items())

    midpoint = len(states) // 2
    midpoint_targets = [target_sweep[midpoint] for target_sweep in sweep.target_sweeps]
    bump_target = next(
        target for target in midpoint_targets if target.point_id == PointID.WHEEL_CENTER
    )
    rack_target = next(
        target
        for target in midpoint_targets
        if target.point_id == PointID.TRACKROD_INBOARD
    )
    finite_difference_step = 0.25
    finite_difference_sweep = SweepConfig(
        target_sweeps=[
            [
                bump_target._replace(value=bump_target.value - finite_difference_step),
                bump_target._replace(value=bump_target.value + finite_difference_step),
            ],
            [rack_target, rack_target],
        ]
    )
    finite_difference_states, _ = solve_sweep(
        suspension,
        finite_difference_sweep,
    )
    left_metrics = compute_metrics_for_state(
        finite_difference_states[0], suspension, suspension.config
    )
    right_metrics = compute_metrics_for_state(
        finite_difference_states[1], suspension, suspension.config
    )
    left_travel = left_metrics["wheel_travel"]
    right_travel = right_metrics["wheel_travel"]
    left_camber = left_metrics["camber"]
    right_camber = right_metrics["camber"]
    left_damper = left_metrics["damper_length"]
    right_damper = right_metrics["damper_length"]
    assert left_travel is not None and right_travel is not None
    assert left_camber is not None and right_camber is not None
    assert left_damper is not None and right_damper is not None

    travel_delta = right_travel - left_travel
    camber_difference = right_camber - left_camber
    expected_camber_derivative = camber_difference / travel_delta
    midpoint_row = cast(MetricRow, result.rows[midpoint])
    assert midpoint_row["deriv_camber_wrt_hub_z"] == pytest.approx(
        expected_camber_derivative,
        rel=2e-3,
    )

    damper_difference = right_damper - left_damper
    expected_damper_derivative = damper_difference / travel_delta
    assert midpoint_row["deriv_damper_length_wrt_hub_z"] == pytest.approx(
        expected_damper_derivative,
        rel=2e-3,
    )


def test_tangent_failure_is_visible_and_preserves_base_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    sweep = parse_sweep_file(TEST_DATA / "sweep.yaml")
    states, _ = solve_sweep(suspension, sweep)

    def fail_tangents(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic tangent failure")

    monkeypatch.setattr("kinematics.main.compute_sweep_tangents", fail_tangents)
    result = compute_sweep_metrics(suspension, sweep, states)

    assert result.derivative_error == "RuntimeError: synthetic tangent failure"
    assert result.tangent_solve_infos is None
    assert len(result.rows) == len(states)
    first_row = cast(MetricRow, result.rows[0])
    assert "camber" in first_row
    assert "deriv_camber_wrt_hub_z" not in first_row


def _anti_context(
    *,
    svic_x: float,
    axle_position: str,
    front_brake_bias: float = 0.6,
    driven_axle: str | None = None,
) -> MetricContext:
    """Build the minimal synthetic context consumed by anti metrics."""
    return cast(
        MetricContext,
        SimpleNamespace(
            config=SimpleNamespace(
                axle_position=axle_position,
                front_brake_bias=front_brake_bias,
                driven_axle=driven_axle,
            ),
            side_view_ic=Point3([svic_x, 800.0, 300.0]),
            contact_patch_center=Point3([0.0, 800.0, 0.0]),
            wheel_center=Point3([0.0, 800.0, 300.0]),
            cg_position=Point3([1250.0, 0.0, 450.0]),
            wheelbase=2500.0,
        ),
    )


def test_anti_dive_and_lift_follow_axle_and_svic_signs() -> None:
    front_behind = _anti_context(svic_x=-500.0, axle_position="front")
    front_ahead = _anti_context(svic_x=500.0, axle_position="front")
    rear_ahead = _anti_context(svic_x=500.0, axle_position="rear")

    anti_dive_behind = calculate_anti_dive_pct(front_behind)
    anti_dive_ahead = calculate_anti_dive_pct(front_ahead)
    anti_lift_ahead = calculate_anti_lift_pct(rear_ahead)

    assert anti_dive_behind is not None and anti_dive_behind > 0.0
    assert anti_dive_behind == pytest.approx(200.0)
    assert anti_dive_ahead is not None and anti_dive_ahead < 0.0
    assert calculate_anti_lift_pct(front_behind) is None
    assert anti_lift_ahead is not None and anti_lift_ahead > 0.0


def test_anti_squat_requires_the_configured_driven_axle() -> None:
    driven_rear = _anti_context(
        svic_x=500.0,
        axle_position="rear",
        driven_axle="rear",
    )
    non_driven_rear = _anti_context(
        svic_x=500.0,
        axle_position="rear",
        driven_axle="front",
    )

    assert calculate_anti_squat_pct(driven_rear) is not None
    assert calculate_anti_squat_pct(non_driven_rear) is None
