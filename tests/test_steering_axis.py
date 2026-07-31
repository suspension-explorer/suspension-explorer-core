"""Integration tests for analytical instantaneous steering axes."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.analysis import analyze_evaluated_sweep
from kinematics.core.rigid_motion import ScrewAxisStatus
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.sweep import solve_evaluated_sweep

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "geometry_file",
    ["geometry.yaml", "macpherson_geometry.yaml"],
    ids=("double_wishbone", "macpherson"),
)
def test_corner_bump_sweep_has_valid_rack_partial_axis_at_every_step(
    geometry_file: str,
) -> None:
    suspension = load_geometry(DATA_DIR / geometry_file)
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert len(evaluated.instantaneous_steering_axes) == len(evaluated.states)
    for frame_results in evaluated.instantaneous_steering_axes:
        assert len(frame_results) == 1
        assert frame_results[0].status is ScrewAxisStatus.VALID
        assert frame_results[0].axis is not None

    midpoint = len(evaluated.states) // 2
    state = evaluated.states[midpoint]
    axis = evaluated.instantaneous_steering_axes[midpoint][0].axis
    assert axis is not None
    assert isinstance(suspension, CornerSuspension)
    lower_key, upper_key = suspension.steering_axis_points()
    lower = state.get(lower_key).data
    physical_direction = state.get(upper_key).data - lower
    physical_direction /= np.linalg.norm(physical_direction)
    assert abs(float(np.dot(axis.direction.data, physical_direction))) > 0.999
    assert np.linalg.norm(np.cross(axis.point.data - lower, physical_direction)) < 6.0


def test_shared_rack_returns_one_axis_per_upright_with_expected_symmetry() -> None:
    suspension = load_geometry(DATA_DIR / "axle_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "axle_sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    for frame_results in evaluated.instantaneous_steering_axes:
        assert [result.upright_label for result in frame_results] == [
            "Left Upright",
            "Right Upright",
        ]
        assert all(result.status is ScrewAxisStatus.VALID for result in frame_results)
    left, right = evaluated.instantaneous_steering_axes[len(evaluated.states) // 2]
    assert left.axis is not None
    assert right.axis is not None
    assert left.axis.point.data == pytest.approx(
        right.axis.point.data * np.array([1.0, -1.0, 1.0]),
        abs=1e-4,
    )
    assert abs(left.axis.pitch) == pytest.approx(abs(right.axis.pitch), rel=1e-6)


def test_unsteered_suspension_has_aligned_empty_axis_frames() -> None:
    suspension = load_geometry(DATA_DIR / "trailing_arm_coilover_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "trailing_arm_sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert evaluated.instantaneous_steering_axes == tuple(() for _ in evaluated.states)


def test_metrics_and_axes_share_one_tangent_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.sweep as sweep_module

    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    original = sweep_module.compute_sweep_tangents
    call_count = 0

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep_module, "compute_sweep_tangents", counted)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert call_count == 1
    assert evaluated.metrics.tangent_solve_infos is not None
    assert all(evaluated.instantaneous_steering_axes)


def test_tangent_failure_keeps_explicit_axis_results_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.sweep as sweep_module

    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)

    def fail_tangents(*_args, **_kwargs):
        raise RuntimeError("synthetic tangent failure")

    monkeypatch.setattr(sweep_module, "compute_sweep_tangents", fail_tangents)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert all(evaluated.metrics.rows)
    assert all(
        results[0].status is ScrewAxisStatus.TANGENT_UNAVAILABLE
        for results in evaluated.instantaneous_steering_axes
    )
    categories = {issue.category for issue in evaluated.diagnostics}
    assert "derivatives" in categories
    assert "steering_axis" in categories


def test_structured_analysis_exposes_axis_and_fit_diagnostics() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep)

    analysis = analyze_evaluated_sweep(suspension, sweep, evaluated)

    result = analysis.frames[len(analysis.frames) // 2].instantaneous_steering_axes[0]
    assert result.status is ScrewAxisStatus.VALID
    assert result.point is not None
    assert result.direction is not None
    assert result.pitch is not None
    assert result.angular_rate is not None
    assert result.fit_rms is not None
    assert result.fit_max is not None
    assert result.fit_rank == 6
    assert result.point_count >= 3
