"""Tests for the structured high-level analysis API."""

from pathlib import Path

import pytest

from kinematics.analysis import analyze_sweep, initial_pose, sweep_parameters
from kinematics.io import load_geometry, load_sweep
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.metrics.main import AxleMetricRows, flatten_metric_rows

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corner_analysis():
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    return analyze_sweep(suspension, sweep)


@pytest.fixture(scope="module")
def axle_analysis():
    suspension = load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")
    sweep = load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", suspension)
    return analyze_sweep(suspension, sweep)


def test_corner_analysis_is_complete_and_structured(corner_analysis) -> None:
    assert corner_analysis.steps == len(corner_analysis.frames) > 0
    assert corner_analysis.locations == []
    assert corner_analysis.corner_metric_keys == []
    first = corner_analysis.frames[0]
    assert "wheel_center" in first.positions
    assert first.metrics
    assert first.corner_metrics == {}
    assert first.solver.converged
    assert "setup" in corner_analysis.references
    assert list(corner_analysis.references["setup"].metrics) == (
        corner_analysis.metric_keys
    )


def test_axle_metrics_keep_corner_location_structural(axle_analysis) -> None:
    assert axle_analysis.locations == ["left", "right"]
    assert axle_analysis.corner_metric_keys
    frame = axle_analysis.frames[0]
    assert set(frame.corner_metrics) == {"left", "right"}
    assert all(frame.corner_metrics.values())
    assert frame.metrics
    assert not any(key.endswith("_left") for key in frame.corner_metrics["left"])
    assert set(axle_analysis.references["setup"].corner_metrics) == {
        "left",
        "right",
    }
    expected_metadata = set(axle_analysis.corner_metric_keys) | set(
        axle_analysis.metric_keys
    )
    assert {display.key for display in axle_analysis.metric_display} == (
        expected_metadata
    )


def test_axle_metric_rows_flatten_only_at_export_boundary() -> None:
    suspension = load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")
    sweep = load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", suspension)
    states, _ = solve_sweep(suspension, sweep)
    row = compute_sweep_metrics(suspension, sweep, states).rows[0]
    assert isinstance(row, AxleMetricRows)
    flat = flatten_metric_rows(row.axle, row.corners)
    assert any(key.endswith("_left") for key in flat)
    assert any(key.endswith("_right") for key in flat)


def test_sweep_parameters_preserve_side_and_axis() -> None:
    suspension = load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")
    sweep = load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", suspension)
    parameters = {(p.point, p.axis, p.side) for p in sweep_parameters(sweep)}
    assert ("left_wheel_center", "z", "left") in parameters
    assert ("right_wheel_center", "z", "right") in parameters


def test_initial_pose_contains_display_geometry() -> None:
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    pose = initial_pose(suspension)
    assert "wheel_center" in pose.positions
    assert pose.wheel is not None
    assert pose.links
    assert pose.wheel_anchors


def test_tangent_failure_is_visible_without_losing_metrics(monkeypatch) -> None:
    import kinematics.main as main

    def fail_tangents(*_args, **_kwargs):
        raise RuntimeError("synthetic tangent failure")

    monkeypatch.setattr(main, "compute_sweep_tangents", fail_tangents)
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    analysis = analyze_sweep(suspension, sweep)
    assert all(frame.metrics for frame in analysis.frames)
    issues = [
        issue for issue in analysis.diagnostics if issue.category == "derivatives"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "synthetic tangent failure" in issues[0].message


def test_diagnostic_failure_is_advisory(monkeypatch) -> None:
    import kinematics.analysis as analysis_module

    def fail_diagnostics(*_args, **_kwargs):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(analysis_module, "diagnose_sweep", fail_diagnostics)
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    analysis = analyze_sweep(suspension, sweep)
    assert analysis.frames
    assert not [
        issue for issue in analysis.diagnostics if issue.category != "derivatives"
    ]
