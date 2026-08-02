"""Integration coverage for isolated virtual steering metrics."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.analysis import analyze_evaluated_sweep
from kinematics.core.enums import Scope
from kinematics.core.metrics.main import MetricRow
from kinematics.core.metrics.registry import MetricKind
from kinematics.core.sweep import solve_evaluated_sweep

DATA_DIR = Path(__file__).parent / "data"

VIRTUAL_KEYS = (
    "caster_virtual",
    "kpi_virtual",
    "steering_axis_offset_ground_virtual",
    "scrub_radius_virtual",
    "mechanical_trail_virtual",
)
PHYSICAL_STEERING_KEYS = (
    "caster",
    "kpi",
    "steering_axis_offset_ground",
    "scrub_radius",
    "mechanical_trail",
)
VIRTUAL_METADATA = {
    "caster_virtual": ("Caster, Virtual", "deg"),
    "kpi_virtual": ("KPI, Virtual", "deg"),
    "steering_axis_offset_ground_virtual": (
        "Steering-Axis Offset at Ground, Virtual",
        "mm",
    ),
    "scrub_radius_virtual": ("Scrub Radius, Virtual", "mm"),
    "mechanical_trail_virtual": ("Mechanical Trail, Virtual", "mm"),
}


def _evaluated_and_analysis(geometry_name: str, sweep_name: str):
    """Solve one fixture and return its core evaluation and analysis models."""
    suspension = load_geometry(DATA_DIR / geometry_name)
    sweep = load_sweep(DATA_DIR / sweep_name, suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep)
    return evaluated, analyze_evaluated_sweep(suspension, sweep, evaluated)


def _assert_finite_values(row: MetricRow, keys: tuple[str, ...]) -> None:
    """Assert that the selected metric values are defined finite scalars."""
    for key in keys:
        value = row[key]
        assert value is not None
        assert np.isfinite(value)


def test_steered_corner_emits_finite_virtual_metrics_and_metadata() -> None:
    _evaluated, analysis = _evaluated_and_analysis(
        "corner_rocker_damper_geometry.yaml",
        "corner_steer_bump_sweep.yaml",
    )
    frame = analysis.frames[len(analysis.frames) // 2]

    assert set(VIRTUAL_KEYS).issubset(frame.metrics)
    _assert_finite_values(frame.metrics, VIRTUAL_KEYS)
    assert set(PHYSICAL_STEERING_KEYS).issubset(frame.metrics)
    _assert_finite_values(frame.metrics, PHYSICAL_STEERING_KEYS)
    ordered_keys = tuple(frame.metrics)
    for physical_key, virtual_key in zip(
        PHYSICAL_STEERING_KEYS,
        VIRTUAL_KEYS,
        strict=True,
    ):
        assert ordered_keys.index(virtual_key) == ordered_keys.index(physical_key) + 1

    display = {item.key: item for item in analysis.metric_display}
    for key, (label, unit) in VIRTUAL_METADATA.items():
        assert key in analysis.metric_keys
        assert display[key].label == label
        assert display[key].unit.symbol == unit
        assert display[key].kind is MetricKind.STATE
        assert display[key].scope is Scope.CORNER

    setup = analysis.references["setup"]
    assert set(VIRTUAL_KEYS).issubset(setup.metrics)
    _assert_finite_values(setup.metrics, VIRTUAL_KEYS)


def test_steered_axle_maps_virtual_metrics_into_both_corner_rows() -> None:
    _evaluated, analysis = _evaluated_and_analysis(
        "axle_geometry_rocker_damper.yaml",
        "axle_steer_sweep.yaml",
    )
    frame = analysis.frames[len(analysis.frames) // 2]

    assert set(VIRTUAL_KEYS).issubset(analysis.corner_metric_keys)
    assert set(VIRTUAL_KEYS).isdisjoint(frame.metrics)
    assert set(frame.corner_metrics) == {"left", "right"}
    for row in frame.corner_metrics.values():
        assert set(VIRTUAL_KEYS).issubset(row)
        _assert_finite_values(row, VIRTUAL_KEYS)
        assert set(PHYSICAL_STEERING_KEYS).issubset(row)

    setup = analysis.references["setup"]
    for row in setup.corner_metrics.values():
        assert set(VIRTUAL_KEYS).issubset(row)
        _assert_finite_values(row, VIRTUAL_KEYS)


def test_authored_tangent_failure_does_not_disable_virtual_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.sweep as sweep_module

    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)

    def fail_tangents(*_args, **_kwargs):
        raise RuntimeError("synthetic tangent failure")

    monkeypatch.setattr(sweep_module, "compute_sweep_tangents", fail_tangents)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    for raw_row in evaluated.metrics.rows:
        row = cast(MetricRow, raw_row)
        assert set(VIRTUAL_KEYS).issubset(row)
        _assert_finite_values(row, VIRTUAL_KEYS)
        assert set(PHYSICAL_STEERING_KEYS).issubset(row)


def test_missing_isolation_definition_keeps_virtual_columns_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    monkeypatch.setattr(suspension, "steering_probe_catalogue", lambda: None)
    evaluated = solve_evaluated_sweep(suspension, sweep)
    analysis = analyze_evaluated_sweep(suspension, sweep, evaluated)

    for raw_row in evaluated.metrics.rows:
        row = cast(MetricRow, raw_row)
        assert set(VIRTUAL_KEYS).issubset(row)
        assert all(row[key] is None for key in VIRTUAL_KEYS)
    assert any(
        result.status.value == "no_isolation_definition"
        for frame in analysis.frames
        for result in frame.steering_response_axes
    )


def test_unsteered_suspension_does_not_advertise_virtual_metrics() -> None:
    _evaluated, analysis = _evaluated_and_analysis(
        "trailing_arm_coilover_geometry.yaml",
        "trailing_arm_sweep.yaml",
    )

    assert set(VIRTUAL_KEYS).isdisjoint(analysis.metric_keys)
    assert set(VIRTUAL_KEYS).isdisjoint(analysis.corner_metric_keys)
    assert set(VIRTUAL_KEYS).isdisjoint(item.key for item in analysis.metric_display)
    assert set(VIRTUAL_KEYS).isdisjoint(analysis.frames[0].metrics)
    assert set(VIRTUAL_KEYS).isdisjoint(analysis.references["setup"].metrics)


def test_virtual_metrics_are_independent_of_the_single_sweep_tangent_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.sweep as sweep_module

    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)
    original = sweep_module.compute_sweep_tangents
    call_count = 0

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep_module, "compute_sweep_tangents", counted)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert call_count == 1
    for raw_row in evaluated.metrics.rows:
        _assert_finite_values(cast(MetricRow, raw_row), VIRTUAL_KEYS)
