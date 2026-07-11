"""Integration tests for basic full-axle composition."""

from pathlib import Path

import pytest

from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.io import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.suspensions.axle import DoubleWishboneAxleSuspension


def test_mirrored_axle_builds_two_explicit_corners(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")

    assert isinstance(axle, DoubleWishboneAxleSuspension)
    assert axle.side is Side.CENTER
    assert set(axle.corners) == {Side.LEFT, Side.RIGHT}
    assert axle.corners[Side.LEFT].side is Side.LEFT
    assert axle.corners[Side.RIGHT].side is Side.RIGHT
    assert len(axle.initial_state().positions) == 30


def test_basic_axle_sweep_solves_and_emits_suffixed_metrics(
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, DoubleWishboneAxleSuspension)
    sweep = parse_sweep_file(test_data_dir / "axle_sweep.yaml", axle)

    states, stats = solve_sweep(axle, sweep)
    metrics = compute_sweep_metrics(axle, sweep, states)

    assert len(states) == 5
    assert all(info.converged for info in stats)
    assert all(info.max_residual < 1e-3 for info in stats)
    assert metrics.derivative_error is None
    midpoint = metrics.rows[2]
    assert "camber_deg_left" in midpoint
    assert "camber_deg_right" in midpoint
    assert "trackrod_inboard_displacement_mm" in midpoint
    assert midpoint["heave_mm"] == pytest.approx(0.0, abs=1e-5)

    final = states[-1]
    left_z = final.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))[Axis.Z]
    right_z = final.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))[Axis.Z]
    assert left_z == pytest.approx(right_z, abs=1e-5)


def test_axle_targets_require_side(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, DoubleWishboneAxleSuspension)

    with pytest.raises(ValueError, match="requires side left or right"):
        axle.resolve_target_key(PointID.WHEEL_CENTER, None)
