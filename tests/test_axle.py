"""Integration tests for basic full-axle composition."""

from pathlib import Path

import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics import (
    AxleMetricRows,
    compute_metrics_for_state_from_suspension,
    compute_metrics_for_sweep,
)
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import (
    ActuationDirect,
    DoubleWishboneSuspension,
)
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep


def test_mirrored_axle_builds_two_explicit_corners(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")

    assert isinstance(axle, AxleSuspension)
    assert axle.side is Side.CENTER
    assert set(axle.corners) == {Side.LEFT, Side.RIGHT}
    assert axle.corners[Side.LEFT].side is Side.LEFT
    assert axle.corners[Side.RIGHT].side is Side.RIGHT
    assert len(axle.initial_state().positions) == 30


def test_explicit_axle_uses_shared_mechanism_and_authored_right_geometry(
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_explicit.yaml")

    assert isinstance(axle, AxleSuspension)
    left = axle.corners[Side.LEFT]
    right = axle.corners[Side.RIGHT]
    assert isinstance(left, DoubleWishboneSuspension)
    assert isinstance(right, DoubleWishboneSuspension)
    assert isinstance(left.actuation, ActuationDirect)
    assert isinstance(right.actuation, ActuationDirect)
    assert left.actuation.spring_pickup_body == left.LOWER_WISHBONE_BODY
    assert right.actuation.spring_pickup_body == right.LOWER_WISHBONE_BODY
    assert right.hardpoints[PointID.AXLE_OUTBOARD][Axis.X] == pytest.approx(-30.0)


def _camber_shim_data(*, side: Side, setup_thickness: float) -> dict[str, object]:
    lateral_sign = side.lateral_sign
    return {
        "shim_face_point_a": {"x": -25.0, "y": 750.0 * lateral_sign, "z": 510.0},
        "shim_face_point_b": {"x": -25.0, "y": 750.0 * lateral_sign, "z": 490.0},
        "shim_face_normal": {"x": 0.0, "y": lateral_sign, "z": 0.0},
        "design_thickness": 30.0,
        "setup_thickness": setup_thickness,
    }


def test_left_corner_setup_is_mirrored_when_right_is_omitted(
    tmp_path: Path,
    test_data_dir: Path,
) -> None:
    data = yaml.safe_load(
        (test_data_dir / "axle_geometry.yaml").read_text(encoding="utf-8")
    )
    data["axle_config"]["left_setup"] = {
        "camber_shim": _camber_shim_data(
            side=Side.LEFT,
            setup_thickness=35.0,
        )
    }
    path = tmp_path / "mirrored_setup.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    axle = load_geometry(path)

    assert isinstance(axle, AxleSuspension)
    left_config = axle.corners[Side.LEFT].config
    right_config = axle.corners[Side.RIGHT].config
    assert left_config is not None
    assert right_config is not None
    left_shim = left_config.camber_shim
    right_shim = right_config.camber_shim
    assert left_shim is not None
    assert right_shim is not None
    assert left_shim.shim_face_point_a[Axis.Y] == pytest.approx(750.0)
    assert right_shim.shim_face_point_a[Axis.Y] == pytest.approx(-750.0)
    assert right_shim.shim_face_normal[Axis.Y] == pytest.approx(-1.0)
    assert right_shim.setup_thickness == pytest.approx(35.0)


def test_explicit_right_corner_keeps_its_own_setup(
    tmp_path: Path,
    test_data_dir: Path,
) -> None:
    data = yaml.safe_load(
        (test_data_dir / "axle_geometry_explicit.yaml").read_text(encoding="utf-8")
    )
    data["axle_config"]["left_setup"] = {
        "camber_shim": _camber_shim_data(
            side=Side.LEFT,
            setup_thickness=35.0,
        )
    }
    data["axle_config"]["right_setup"] = {
        "camber_shim": _camber_shim_data(
            side=Side.RIGHT,
            setup_thickness=37.0,
        )
    }
    path = tmp_path / "explicit_setup.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    axle = load_geometry(path)

    assert isinstance(axle, AxleSuspension)
    left_config = axle.corners[Side.LEFT].config
    right_config = axle.corners[Side.RIGHT].config
    assert left_config is not None
    assert right_config is not None
    left_shim = left_config.camber_shim
    right_shim = right_config.camber_shim
    assert left_shim is not None
    assert right_shim is not None
    assert left_shim.setup_thickness == pytest.approx(35.0)
    assert right_shim.setup_thickness == pytest.approx(37.0)
    assert right_shim.shim_face_point_a[Axis.Y] == pytest.approx(-750.0)


def test_basic_axle_sweep_solves_and_emits_structural_metrics(
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_sweep.yaml", axle)

    states, stats = solve_sweep(axle, sweep)
    metrics = compute_sweep_metrics(axle, sweep, states)

    assert len(states) == 5
    assert all(info.converged for info in stats)
    assert all(info.max_residual < 1e-3 for info in stats)
    assert metrics.derivative_error is None
    midpoint = metrics.rows[2]
    assert isinstance(midpoint, AxleMetricRows)
    assert "camber" in midpoint.corners["left"]
    assert "camber" in midpoint.corners["right"]
    assert "camber_left" not in midpoint.corners["left"]
    assert "rack_displacement" in midpoint.axle
    assert midpoint.axle["heave"] == pytest.approx(0.0, abs=1e-5)

    final = states[-1]
    left_z = final.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))[Axis.Z]
    right_z = final.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))[Axis.Z]
    assert left_z == pytest.approx(right_z, abs=1e-5)


def test_axle_targets_require_side(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)

    with pytest.raises(ValueError, match="requires side left or right"):
        axle.resolve_target_key(PointID.WHEEL_CENTER, None)


def test_generic_metric_helpers_preserve_structural_axle_rows(
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    state = axle.initial_state()
    assert axle.config is not None

    state_metrics = compute_metrics_for_state_from_suspension(state, axle)
    sweep_metrics = compute_metrics_for_sweep([state], axle, axle.config)

    assert isinstance(state_metrics, AxleMetricRows)
    assert isinstance(sweep_metrics[0], AxleMetricRows)
    assert state_metrics.corners.keys() == {"left", "right"}
    assert "track" in state_metrics.axle
    assert "camber" in state_metrics.corners["left"]
