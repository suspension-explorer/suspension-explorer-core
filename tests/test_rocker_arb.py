"""Integration tests for explicit rocker and anti-roll-bar variants."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.constraints import ScalarTripleProductConstraint
from kinematics.core.diagnostics import diagnose_sweep
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import DoubleWishbonePushrodRockerAxleSuspension
from kinematics.core.suspensions.corner import DoubleWishbonePushrodRockerSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep


def _definition_names(suspension) -> set[str]:
    return {
        definition.column_name
        for definition in suspension.derivative_metric_definitions()
    }


def test_corner_spring_type_owns_derivative_declarations(
    test_data_dir: Path,
) -> None:
    torsion = load_geometry(test_data_dir / "corner_rocker_geometry.yaml")
    coilover = load_geometry(test_data_dir / "corner_strut_rocker_geometry.yaml")
    assert isinstance(torsion, DoubleWishbonePushrodRockerSuspension)
    assert isinstance(coilover, DoubleWishbonePushrodRockerSuspension)

    assert _definition_names(torsion) == {
        "deriv_rocker_angle_wrt_hub_z",
        "deriv_torsion_bar_twist_wrt_hub_z",
    }
    assert _definition_names(coilover) == {
        "deriv_rocker_angle_wrt_hub_z",
        "deriv_damper_length_wrt_hub_z",
    }
    assert (
        sum(
            isinstance(constraint, ScalarTripleProductConstraint)
            for constraint in coilover.constraints()
        )
        == 1
    )


def test_corner_derivative_rejects_coincident_rocker_axis(
    test_data_dir: Path,
) -> None:
    corner = load_geometry(test_data_dir / "corner_rocker_geometry.yaml")
    assert isinstance(corner, DoubleWishbonePushrodRockerSuspension)
    corner.hardpoints[PointID.ROCKER_AXIS_REAR] = corner.hardpoints[
        PointID.ROCKER_AXIS_FRONT
    ]

    with pytest.raises(ValueError, match="distinct rocker axis points"):
        corner.derivative_metric_definitions()


def test_axle_derivative_rejects_coincident_arb_axis(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)
    axle.center_points[PointID.ARB_AXIS_B] = axle.center_points[PointID.ARB_AXIS_A]

    with pytest.raises(ValueError, match="distinct ARB axis points"):
        axle.derivative_metric_definitions()


def test_arb_axle_uses_chirality_constraints(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)

    chirality = [
        constraint
        for constraint in axle.constraints()
        if isinstance(constraint, ScalarTripleProductConstraint)
    ]

    assert len(chirality) == 2


def test_arb_axle_emits_hub_relative_derivatives(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_rocker_sweep.yaml", axle)
    states, _ = solve_sweep(axle, sweep)

    result = compute_sweep_metrics(axle, sweep, states)

    assert result.derivative_error is None
    expected_corner = {
        "deriv_rocker_angle_wrt_hub_z",
        "deriv_torsion_bar_twist_wrt_hub_z",
    }
    expected_axle = {
        "deriv_arb_twist_wrt_hub_z_left",
        "deriv_arb_twist_wrt_hub_z_right",
    }
    for row in result.rows:
        assert isinstance(row, AxleMetricRows)
        assert expected_axle <= row.axle.keys()
        assert all(row.axle[key] is not None for key in expected_axle)
        assert "arb_twist" in row.axle
        for location in ("left", "right"):
            corner = row.corners[location]
            assert expected_corner <= corner.keys()
            assert all(corner[key] is not None for key in expected_corner)
            assert "rocker_angle" in corner
            assert "arb_arm_angle" in corner


def test_arb_diagnostics_detect_mirrored_arm_branch(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_rocker_sweep.yaml", axle)
    states, stats = solve_sweep(axle, sweep)
    step = len(states) // 2
    state = states[step].copy()
    side = Side.LEFT
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A)).data
    axis_b = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)).data
    rocker = state.get(PointRef(side, PointID.DROPLINK_ROCKER)).data
    arm_key = PointRef(side, PointID.DROPLINK_ARB)
    arm = state.get(arm_key).data
    normal = np.cross(axis_b - axis_a, rocker - axis_a)
    normal /= np.linalg.norm(normal)
    state.positions[arm_key] = Point3(
        arm - 2.0 * float(np.dot(arm - axis_a, normal)) * normal
    )
    states[step] = state

    diagnostics = diagnose_sweep(axle, states, stats)

    issues = [issue for issue in diagnostics.errors if issue.category == "chirality"]
    assert len(issues) == 1
    assert issues[0].step == step


def test_arb_diagnostics_detect_chirality_boundary(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_rocker_sweep.yaml", axle)
    states, stats = solve_sweep(axle, sweep)
    step = len(states) // 2
    state = states[step].copy()
    side = Side.LEFT
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A)).data
    axis_b = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)).data
    rocker = state.get(PointRef(side, PointID.DROPLINK_ROCKER)).data
    arm_key = PointRef(side, PointID.DROPLINK_ARB)
    arm = state.get(arm_key).data
    normal = np.cross(axis_b - axis_a, rocker - axis_a)
    normal /= np.linalg.norm(normal)
    state.positions[arm_key] = Point3(
        arm - float(np.dot(arm - axis_a, normal)) * normal
    )
    states[step] = state

    diagnostics = diagnose_sweep(axle, states, stats)

    issues = [issue for issue in diagnostics.errors if issue.category == "chirality"]
    assert len(issues) == 1
    assert issues[0].step == step
    assert "boundary" in issues[0].message
