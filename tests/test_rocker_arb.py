"""Integration tests for explicit rocker and anti-roll-bar variants."""

from pathlib import Path

import numpy as np

from kinematics.constraints import ScalarTripleProductConstraint
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointRef, Side
from kinematics.diagnostics import diagnose_sweep
from kinematics.io import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.suspensions.axle import DoubleWishbonePushrodRockerAxleSuspension
from kinematics.suspensions.corner import DoubleWishbonePushrodRockerSuspension


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
    sweep = parse_sweep_file(test_data_dir / "axle_rocker_sweep.yaml", axle)
    states, _ = solve_sweep(axle, sweep)

    result = compute_sweep_metrics(axle, sweep, states)

    assert result.derivative_error is None
    expected = {
        "deriv_rocker_angle_wrt_hub_z_left",
        "deriv_torsion_bar_twist_wrt_hub_z_left",
        "deriv_rocker_angle_wrt_hub_z_right",
        "deriv_torsion_bar_twist_wrt_hub_z_right",
        "deriv_arb_twist_wrt_hub_z_left",
        "deriv_arb_twist_wrt_hub_z_right",
    }
    for row in result.rows:
        assert expected <= row.keys()
        assert all(row[column] is not None for column in expected)
        assert "rocker_angle_deg_left" in row
        assert "rocker_angle_deg_right" in row
        assert "arb_arm_angle_deg_left" in row
        assert "arb_arm_angle_deg_right" in row
        assert "arb_twist_deg" in row


def test_arb_diagnostics_detect_mirrored_arm_branch(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(axle, DoubleWishbonePushrodRockerAxleSuspension)
    sweep = parse_sweep_file(test_data_dir / "axle_rocker_sweep.yaml", axle)
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
