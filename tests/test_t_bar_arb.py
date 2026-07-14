"""Tests for the composed rigid T-bar anti-roll mechanism."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.elements import TBarElement
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.presentation import PointMidpoint, element_paths
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import (
    ArbTBar,
    AxleSuspension,
    HeaveLinkNone,
)
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep


@pytest.fixture
def t_bar_axle(test_data_dir: Path) -> AxleSuspension:
    suspension = load_geometry(test_data_dir / "axle_geometry_t_bar.yaml")
    assert isinstance(suspension, AxleSuspension)
    assert isinstance(suspension.anti_roll, ArbTBar)
    assert isinstance(suspension.heave_link, HeaveLinkNone)
    return suspension


def test_t_bar_assembly_has_only_pivot_and_crossbar_endpoints(
    t_bar_axle: AxleSuspension,
) -> None:
    """The crossbar midpoint is presentation geometry, not a solver point."""
    assembly = t_bar_axle.assembly()
    pivot = PointRef(Side.CENTER, PointID.ARB_T_BAR_PIVOT)
    left = PointRef(Side.LEFT, PointID.DROPLINK_T_BAR)
    right = PointRef(Side.RIGHT, PointID.DROPLINK_T_BAR)
    center = PointMidpoint(left, right)

    assert pivot in assembly.points.fixed
    assert assembly.points.derived.isdisjoint({pivot, left, right})
    for side in (Side.LEFT, Side.RIGHT):
        assert PointRef(side, PointID.DROPLINK_T_BAR) in assembly.points.free

    t_bar = next(
        element
        for element in assembly.elements
        if isinstance(element, TBarElement) and element.label == "T-Bar Anti-Roll Bar"
    )
    assert (t_bar.pivot, t_bar.left_attachment, t_bar.right_attachment) == (
        pivot,
        left,
        right,
    )
    assert tuple(
        path.points for path in element_paths(assembly) if path.label == t_bar.label
    ) == (
        (pivot, center),
        (left, center, right),
    )
    assert t_bar_axle.heave_link.elements() == ()


def test_demo_rocker_axis_is_oblique_and_nearly_perpendicular_to_pushrod(
    t_bar_axle: AxleSuspension,
) -> None:
    """The demonstration uses the intended oblique rocker-axis arrangement."""
    corner = t_bar_axle.corners[Side.LEFT]
    axis = (
        corner.hardpoints[PointID.ROCKER_AXIS_B]
        - corner.hardpoints[PointID.ROCKER_AXIS_A]
    ).data
    pushrod = (
        corner.hardpoints[PointID.PUSHROD_OUTBOARD]
        - corner.hardpoints[PointID.PUSHROD_INBOARD]
    ).data

    assert abs(float(axis[0])) < 1e-12
    assert abs(float(axis[1])) > 1.0
    assert abs(float(axis[2])) > 1.0
    normalized_dot = abs(float(np.dot(axis, pushrod))) / (
        float(np.linalg.norm(axis)) * float(np.linalg.norm(pushrod))
    )
    assert normalized_dot < 0.3


def test_bump_sweep_preserves_rigid_t_bar_distances(
    t_bar_axle: AxleSuspension,
    test_data_dir: Path,
) -> None:
    """Same-direction wheel travel moves the T-bar stem through its XZ arc."""
    sweep = load_sweep(test_data_dir / "axle_t_bar_bump_sweep.yaml", t_bar_axle)
    states, stats = solve_sweep(t_bar_axle, sweep)
    assert all(info.converged for info in stats)

    pivot_key = PointRef(Side.CENTER, PointID.ARB_T_BAR_PIVOT)
    wheel_key = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
    left_arm = PointRef(Side.LEFT, PointID.DROPLINK_T_BAR)
    right_arm = PointRef(Side.RIGHT, PointID.DROPLINK_T_BAR)
    design = t_bar_axle.initial_state()
    design_lengths = {
        (left_arm, right_arm): (design.get(left_arm) - design.get(right_arm)).norm(),
        (left_arm, pivot_key): (design.get(left_arm) - design.get(pivot_key)).norm(),
        (right_arm, pivot_key): (design.get(right_arm) - design.get(pivot_key)).norm(),
    }

    center_x: list[float] = []
    wheel_travel: list[float] = []
    for state in states:
        center = (
            state.get(left_arm) + (state.get(right_arm) - state.get(left_arm)) / 2.0
        )
        assert float(center[1]) == pytest.approx(0.0, abs=1e-7)
        for (point_a, point_b), design_length in design_lengths.items():
            current_length = (state.get(point_a) - state.get(point_b)).norm()
            assert current_length == pytest.approx(design_length, abs=1e-5)
        center_x.append(float(center[0]))
        wheel_travel.append(float(state.get(wheel_key)[2] - design.get(wheel_key)[2]))

    assert min(wheel_travel) == pytest.approx(-50.0, abs=1e-5)
    assert max(wheel_travel) == pytest.approx(50.0, abs=1e-5)
    assert max(center_x) - min(center_x) > 3.0


def test_roll_sweep_produces_differential_t_bar_twist(
    t_bar_axle: AxleSuspension,
    test_data_dir: Path,
) -> None:
    """Opposed wheel travel rotates the rigid crossbar about the T-bar stem."""
    sweep = load_sweep(test_data_dir / "axle_t_bar_roll_sweep.yaml", t_bar_axle)
    states, stats = solve_sweep(t_bar_axle, sweep)
    assert all(info.converged for info in stats)

    design = t_bar_axle.initial_state()
    left_wheel = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
    right_wheel = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
    left_travel = [
        float(state.get(left_wheel)[2] - design.get(left_wheel)[2]) for state in states
    ]
    right_travel = [
        float(state.get(right_wheel)[2] - design.get(right_wheel)[2])
        for state in states
    ]
    assert min(left_travel) == pytest.approx(-50.0, abs=1e-5)
    assert max(left_travel) == pytest.approx(50.0, abs=1e-5)
    assert min(right_travel) == pytest.approx(-50.0, abs=1e-5)
    assert max(right_travel) == pytest.approx(50.0, abs=1e-5)

    result = compute_sweep_metrics(t_bar_axle, sweep, states)
    assert result.derivative_error is None
    expected_derivatives = {
        "deriv_t_bar_center_x_wrt_hub_z_left",
        "deriv_t_bar_center_x_wrt_hub_z_right",
        "deriv_arb_twist_wrt_hub_z_left",
        "deriv_arb_twist_wrt_hub_z_right",
    }
    twists = []
    for row in result.rows:
        assert isinstance(row, AxleMetricRows)
        assert expected_derivatives <= row.axle.keys()
        assert all(row.axle[key] is not None for key in expected_derivatives)
        assert "t_bar_twist" not in row.axle
        twist = row.axle["arb_twist"]
        assert twist is not None
        twists.append(float(twist))

    assert max(twists) - min(twists) > 1.0
