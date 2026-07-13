"""MacPherson strut corner: solving, physical invariants, and metrics."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.geometry import Point3
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import MacPhersonSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def macpherson() -> MacPhersonSuspension:
    suspension = load_geometry(TEST_DATA / "macpherson_geometry.yaml")
    assert isinstance(suspension, MacPhersonSuspension)
    return suspension


def _distance(state: SuspensionState, point_a: PointID, point_b: PointID) -> float:
    return float((state.get(point_a) - state.get(point_b)).norm())


def test_macpherson_declares_expected_point_roles(macpherson):
    assert macpherson.reported_type_key() == "macpherson"
    assert macpherson.steering_axis_points() == (
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.STRUT_TOP,
    )
    assert macpherson.rack_attachment_point() is PointID.TRACKROD_INBOARD
    assert macpherson.wheel_axis_points() == (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    assert macpherson.damper_points() == (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    assert PointID.STRUT_TOP not in macpherson.free_points()
    assert PointID.STRUT_BOTTOM not in macpherson.free_points()
    assert PointID.STRUT_BOTTOM in macpherson.derived_spec().functions


def test_static_state_reproduces_input_geometry(macpherson):
    state = macpherson.initial_state()
    # The authored clamp is projected onto the steering axis, so allow float
    # noise from that projection.
    for point, authored in macpherson.hardpoints.items():
        assert (state.get(point) - authored).norm() == pytest.approx(0.0, abs=1e-6)

    # The strut clamp lies on the design strut axis between the ball joint
    # and the top mount.
    ball_joint = state.get(PointID.LOWER_WISHBONE_OUTBOARD)
    strut_top = state.get(PointID.STRUT_TOP)
    clamp = state.get(PointID.STRUT_BOTTOM)
    axis = (strut_top - ball_joint).normalize()
    off_axis = (clamp - ball_joint).data - (
        float((clamp - ball_joint).data @ axis.data) * axis.data
    )
    assert float((off_axis**2).sum()) == pytest.approx(0.0, abs=1e-12)


def test_sweep_solves_with_physical_invariants(macpherson):
    sweep = load_sweep(TEST_DATA / "sweep.yaml")
    states, infos = solve_sweep(macpherson, sweep)
    assert all(info.converged for info in infos)
    assert all(info.max_residual < 1e-3 for info in infos)

    design = macpherson.initial_state()
    rigid_pairs = [
        (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
        (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
        (PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD),
        (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
        (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
        (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
        (PointID.STRUT_BOTTOM, PointID.LOWER_WISHBONE_OUTBOARD),
        (PointID.STRUT_BOTTOM, PointID.AXLE_INBOARD),
    ]
    for state in states:
        # Fixed chassis mount.
        assert (
            state.get(PointID.STRUT_TOP) - design.get(PointID.STRUT_TOP)
        ).norm() == pytest.approx(0.0)
        # Rigid links and upright keep their authored lengths.
        for point_a, point_b in rigid_pairs:
            assert _distance(state, point_a, point_b) == pytest.approx(
                _distance(design, point_a, point_b), abs=1e-3
            )
        # The fixed top mount stays on the moving strut axis.
        ball_joint = state.get(PointID.LOWER_WISHBONE_OUTBOARD)
        axis = (state.get(PointID.STRUT_BOTTOM) - ball_joint).normalize()
        to_top = state.get(PointID.STRUT_TOP) - ball_joint
        off_axis = to_top.data - float(to_top.data @ axis.data) * axis.data
        assert float((off_axis**2).sum()) ** 0.5 == pytest.approx(0.0, abs=1e-2)

    # The strut telescopes: full bump compresses it substantially.
    design_length = _distance(design, PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    bump_length = _distance(states[-1], PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    droop_length = _distance(states[0], PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    assert bump_length < design_length - 50.0
    assert droop_length > design_length + 10.0


def test_sweep_metrics_and_derivatives(macpherson):
    sweep = load_sweep(TEST_DATA / "sweep.yaml")
    states, _ = solve_sweep(macpherson, sweep)
    metrics = compute_sweep_metrics(macpherson, sweep, states)

    assert metrics.derivative_error is None
    assert metrics.tangent_solve_infos is not None
    assert all(not info.rank_deficient for info in metrics.tangent_solve_infos)
    droop_row = metrics.rows[0]
    bump_row = metrics.rows[-1]
    # A corner sweep yields plain metric rows, never axle row bundles.
    assert not isinstance(droop_row, AxleMetricRows)
    assert not isinstance(bump_row, AxleMetricRows)
    for key in ("camber", "caster", "kpi", "damper_length", "wheel_travel"):
        assert droop_row[key] is not None
        assert bump_row[key] is not None

    # Steered corner: the bump-steer derivative is declared and evaluated.
    assert bump_row["deriv_roadwheel_angle_wrt_rack_displacement"] is not None
    assert bump_row["deriv_damper_length_wrt_hub_z"] is not None

    # The damper length metric reads the strut through damper_points().
    assert bump_row["damper_length"] == pytest.approx(
        _distance(states[-1], PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    )

    # Camber changes over travel: the strut geometry is actually articulating.
    droop_camber = droop_row["camber"]
    bump_camber = bump_row["camber"]
    assert droop_camber is not None and bump_camber is not None
    assert abs(bump_camber - droop_camber) > 0.5


def test_strut_top_coincident_with_ball_joint_rejected(macpherson):
    hardpoints = macpherson.get_hardpoints_copy()
    hardpoints[PointID.STRUT_TOP] = Point3(
        hardpoints[PointID.LOWER_WISHBONE_OUTBOARD].data.copy()
    )
    with pytest.raises(ValueError, match="steering axis would be undefined"):
        MacPhersonSuspension(
            name="degenerate",
            side=macpherson.side,
            hardpoints=hardpoints,
            config=macpherson.config,
        )


def test_strut_clamp_off_the_steering_axis_rejected(macpherson):
    hardpoints = macpherson.get_hardpoints_copy()
    off_axis = hardpoints[PointID.STRUT_BOTTOM].data.copy()
    off_axis[0] += 25.0
    hardpoints[PointID.STRUT_BOTTOM] = Point3(off_axis)
    with pytest.raises(ValueError, match="off the line from"):
        MacPhersonSuspension(
            name="offset_clamp",
            side=macpherson.side,
            hardpoints=hardpoints,
            config=macpherson.config,
        )


def test_macpherson_axle_composes_through_generic_axle():
    axle = load_geometry(TEST_DATA / "macpherson_axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    assert axle.reported_type_key() == "macpherson"
    assert all(
        isinstance(corner, MacPhersonSuspension) for corner in axle.corners.values()
    )

    sweep = load_sweep(TEST_DATA / "axle_sweep.yaml", axle)
    states, infos = solve_sweep(axle, sweep)
    assert all(info.converged for info in infos)
    assert all(info.max_residual < 1e-3 for info in infos)

    metrics = compute_sweep_metrics(axle, sweep, states)
    assert metrics.derivative_error is None
    midpoint = metrics.rows[len(metrics.rows) // 2]
    assert isinstance(midpoint, AxleMetricRows)
    assert midpoint.corners["left"]["camber"] is not None
    assert midpoint.corners["right"]["kpi"] is not None
    # Both corners are steered, so the axle carries the rigid rack coupling
    # and reports rack displacement.
    assert axle.rack_attachment_points() == (
        PointID.TRACKROD_INBOARD,
        PointID.TRACKROD_INBOARD,
    )
    assert midpoint.axle["rack_displacement"] is not None
