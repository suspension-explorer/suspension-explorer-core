"""Integration tests for basic full-axle composition."""

from math import hypot
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
from kinematics.core.metrics import main as metrics_main
from kinematics.core.points.derived import ground as ground_module
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.road import RoadPlane
from kinematics.core.solver import solve_suspension_sweep
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import (
    ActuationDirect,
    DoubleWishboneSuspension,
)
from kinematics.core.sweep import (
    compute_sweep_metrics,
    evaluate_solved_sweep,
    solve_sweep,
)


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
    assert "camber" in midpoint.corners[Side.LEFT]
    assert "camber" in midpoint.corners[Side.RIGHT]
    assert "camber_left" not in midpoint.corners[Side.LEFT]
    assert "rack_displacement" in midpoint.axle
    for key in (
        "steering_axis_offset_ground",
        "scrub_radius",
        "mechanical_trail",
    ):
        assert midpoint.corners[Side.LEFT][key] == pytest.approx(
            midpoint.corners[Side.RIGHT][key]
        )
    left_offset = midpoint.corners[Side.LEFT]["steering_axis_offset_ground"]
    left_scrub = midpoint.corners[Side.LEFT]["scrub_radius"]
    left_trail = midpoint.corners[Side.LEFT]["mechanical_trail"]
    assert left_offset is not None
    assert left_scrub is not None
    assert left_trail is not None
    assert left_scrub == pytest.approx((left_offset**2 + left_trail**2) ** 0.5)
    assert midpoint.axle["heave"] == pytest.approx(0.0, abs=1e-5)
    assert midpoint.axle["ride_height_change"] == pytest.approx(0.0, abs=1e-5)

    final = states[-1]
    left_z = final.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))[Axis.Z]
    right_z = final.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))[Axis.Z]
    assert left_z == pytest.approx(right_z, abs=1e-5)


def test_ground_closure_threads_one_seeded_search_per_solved_state(
    test_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_sweep.yaml", axle)
    # Build and cache the design state before recording, so the closure run at
    # initial-state construction is not counted against the sweep.
    axle.initial_state()
    original_search = ground_module._search_ground_normal_angle
    searches: list[tuple[float | None, float]] = []

    def record_search(*args, seed=None):
        root = original_search(*args, seed=seed)
        searches.append((seed, root))
        return root

    monkeypatch.setattr(
        ground_module,
        "_search_ground_normal_angle",
        record_search,
    )

    states, _ = solve_sweep(axle, sweep)
    solve_searches = list(searches)

    assert len(solve_searches) == len(states), (
        "The closure solves the shared plane exactly once per accepted state"
    )
    assert solve_searches[0][0] is not None, (
        "The first state seeds from the stored design tangents, not from nothing"
    )
    for (seed, _), (_, previous_root) in zip(
        solve_searches[1:], solve_searches[:-1], strict=True
    ):
        assert seed == pytest.approx(previous_root), (
            "Each state must be seeded from the previously accepted root"
        )

    solved_roots = [root for _, root in solve_searches]
    metrics = compute_sweep_metrics(axle, sweep, states)
    metric_searches = searches[len(solve_searches) :]

    assert metrics.derivative_error is None
    # Metrics may re-solve the plane for their dual evaluations, but every such
    # solve must be handed a seed that is already on the solved branch.
    assert all(seed is not None for seed, _ in metric_searches), (
        "Metric-time searches must reuse a solved root as their seed"
    )
    deviations = [
        min(abs(seed - root) for root in solved_roots)
        for seed, _ in metric_searches
        if seed is not None
    ]
    assert max(deviations, default=0.0) < 1e-6


def test_ground_closure_is_applied_at_every_public_state_boundary(
    test_data_dir: Path,
) -> None:
    """No public path may hand out axle states with stale closure outputs.

    The low-level solver requires an explicit finaliser; a deliberate no-op
    yields kinematic intermediates whose tangents still sit at design values.
    solve_sweep() finalises at the solver's accept boundary, and
    evaluate_solved_sweep() finalises copies of externally supplied states
    without mutating the caller's, so both public boundaries must agree with
    a direct closure of the raw states.
    """
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_sweep.yaml", axle)
    tangent_refs = (
        PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT),
        PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT),
    )

    raw_states, raw_stats = solve_suspension_sweep(
        initial_state=axle.initial_state(),
        constraints=axle.constraints(),
        sweep_config=sweep,
        derived_manager=DerivedPointsManager(axle.derived_spec()),
        finalize_state=lambda positions: None,
    )
    # The no-op finaliser documents the hazard: tangents still sit at their
    # design positions while the wheel has moved through the sweep.
    stale_left = raw_states[0].get(tangent_refs[0]).copy()
    design_left = axle.initial_state().get(tangent_refs[0])
    assert float((stale_left - design_left).norm()) < 1e-9

    solved_states, _ = solve_sweep(axle, sweep)
    step = 0  # -30 mm bump: the stale error is macroscopic here.
    closed_left = solved_states[step].get(tangent_refs[0])
    assert float((closed_left - stale_left).norm()) > 10.0

    # evaluate_solved_sweep() must finalise COPIES of the raw states to the
    # same closure, leaving the caller's states untouched.
    evaluated = evaluate_solved_sweep(axle, sweep, raw_states, raw_stats)
    for state, reference in zip(evaluated.states, solved_states, strict=True):
        for ref in tangent_refs:
            difference = state.get(ref) - reference.get(ref)
            assert float(difference.norm()) < 1e-9
    untouched_left = raw_states[0].get(tangent_refs[0])
    assert float((untouched_left - stale_left).norm()) < 1e-12, (
        "evaluate_solved_sweep must not mutate the states it was handed"
    )


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
    assert state_metrics.corners.keys() == {Side.LEFT, Side.RIGHT}
    assert "track" in state_metrics.axle
    assert "ride_height_change" in state_metrics.axle
    assert "camber" in state_metrics.corners[Side.LEFT]


def test_axle_metrics_do_not_depend_on_world_placement(
    monkeypatch: pytest.MonkeyPatch,
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)

    def fail_world_placement(*_args, **_kwargs):
        raise AssertionError("metric calculation must not construct WorldSpace")

    # ``raising=False`` keeps this assertion valid after the production import
    # has been removed: an accidentally reintroduced module-level call will
    # still fail, while the desired road-datum path ignores this sentinel.
    monkeypatch.setattr(
        metrics_main,
        "world_space_for_axle_state",
        fail_world_placement,
        raising=False,
    )

    rows = compute_metrics_for_state_from_suspension(axle.initial_state(), axle)

    assert isinstance(rows, AxleMetricRows)
    assert rows.axle["ride_height_change"] == pytest.approx(0.0)


def test_axle_metrics_reject_degenerate_road_contacts(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    state = axle.initial_state().copy()
    left_tangent = PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)
    right_tangent = PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT)
    state.set(left_tangent, state.get(right_tangent))

    with pytest.raises(ValueError):
        compute_metrics_for_state_from_suspension(state, axle)


def test_axle_metrics_share_one_ground_line_instance_with_both_corners(
    monkeypatch: pytest.MonkeyPatch,
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    assert axle.config is not None
    received_ground: list[RoadPlane | None] = []
    original_compute = metrics_main.compute_metrics_for_state

    def capture_ground(*args, **kwargs):
        received_ground.append(kwargs["road"])
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(metrics_main, "compute_metrics_for_state", capture_ground)
    states, _ = solve_sweep(
        axle,
        load_sweep(test_data_dir / "axle_sweep.yaml", axle),
    )
    state = states[-1]
    metrics_main.compute_metrics_for_axle_state(state, axle, axle.config)

    assert len(received_ground) == 2
    assert received_ground[0] is not None
    assert received_ground[0] is received_ground[1]
    road = received_ground[0]
    assert road.normal[Axis.X] == pytest.approx(0.0, abs=1e-12)
    for side in (Side.LEFT, Side.RIGHT):
        tangent = state.get(PointRef(side, PointID.WHEEL_GROUND_TANGENT))
        assert road.signed_distance(tangent) == pytest.approx(0.0, abs=1e-8)


def _road_datum_from_axle_tangents(state: SuspensionState) -> RoadPlane:
    """Build the zero-grade YZ road plane implied by the stored contacts."""
    left = state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT))
    right = state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT))
    dy = float(left[Axis.Y] - right[Axis.Y])
    dz = float(left[Axis.Z] - right[Axis.Z])
    magnitude = hypot(dy, dz)
    normal_y = -dz / magnitude
    normal_z = dy / magnitude
    if normal_z < 0.0:
        normal_y = -normal_y
        normal_z = -normal_z
    return RoadPlane.through(Direction3((0.0, normal_y, normal_z)), left)


def test_ride_height_change_uses_axle_local_road_plane(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    states, _ = solve_sweep(
        axle,
        load_sweep(test_data_dir / "axle_sweep.yaml", axle),
    )
    state = states[-1]
    rows = compute_metrics_for_state_from_suspension(state, axle)
    assert isinstance(rows, AxleMetricRows)
    current_ground = _road_datum_from_axle_tangents(state)
    design_road = axle.design_road_plane
    assert design_road is not None
    chassis_origin = Point3((0.0, 0.0, 0.0))
    expected = current_ground.signed_distance(
        chassis_origin
    ) - design_road.signed_distance(chassis_origin)
    assert rows.axle["ride_height_change"] == pytest.approx(expected)


def test_axle_reuses_one_design_road_plane(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)

    assert axle.design_road_plane is not None
    assert axle.design_road_plane is axle.design_road_plane
