"""Integration tests for analytical steering-response axes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.analysis import analyze_evaluated_sweep
from kinematics.core.enums import PointID
from kinematics.core.holds import CoordinateHold
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.rigid_motion import ScrewAxisStatus
from kinematics.core.sensitivity import compute_state_tangents
from kinematics.core.steering_axis import compute_steering_response_tangent
from kinematics.core.steering_response import (
    SteeringResponseDefinition,
    materialize_steering_response_targets,
)
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.sweep import solve_evaluated_sweep

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "geometry_file",
    ["corner_rocker_damper_geometry.yaml", "macpherson_geometry.yaml"],
    ids=("double_wishbone_rocker", "macpherson"),
)
def test_corner_bump_sweep_has_valid_steering_response_axis_at_every_step(
    geometry_file: str,
) -> None:
    suspension = load_geometry(DATA_DIR / geometry_file)
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert len(evaluated.steering_response_axes) == len(evaluated.states)
    for frame_results in evaluated.steering_response_axes:
        assert len(frame_results) == 1
        assert frame_results[0].status is ScrewAxisStatus.VALID
        assert frame_results[0].axis is not None

    midpoint = len(evaluated.states) // 2
    state = evaluated.states[midpoint]
    axis = evaluated.steering_response_axes[midpoint][0].axis
    assert axis is not None
    assert isinstance(suspension, CornerSuspension)
    axis_points = suspension.steering_axis_points()
    assert axis_points is not None
    lower_key, upper_key = axis_points
    lower = state.get(lower_key).data
    physical_direction = state.get(upper_key).data - lower
    physical_direction /= np.linalg.norm(physical_direction)
    assert abs(float(np.dot(axis.direction.data, physical_direction))) > 0.999


def test_shared_rack_returns_one_axis_per_upright_with_expected_symmetry() -> None:
    suspension = load_geometry(DATA_DIR / "axle_geometry_rocker_damper.yaml")
    sweep = load_sweep(DATA_DIR / "axle_steer_sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    for frame_results in evaluated.steering_response_axes:
        assert [result.upright_label for result in frame_results] == [
            "Left Upright",
            "Right Upright",
        ]
        assert all(result.status is ScrewAxisStatus.VALID for result in frame_results)
    left, right = evaluated.steering_response_axes[len(evaluated.states) // 2]
    assert left.axis is not None
    assert right.axis is not None
    assert left.axis.point.data == pytest.approx(
        right.axis.point.data * np.array([1.0, -1.0, 1.0]),
        abs=1e-4,
    )
    assert abs(left.axis.pitch) == pytest.approx(abs(right.axis.pitch), rel=1e-6)


def test_macpherson_response_recovers_balljoint_to_strut_top_line() -> None:
    suspension = load_geometry(DATA_DIR / "macpherson_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep)

    for state, frame_results in zip(
        evaluated.states,
        evaluated.steering_response_axes,
        strict=True,
    ):
        result = frame_results[0]
        assert result.status is ScrewAxisStatus.VALID
        assert result.axis is not None
        lower = state.get(PointID.LOWER_WISHBONE_OUTBOARD).data
        upper = state.get(PointID.STRUT_TOP).data
        physical_direction = upper - lower
        physical_direction /= np.linalg.norm(physical_direction)
        assert abs(float(np.dot(result.axis.direction.data, physical_direction))) == (
            pytest.approx(1.0, abs=1e-10)
        )
        assert (
            np.linalg.norm(np.cross(result.axis.point.data - lower, physical_direction))
            < 1e-8
        )
        assert result.axis.pitch == pytest.approx(0.0, abs=1e-8)


def test_macpherson_lower_arm_option_is_equivalent_to_strut_length() -> None:
    suspension = load_geometry(DATA_DIR / "macpherson_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)
    strut = solve_evaluated_sweep(suspension, sweep)
    lower_arm = solve_evaluated_sweep(
        suspension,
        replace(sweep, suspension_hold_id="lower_arm_angle"),
    )

    for strut_results, lower_results in zip(
        strut.steering_response_axes,
        lower_arm.steering_response_axes,
        strict=True,
    ):
        strut_axis = strut_results[0].axis
        lower_axis = lower_results[0].axis
        assert strut_axis is not None and lower_axis is not None
        assert strut_axis.point.data == pytest.approx(lower_axis.point.data, abs=1e-8)
        assert abs(
            float(np.dot(strut_axis.direction.data, lower_axis.direction.data))
        ) == pytest.approx(1.0, abs=1e-10)
        assert strut_axis.pitch == pytest.approx(lower_axis.pitch, abs=1e-10)


def test_unsteered_suspension_has_aligned_empty_axis_frames() -> None:
    suspension = load_geometry(DATA_DIR / "trailing_arm_coilover_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "trailing_arm_sweep.yaml", suspension)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert evaluated.steering_response_axes == tuple(() for _ in evaluated.states)


def test_incomplete_suspension_hold_is_reported_as_rank_deficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    steering = suspension.steering_actuator_dof()
    assert steering is not None
    incomplete = SteeringResponseDefinition(
        steering_actuator=steering,
        hold=CoordinateHold(),
        owner="test",
        definition_id="missing_jounce_hold",
    )
    monkeypatch.setattr(
        suspension,
        "resolve_suspension_hold",
        lambda _requested=None: incomplete,
    )

    response = compute_steering_response_tangent(
        suspension,
        suspension.initial_state(),
    )

    assert response.status is ScrewAxisStatus.RANK_DEFICIENT
    assert response.solve_info is not None
    assert response.solve_info.mobility == 2
    assert response.solve_info.target_rank == 1
    assert response.solve_info.nullity == 1
    assert response.message is not None
    assert "missing_jounce_hold" in response.message
    assert "nullity 1" in response.message


def test_redundant_consistent_wishbone_holds_are_accepted() -> None:
    """Two coordinates spanning the same travel mode remain a valid hold."""
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    catalogue = suspension.suspension_hold_catalogue()
    definition = suspension.resolve_suspension_hold()
    assert catalogue is not None
    assert definition is not None
    lower = catalogue.option("lower_wishbone_angle").hold.coordinates[0]
    upper = catalogue.option("upper_wishbone_angle").hold.coordinates[0]
    redundant = replace(
        definition,
        hold=CoordinateHold((lower, upper)),
        definition_id="both_wishbone_angles",
    )
    state = suspension.initial_state()

    response = compute_steering_response_tangent(
        suspension,
        state,
        definition=redundant,
    )

    assert response.status is ScrewAxisStatus.VALID
    assert response.tangent is not None
    assert response.solve_info is not None
    steering_info = response.solve_info.response_for_target(0)
    assert response.solve_info.mobility == 2
    assert response.solve_info.target_rank == 2
    assert response.solve_info.full_column_rank
    assert steering_info.rate_consistent
    assert steering_info.unique
    assert steering_info.max_constraint_rate_residual <= (
        steering_info.consistency_tolerance
    )
    assert steering_info.max_other_target_rate_residual <= (
        steering_info.consistency_tolerance
    )
    for point in (
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_OUTBOARD,
    ):
        assert response.tangent.velocity(point) == pytest.approx(
            np.zeros(3),
            abs=2e-10,
        )


def test_conflicting_real_hold_basis_is_rejected_as_inconsistent() -> None:
    """Fixed wishbone travel cannot also fix a steering-driven damper."""
    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    catalogue = suspension.suspension_hold_catalogue()
    definition = suspension.resolve_suspension_hold()
    assert catalogue is not None
    assert definition is not None
    lower = catalogue.option("lower_wishbone_angle").hold.coordinates[0]
    damper = catalogue.option("damper_length").hold.coordinates[0]
    state = suspension.initial_state()

    canonical = compute_steering_response_tangent(
        suspension,
        state,
        definition=definition,
    )
    assert canonical.status is ScrewAxisStatus.VALID
    assert canonical.tangent is not None
    damper_rate = sum(
        float(partial @ canonical.tangent.velocity(point))
        for point, partial in damper.current_value_target(
            state.positions
        ).point_partials(state.positions)
    )
    assert abs(damper_rate) > 1e-3

    conflicting = replace(
        definition,
        hold=CoordinateHold((lower, damper)),
        definition_id="wishbone_angle_and_steering_driven_damper",
    )
    response = compute_steering_response_tangent(
        suspension,
        state,
        definition=conflicting,
    )

    assert response.status is ScrewAxisStatus.INCONSISTENT_TANGENT
    assert response.tangent is None
    assert response.solve_info is not None
    steering_info = response.solve_info.response_for_target(0)
    assert response.solve_info.mobility == 2
    assert response.solve_info.target_rank == 2
    assert response.solve_info.full_column_rank
    assert not steering_info.rate_consistent
    assert not steering_info.unique
    assert steering_info.max_rate_residual > steering_info.consistency_tolerance
    assert response.message is not None
    assert "rate-inconsistent" in response.message
    assert "wishbone_angle_and_steering_driven_damper" in response.message


def test_full_rank_inconsistent_suspension_hold_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.steering_axis as steering_axis_module

    suspension = load_geometry(DATA_DIR / "macpherson_geometry.yaml")
    state = suspension.initial_state()
    response_targets = materialize_steering_response_targets(
        suspension.resolve_suspension_hold(),
        state,
    )
    assert response_targets is not None
    fields, solve_info = compute_state_tangents(
        state,
        suspension.constraints(),
        DerivedPointsManager(suspension.derived_spec()),
        response_targets.targets,
        post_derived_update=suspension.apply_ground_closure,
    )
    steering_info = solve_info.response_for_target(0)
    bad_steering_info = replace(
        steering_info,
        target_rate_residuals=(0.1, *steering_info.target_rate_residuals[1:]),
    )
    bad_info = replace(
        solve_info,
        responses=(bad_steering_info, *solve_info.responses[1:]),
    )
    monkeypatch.setattr(
        steering_axis_module,
        "compute_state_tangents",
        lambda *_args, **_kwargs: (fields, bad_info),
    )

    response = compute_steering_response_tangent(suspension, state)

    assert response.status is ScrewAxisStatus.INCONSISTENT_TANGENT
    assert response.tangent is None
    assert response.message is not None
    assert "rate-inconsistent" in response.message
    assert "0.1" in response.message


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
    assert all(evaluated.steering_response_axes)


def test_suspension_hold_failure_keeps_explicit_axis_results_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinematics.core.steering_axis as steering_axis_module

    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)

    def fail_tangents(*_args, **_kwargs):
        raise RuntimeError("synthetic tangent failure")

    monkeypatch.setattr(steering_axis_module, "compute_state_tangents", fail_tangents)

    evaluated = solve_evaluated_sweep(suspension, sweep)

    assert all(evaluated.metrics.rows)
    assert all(
        results[0].status is ScrewAxisStatus.TANGENT_UNAVAILABLE
        for results in evaluated.steering_response_axes
    )
    categories = {issue.category for issue in evaluated.diagnostics}
    assert "steering_axis" in categories


def test_structured_analysis_exposes_axis_and_fit_diagnostics() -> None:
    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    sweep = load_sweep(DATA_DIR / "corner_steer_bump_sweep.yaml", suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep)

    analysis = analyze_evaluated_sweep(suspension, sweep, evaluated)

    definition = analysis.steering_response
    assert definition is not None
    assert definition.owner == "double_wishbone"
    assert definition.definition_id == "lower_wishbone_angle"
    assert definition.provenance == "double_wishbone:lower_wishbone_angle"
    assert definition.steering_coordinate_id == "rack"
    assert definition.requested_option_id == "layout_default"
    assert definition.resolved_option_id == "lower_wishbone_angle"
    assert definition.selection_source == "layout_default"
    assert [coordinate.id for coordinate in definition.held_coordinates] == [
        "lower_wishbone_angle"
    ]
    assert [coordinate.side for coordinate in definition.held_coordinates] == ["left"]

    result = analysis.frames[len(analysis.frames) // 2].steering_response_axes[0]
    assert result.status is ScrewAxisStatus.VALID
    assert result.point is not None
    assert result.direction is not None
    assert result.pitch is not None
    assert result.angular_rate is not None
    assert result.fit_rms is not None
    assert result.fit_max is not None
    assert result.fit_rank == 6
    assert result.point_count >= 3
