"""Integration tests for world spaces derived from solved axle states."""

from __future__ import annotations

from math import atan2, isclose
from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis, PointID
from kinematics.core.input import build_sweep
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.pose import (
    GravityModel,
    world_space_for_axle_state,
    world_spaces_for_sweep,
)
from kinematics.core.primitives.geometry import Direction3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.sweep import solve_sweep

TANGENT_REFS = (
    PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT),
    PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT),
)


def _solve_states(
    test_data_dir: Path, *, left_z: float, right_z: float
) -> tuple[AxleSuspension, list[SuspensionState]]:
    """Solve one axle state with independent left/right wheel-centre travel."""
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    spec = yaml.safe_load(
        (test_data_dir / "axle_sweep.yaml").read_text(encoding="utf-8")
    )
    del spec["steps"]
    spec["targets"][0]["values"] = [left_z]
    spec["targets"][1]["values"] = [right_z]
    spec["targets"][2]["values"] = [0.0]
    for target in spec["targets"]:
        target.pop("start", None)
        target.pop("stop", None)
    states, _ = solve_sweep(axle, build_sweep(spec, axle))
    return axle, states


def _state_datum(state: SuspensionState) -> GroundDatum:
    ground = GroundDatum.from_wheel_ground_tangents(
        state.get(TANGENT_REFS[0]), state.get(TANGENT_REFS[1])
    )
    assert ground is not None
    return ground


def test_road_level_puts_world_up_along_the_ground_normal(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    state = states[0]
    ground = _state_datum(state)
    assert abs(ground.angle_deg) > 0.5, "The test state must be genuinely banked"

    space = world_space_for_axle_state(axle, state, GravityModel.ROAD_LEVEL)

    assert space is not None
    assert space.gravity_model is GravityModel.ROAD_LEVEL
    np.testing.assert_allclose(space.z.data, ground.normal.data, atol=1e-12)
    # The road is world-level under this model, so both solved tangent
    # points map onto the world Z = 0 plane.
    for ref in TANGENT_REFS:
        mapped = space.to_world(state.get(ref))
        assert mapped[Axis.Z] == pytest.approx(0.0, abs=1e-9)


def test_chassis_level_reads_the_same_state_as_road_bank(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    state = states[0]

    space = world_space_for_axle_state(axle, state, GravityModel.CHASSIS_LEVEL)

    assert space is not None
    # The rig interpretation: world axes coincide with chassis axes, and the
    # fitted ground angle reads as road bank under a level vehicle.
    np.testing.assert_allclose(space.x.data, (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(space.y.data, (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(space.z.data, (0.0, 0.0, 1.0), atol=1e-12)


def test_opposite_axle_fixed_tips_gravity_by_the_ride_height_pitch(
    test_data_dir,
) -> None:
    axle, states = _solve_states(test_data_dir, left_z=-30.0, right_z=-30.0)
    state = states[0]
    ground = _state_datum(state)
    design_ground = axle.design_ground
    assert design_ground is not None
    current_z = ground.z_at(0.0)
    design_z = design_ground.z_at(0.0)
    assert current_z is not None and design_z is not None
    compression = current_z - design_z
    assert abs(compression) > 5.0, "Bump travel must change the ride height"

    space = world_space_for_axle_state(axle, state, GravityModel.OPPOSITE_AXLE_FIXED)

    assert space is not None
    config = axle.corners[Side.LEFT].config
    assert config is not None
    expected_pitch = atan2(compression, config.wheelbase)
    # Front-axle gravity under pitch theta is (sin(theta), 0, -cos(theta)), so
    # world up carries z_x = -sin(theta) and the extraction below recovers
    # theta with its sign: nose-down positive for front-axle compression.
    actual_pitch = atan2(-float(space.z[Axis.X]), float(space.z[Axis.Z]))
    assert actual_pitch == pytest.approx(expected_pitch, abs=1e-12)
    assert isclose(float(np.linalg.norm(space.z.data)), 1.0)


def test_string_assumptions_match_their_enum_members(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    state = states[0]

    for assumption in GravityModel:
        by_enum = world_space_for_axle_state(axle, state, assumption)
        by_string = world_space_for_axle_state(axle, state, assumption.value)
        assert by_enum is not None and by_string is not None
        np.testing.assert_allclose(
            by_enum.rotation_chassis_to_world,
            by_string.rotation_chassis_to_world,
            atol=0.0,
        )
        assert by_string.gravity_model is assumption

    with pytest.raises(ValueError):
        world_space_for_axle_state(axle, state, "pure_heave")


def test_explicit_gravity_bypasses_every_assumption(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    state = states[0]
    gravity = Direction3((0.05, 0.1, -0.99))

    space = world_space_for_axle_state(axle, state, gravity)

    assert space is not None
    assert space.gravity_model is None
    np.testing.assert_allclose(space.gravity.data, gravity.data, atol=1e-12)


def test_sweep_helper_reports_one_space_per_state(test_data_dir) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    from kinematics.cli.io.sweep_loader import load_sweep

    states, _ = solve_sweep(axle, load_sweep(test_data_dir / "axle_sweep.yaml", axle))

    spaces = world_spaces_for_sweep(axle, states, GravityModel.ROAD_LEVEL)
    assert len(spaces) == len(states)
    assert all(space is not None for space in spaces)

    per_state = [Direction3((0.0, 0.0, -1.0)) for _ in states]
    explicit = world_spaces_for_sweep(axle, states, per_state)
    assert len(explicit) == len(states)
    assert all(space is not None and space.gravity_model is None for space in explicit)

    with pytest.raises(ValueError, match="one direction per state"):
        world_spaces_for_sweep(axle, states, per_state[:-1])
