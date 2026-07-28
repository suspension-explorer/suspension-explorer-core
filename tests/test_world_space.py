"""Integration tests for the axle-local world presentation transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis, PointID
from kinematics.core.input import build_sweep
from kinematics.core.pose import world_space_for_axle_state, world_spaces_for_sweep
from kinematics.core.primitives.geometry import Point3
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


def _assert_supported_placement(
    axle: AxleSuspension,
    state: SuspensionState,
) -> None:
    space = world_space_for_axle_state(axle, state)
    assert space is not None
    for ref in TANGENT_REFS:
        tangent = state.get(ref)
        world_tangent = space.to_world(tangent)
        assert world_tangent[Axis.Z] == pytest.approx(0.0, abs=1e-8)
        assert world_tangent[Axis.X] == pytest.approx(tangent[Axis.X], abs=1e-12)

    np.testing.assert_allclose(space.to_world(space.origin).data, 0.0, atol=1e-12)
    np.testing.assert_allclose(space.x.data, (1.0, 0.0, 0.0), atol=1e-12)
    assert space.z[Axis.X] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        space.rotation_chassis_to_world @ space.gravity.data,
        (0.0, 0.0, -1.0),
        atol=1e-12,
    )


def test_design_transform_is_axis_aligned(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=0.0, right_z=0.0)
    space = world_space_for_axle_state(axle, states[0])

    assert space is not None
    np.testing.assert_allclose(space.rotation_chassis_to_world, np.eye(3), atol=1e-8)
    _assert_supported_placement(axle, states[0])


@pytest.mark.parametrize(
    ("left_z", "right_z"),
    [(-30.0, -30.0), (20.0, -20.0), (-15.0, 25.0)],
)
def test_contacts_complete_axle_local_heave_and_roll_placement(
    test_data_dir, left_z, right_z
) -> None:
    axle, states = _solve_states(
        test_data_dir,
        left_z=left_z,
        right_z=right_z,
    )
    _assert_supported_placement(axle, states[0])


def test_single_axle_world_placement_does_not_infer_pitch(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    space = world_space_for_axle_state(axle, states[0])

    assert space is not None
    np.testing.assert_allclose(space.x.data, (1.0, 0.0, 0.0), atol=1e-12)
    assert space.z[Axis.X] == pytest.approx(0.0, abs=1e-12)


def test_sweep_helper_reports_one_space_per_state(test_data_dir) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    from kinematics.cli.io.sweep_loader import load_sweep

    states, _ = solve_sweep(axle, load_sweep(test_data_dir / "axle_sweep.yaml", axle))
    spaces = world_spaces_for_sweep(axle, states)

    assert len(spaces) == len(states)
    assert all(space is not None for space in spaces)


def test_banked_authored_design_is_rejected(test_data_dir) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    design = axle.initial_state()
    left = design.get(TANGENT_REFS[0])
    design.set(
        TANGENT_REFS[0],
        Point3((left[Axis.X], left[Axis.Y], float(left[Axis.Z]) + 1.0)),
    )

    assert axle.design_road_plane is None
    assert world_space_for_axle_state(axle, design) is None


def test_degenerate_current_contacts_are_rejected(test_data_dir) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    state = axle.initial_state().copy()
    state.set(TANGENT_REFS[1], state.get(TANGENT_REFS[0]))

    assert world_space_for_axle_state(axle, state) is None
