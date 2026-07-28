"""Integration tests for flat WorldSpace placement of solved axle states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis, AxlePosition, PointID
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


def _pivot_pair(axle: AxleSuspension) -> tuple[Point3, Point3]:
    config = axle.config
    assert config is not None and config.axle_position is not None
    design = axle.initial_state()
    axle_x = 0.5 * sum(
        float(design.get(PointRef(side, PointID.WHEEL_CENTER))[Axis.X])
        for side in (Side.LEFT, Side.RIGHT)
    )
    ground_z = 0.5 * sum(
        float(design.get(ref)[Axis.Z]) for ref in TANGENT_REFS
    )
    sign = -1.0 if config.axle_position is AxlePosition.FRONT else 1.0
    pivot_x = axle_x + sign * config.wheelbase
    return (
        Point3(
            (
                pivot_x,
                0.0,
                ground_z + config.opposite_axle_axis_height,
            )
        ),
        Point3((pivot_x, 0.0, config.opposite_axle_axis_height)),
    )


def _assert_completed_placement(
    axle: AxleSuspension,
    state: SuspensionState,
) -> None:
    space = world_space_for_axle_state(axle, state)
    assert space is not None
    for ref in TANGENT_REFS:
        assert space.to_world(state.get(ref))[Axis.Z] == pytest.approx(
            0.0, abs=1e-8
        )
    chassis_pivot, world_target = _pivot_pair(axle)
    np.testing.assert_allclose(
        space.to_world(chassis_pivot).data,
        world_target.data,
        atol=1e-8,
    )
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
    _assert_completed_placement(axle, states[0])


@pytest.mark.parametrize(
    ("left_z", "right_z"),
    [(-30.0, -30.0), (20.0, -20.0), (-15.0, 25.0)],
)
def test_contacts_and_opposite_pivot_complete_bump_and_roll_placement(
    test_data_dir, left_z, right_z
) -> None:
    axle, states = _solve_states(
        test_data_dir,
        left_z=left_z,
        right_z=right_z,
    )
    _assert_completed_placement(axle, states[0])


def test_heading_is_projected_chassis_forward_with_no_yaw_input(test_data_dir) -> None:
    axle, states = _solve_states(test_data_dir, left_z=20.0, right_z=-20.0)
    space = world_space_for_axle_state(axle, states[0])

    assert space is not None
    expected = np.array((1.0, 0.0, 0.0)) - float(space.z[0]) * space.z.data
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(space.x.data, expected, atol=1e-12)


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

    assert axle.design_ground is None
    assert world_space_for_axle_state(axle, design) is None


def test_degenerate_current_contacts_are_rejected(test_data_dir) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    state = axle.initial_state().copy()
    state.set(TANGENT_REFS[1], state.get(TANGENT_REFS[0]))

    assert world_space_for_axle_state(axle, state) is None
