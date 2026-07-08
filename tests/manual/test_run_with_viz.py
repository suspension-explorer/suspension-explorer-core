from pathlib import Path

import numpy as np
import pytest

from kinematics.constraints import DistanceConstraint
from kinematics.core.constants import TEST_TOLERANCE
from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.types import PointTargetAxis, SweepConfig
from kinematics.io import load_geometry
from kinematics.main import solve_sweep
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import PointTarget


@pytest.fixture
def displacements():
    n_steps = 31

    hub_range = [-30, 90]
    hub_displacements = list(np.linspace(hub_range[0], hub_range[1], n_steps))

    steer_range = [-30, 30]
    steer_displacements = list(np.linspace(steer_range[0], steer_range[1], n_steps))

    return hub_displacements, steer_displacements


@pytest.fixture
def sweep_config_fixture(displacements):
    hub_displacements, steer_displacements = displacements

    # Create hub displacement sweep.
    hub_targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(Axis.Z),
            value=x,
            mode=TargetPositionMode.RELATIVE,
        )
        for x in hub_displacements
    ]

    # Create steer sweep.
    steer_targets = [
        PointTarget(
            point_id=PointID.TRACKROD_INBOARD,
            direction=PointTargetAxis(Axis.Y),
            value=x,
            mode=TargetPositionMode.RELATIVE,
        )
        for x in steer_displacements
    ]

    # Create sweep config.
    sweep_config = SweepConfig([hub_targets, steer_targets])

    return sweep_config


@pytest.mark.manual
def test_run_solver(
    double_wishbone_geometry_file: Path, sweep_config_fixture, displacements
) -> None:
    """Run solver with visualization (manual test)."""
    hub_displacements, _ = displacements

    suspension = load_geometry(double_wishbone_geometry_file)
    if suspension.TYPE_KEY not in (
        "double_wishbone",
        "double_wishbone_front",
        "double_wishbone_rear",
    ):
        raise ValueError("Manual viz test only supports double wishbone suspensions")

    # Solve for all positions.
    position_states, _ = solve_sweep(suspension, sweep_config_fixture)

    print("Solve complete, verifying constraints...")

    # Get initial positions for comparison using the suspension.
    derived_manager = DerivedPointsManager(suspension.derived_spec())

    initial_state = suspension.initial_state()
    derived_manager.update_in_place(initial_state.positions)
    initial_positions = initial_state.positions.copy()

    # Get only the length constraints for verification
    all_constraints = suspension.constraints()

    length_constraints = [
        c for c in all_constraints if isinstance(c, DistanceConstraint)
    ]
    target_point_id = PointID.WHEEL_CENTER

    # Verify constraints are maintained.
    for state, displacement in zip(position_states, hub_displacements):
        # Verify length constraints.
        for constraint in length_constraints:
            p1 = state.positions[constraint.p1]
            p2 = state.positions[constraint.p2]
            current_length = np.linalg.norm(p1 - p2)

            assert (
                np.abs(current_length - constraint.target_distance) < TEST_TOLERANCE
            ), (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.p1.name} to {constraint.p2.name}"
            )

        # Verify target point z position.
        target_point_position = state.positions[target_point_id]
        initial_target_point_position = initial_positions[target_point_id]
        target_z = initial_target_point_position[2] + displacement

        assert np.abs(target_point_position[2] - target_z) < TEST_TOLERANCE, (
            f"Failed to maintain {target_point_id} at displacement {displacement}"
        )

    print("Creating animation...")

    # Defer visualization imports to avoid collection errors when matplotlib is missing.
    from kinematics.visualization.animation import create_animation
    from kinematics.visualization.main import SuspensionVisualizer, WheelVisualization

    # Extract positions from SuspensionState objects for animation.
    position_states_positions = [state.positions for state in position_states]
    output_path = (
        Path(__file__).parent.parent / "data" / "manual" / "suspension_motion.mp4"
    )

    r_aspect = 0.55
    x_section = 270
    x_diameter = 13 * 25.4

    wheel_config = WheelVisualization(
        diameter=x_diameter + r_aspect * x_section * 2,
        width=225,
    )

    visualization_links = suspension.get_visualization_links()
    visualizer = SuspensionVisualizer(visualization_links, wheel_config)
    create_animation(
        position_states_positions,
        initial_positions,
        visualizer,
        output_path,
    )
