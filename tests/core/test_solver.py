import numpy as np
import pytest

from kinematics.core.constraints import DistanceConstraint
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.primitives.enums import Axis, PointID, TargetPositionMode
from kinematics.core.solver import (
    SolverConfig,
    solve_least_squares_problem,
    solve_suspension_sweep,
)
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig


@pytest.fixture
def simple_positions():
    from kinematics.core.primitives.geometry import Point3

    positions_dict = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3([-1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: Point3([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([0.0, 1.0, 0.0]),
    }
    return positions_dict


@pytest.fixture
def length_forward_leg(simple_positions):
    x_forward_leg = np.linalg.norm(
        simple_positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        - simple_positions[PointID.LOWER_WISHBONE_OUTBOARD],
    )
    return x_forward_leg


@pytest.fixture
def length_rearward_leg(simple_positions):
    x_rearward_leg = np.linalg.norm(
        simple_positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        - simple_positions[PointID.LOWER_WISHBONE_OUTBOARD]
    )
    return x_rearward_leg


@pytest.fixture
def simple_constraints(simple_positions, length_forward_leg, length_rearward_leg):
    return [
        DistanceConstraint(
            p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            target_distance=length_forward_leg,
        ),
        DistanceConstraint(
            p1=PointID.LOWER_WISHBONE_INBOARD_REAR,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            target_distance=length_rearward_leg,
        ),
    ]


@pytest.fixture
def simple_sweep_config():
    displacements = [0.0, 0.5, 1.0]
    point_targets = [
        PointTarget(
            point_id=PointID.LOWER_WISHBONE_OUTBOARD,
            direction=PointTargetAxis(Axis.Z),
            value=d,
            mode=TargetPositionMode.RELATIVE,
        )
        for d in displacements
    ]
    return SweepConfig([point_targets])


def make_noop_derived_manager():
    # No derived points -> empty spec
    spec = DerivedPointsSpec(functions={}, dependencies={})
    return DerivedPointsManager(spec)


def test_solve_sweep(
    simple_positions,
    simple_constraints,
    simple_sweep_config,
    length_forward_leg,
    length_rearward_leg,
):
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create SuspensionState instead of separate positions and free_points
    initial_state = SuspensionState(positions=simple_positions, free_points=free_points)

    # Extract displacement values for assertions
    displacement_values = [
        target.value for target in simple_sweep_config.target_sweeps[0]
    ]

    states, solver_stats = solve_suspension_sweep(
        initial_state=initial_state,
        constraints=simple_constraints,
        sweep_config=simple_sweep_config,
        derived_manager=make_noop_derived_manager(),
        solver_config=SolverConfig(ftol=1e-6, xtol=1e-6, verbose=0),
    )

    assert len(states) == len(displacement_values)
    assert len(solver_stats) == len(displacement_values)

    # Check solver infos have expected structure
    for stat in solver_stats:
        assert hasattr(stat, "converged")
        assert hasattr(stat, "nfev")
        assert hasattr(stat, "max_residual")
        assert stat.converged is True  # Should converge for this simple test

    # Check each state maintains constraints
    for i, state in enumerate(states):
        p_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        p_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        p_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        # Distance constraints
        assert np.linalg.norm(p_outboard - p_front) == pytest.approx(
            length_forward_leg, rel=TEST_TOLERANCE
        )
        assert np.linalg.norm(p_outboard - p_rear) == pytest.approx(
            length_rearward_leg, rel=TEST_TOLERANCE
        )

        # Target displacement
        assert p_outboard[2] == pytest.approx(
            displacement_values[i], rel=TEST_TOLERANCE
        )


def test_solve_least_squares_problem_rejects_underdetermined_lm():
    def residual_function(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[1]])

    with pytest.raises(ValueError, match="System is underdetermined"):
        solve_least_squares_problem(
            residual_function=residual_function,
            x_0=np.zeros(2),
            n_residuals=1,
        )


def test_solve_sweep_rejects_converged_infeasible_target(
    simple_positions,
    simple_constraints,
):
    initial_state = SuspensionState(
        positions=simple_positions,
        free_points={PointID.LOWER_WISHBONE_OUTBOARD},
    )
    infeasible_target = PointTarget(
        point_id=PointID.LOWER_WISHBONE_OUTBOARD,
        direction=PointTargetAxis(Axis.Z),
        value=10.0,
        mode=TargetPositionMode.ABSOLUTE,
    )

    with pytest.raises(RuntimeError, match="sweep step 0.*Worst residual row"):
        solve_suspension_sweep(
            initial_state=initial_state,
            constraints=simple_constraints,
            sweep_config=SweepConfig([[infeasible_target]]),
            derived_manager=make_noop_derived_manager(),
        )
