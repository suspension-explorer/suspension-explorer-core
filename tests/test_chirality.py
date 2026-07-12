"""Focused tests for signed-volume chirality constraints."""

import numpy as np
import pytest

from kinematics.constraints import ScalarTripleProductConstraint
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.solver import ResidualComputer
from kinematics.state import SuspensionState

POINTS = (
    PointID.ROCKER_AXIS_FRONT,
    PointID.ROCKER_AXIS_REAR,
    PointID.PUSHROD_INBOARD,
    PointID.DROPLINK_ROCKER,
)


def _positions() -> dict[PointKey, Point3]:
    return {
        POINTS[0]: Point3([0.0, 0.0, 0.0]),
        POINTS[1]: Point3([2.0, 0.0, 0.0]),
        POINTS[2]: Point3([0.0, 3.0, 0.0]),
        POINTS[3]: Point3([0.0, 0.0, 4.0]),
    }


def test_chirality_constraint_rejects_mirrored_branch() -> None:
    positions = _positions()
    constraint = ScalarTripleProductConstraint(
        POINTS[0],
        POINTS[1],
        POINTS[2],
        POINTS[3],
        target_volume=24.0,
        scale=24.0,
    )

    assert constraint.residual(positions) == pytest.approx(0.0)
    positions[POINTS[3]] = Point3([0.0, 0.0, -4.0])
    assert constraint.residual(positions) == pytest.approx(-2.0)


def test_chirality_solver_jacobian_matches_finite_difference() -> None:
    positions = _positions()
    constraint = ScalarTripleProductConstraint(
        POINTS[0],
        POINTS[1],
        POINTS[2],
        POINTS[3],
        target_volume=24.0,
        scale=24.0,
    )
    state = SuspensionState(positions, set[PointKey](POINTS))
    manager = DerivedPointsManager(DerivedPointsSpec({}, {}))
    computer = ResidualComputer([constraint], manager, state.copy(), 0)
    free_array = state.get_free_array()

    analytic = computer.compute_jacobian(free_array, [])[0]
    numerical = np.zeros_like(analytic)
    step = 1e-5
    for index in range(free_array.size):
        high = free_array.copy()
        low = free_array.copy()
        high[index] += step
        low[index] -= step
        numerical[index] = (
            computer.compute(high, [])[0] - computer.compute(low, [])[0]
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, atol=1e-7)
