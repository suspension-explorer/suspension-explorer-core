import numpy as np
import pytest

from kinematics.core.enums import Axis
from kinematics.core.geometry import Direction3
from kinematics.core.types import PointTargetAxis, PointTargetVector, WorldAxisSystem
from kinematics.targets import resolve_target


def test_resolve_axis_targets_returns_unit_axes():
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.X)).data, WorldAxisSystem.X.data
    )
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.Y)).data, WorldAxisSystem.Y.data
    )
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.Z)).data, WorldAxisSystem.Z.data
    )


def test_resolve_vector_target_normalizes():
    direction = resolve_target(PointTargetVector(Direction3([10.0, 0.0, 0.0])))

    np.testing.assert_allclose(direction.data, WorldAxisSystem.X.data)
    assert np.isclose(np.linalg.norm(direction.data), 1.0)


def test_resolve_raw_vector_target_normalizes():
    direction = resolve_target(PointTargetVector(np.array([0.0, 0.0, 10.0])))

    np.testing.assert_allclose(direction.data, WorldAxisSystem.Z.data)
    assert np.isclose(np.linalg.norm(direction.data), 1.0)


def test_resolve_vector_target_zero_raises():
    with pytest.raises(ValueError):
        resolve_target(PointTargetVector(np.array([0.0, 0.0, 0.0])))
