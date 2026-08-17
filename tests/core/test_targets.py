import numpy as np
import pytest

from kinematics.core.coordinates import (
    ChassisAxisSystem,
    CoordinateAxis,
    CoordinateVector,
    resolve_direction,
)
from kinematics.core.enums import Axis
from kinematics.core.primitives.geometry import Direction3


def test_resolve_axis_targets_returns_unit_axes():
    np.testing.assert_allclose(
        resolve_direction(CoordinateAxis(Axis.X)).data, ChassisAxisSystem.X.data
    )
    np.testing.assert_allclose(
        resolve_direction(CoordinateAxis(Axis.Y)).data, ChassisAxisSystem.Y.data
    )
    np.testing.assert_allclose(
        resolve_direction(CoordinateAxis(Axis.Z)).data, ChassisAxisSystem.Z.data
    )


def test_resolve_vector_target_normalizes():
    direction = resolve_direction(CoordinateVector(Direction3([10.0, 0.0, 0.0])))

    np.testing.assert_allclose(direction.data, ChassisAxisSystem.X.data)
    assert np.isclose(np.linalg.norm(direction.data), 1.0)


def test_resolve_vector_target_zero_raises():
    with pytest.raises(ValueError):
        CoordinateVector(Direction3([0.0, 0.0, 0.0]))
