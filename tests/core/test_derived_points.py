from functools import partial

import numpy as np
import pytest

from kinematics.core.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.enums import PointID


@pytest.fixture
def sample_positions():
    from kinematics.core.primitives.geometry import Point3

    # Set up a simple configuration:
    # - Axle inboard at origin
    # - Axle outboard at x=2
    # - This makes the axle 2 units wide pointing along x-axis
    return {
        PointID.AXLE_INBOARD: Point3([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: Point3([2.0, 0.0, 0.0]),
    }


def test_axle_midpoint(sample_positions):
    result = get_axle_midpoint(sample_positions)
    np.testing.assert_array_equal(result.data, np.array([1.0, 0.0, 0.0]))


def test_wheel_center(sample_positions):
    # Positive wheel offset (ET) moves centerline inboard from hub face.
    # With axle outboard at x=2.0 and offset of +0.5, center should be at x=1.5.
    result = get_wheel_center(sample_positions, wheel_offset=0.5)
    np.testing.assert_array_equal(result.data, np.array([1.5, 0.0, 0.0]))


def test_wheel_center_zero_offset(sample_positions):
    # Zero wheel offset leaves the centerline at the hub face (axle outboard point).
    # With axle outboard at x=2.0 and offset of 0.0, center should be at x=2.0.
    result = get_wheel_center(sample_positions, wheel_offset=0.0)
    np.testing.assert_array_equal(result.data, np.array([2.0, 0.0, 0.0]))


def test_wheel_center_negative_offset(sample_positions):
    # Negative wheel offset moves centerline outboard from hub face.
    # With axle outboard at x=2.0 and offset of -0.5, center should be at x=2.5.
    result = get_wheel_center(sample_positions, wheel_offset=-0.5)
    np.testing.assert_array_equal(result.data, np.array([2.5, 0.0, 0.0]))


def test_wheel_inboard_outboard(sample_positions):
    # First compute wheel center
    wheel_center = get_wheel_center(sample_positions, wheel_offset=0.5)
    positions_dict = sample_positions.copy()
    positions_dict[PointID.WHEEL_CENTER] = wheel_center

    # Test inboard point (should be 0.5 units inboard of wheel center)
    inboard = get_wheel_inboard(positions_dict, wheel_width=1.0)
    np.testing.assert_array_equal(inboard.data, np.array([1.0, 0.0, 0.0]))

    # Test outboard point (should be 0.5 units outboard of wheel center)
    outboard = get_wheel_outboard(positions_dict, wheel_width=1.0)
    np.testing.assert_array_equal(outboard.data, np.array([2.0, 0.0, 0.0]))


def test_dependency_manager_basic(sample_positions):
    """
    Test that the DerivedPointsManager can calculate axle midpoint.
    """
    functions = {
        PointID.AXLE_MIDPOINT: get_axle_midpoint,
    }
    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
    }
    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointsManager(spec)

    manager.update_in_place(sample_positions)

    # Should have original positions plus the derived point
    assert len(sample_positions) == 3
    assert PointID.AXLE_MIDPOINT in sample_positions
    np.testing.assert_array_equal(
        sample_positions[PointID.AXLE_MIDPOINT].data, np.array([1.0, 0.0, 0.0])
    )


def test_dependency_manager_computes_relevant_point_jacobian(sample_positions):
    """
    Test that point Jacobians include only requested variable dependencies.
    """
    spec = DerivedPointsSpec(
        functions={PointID.AXLE_MIDPOINT: get_axle_midpoint},
        dependencies={
            PointID.AXLE_MIDPOINT: {
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
            }
        },
    )
    manager = DerivedPointsManager(spec)

    blocks = manager.compute_point_jacobian(
        PointID.AXLE_MIDPOINT,
        sample_positions,
        variable_points={PointID.AXLE_OUTBOARD},
    )

    assert set(blocks) == {PointID.AXLE_OUTBOARD}
    np.testing.assert_allclose(blocks[PointID.AXLE_OUTBOARD], 0.5 * np.eye(3))


def test_dependency_manager_complex(sample_positions):
    """
    Test that the DerivedPointsManager can handle dependencies between derived points.
    """
    functions = {
        PointID.AXLE_MIDPOINT: get_axle_midpoint,
        PointID.WHEEL_CENTER: partial(get_wheel_center, wheel_offset=0.5),
        PointID.WHEEL_INBOARD: partial(get_wheel_inboard, wheel_width=1.0),
        PointID.WHEEL_OUTBOARD: partial(get_wheel_outboard, wheel_width=1.0),
    }
    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
    }
    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointsManager(spec)

    manager.update_in_place(sample_positions)

    # Should have original positions plus all derived points
    assert len(sample_positions) == 6
    assert PointID.AXLE_MIDPOINT in sample_positions
    assert PointID.WHEEL_CENTER in sample_positions
    assert PointID.WHEEL_INBOARD in sample_positions
    assert PointID.WHEEL_OUTBOARD in sample_positions

    # Check values
    np.testing.assert_array_equal(
        sample_positions[PointID.AXLE_MIDPOINT].data, np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_CENTER].data, np.array([1.5, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_INBOARD].data, np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_OUTBOARD].data, np.array([2.0, 0.0, 0.0])
    )


def test_circular_dependency_detection():
    """
    Test that circular dependencies are detected.
    """

    def dummy_func(positions):
        return positions[PointID.AXLE_INBOARD]

    functions = {
        PointID.WHEEL_CENTER: dummy_func,
        PointID.WHEEL_OUTBOARD: dummy_func,
    }
    dependencies = {
        PointID.WHEEL_CENTER: {PointID.WHEEL_OUTBOARD},
        PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER},
    }

    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)

    with pytest.raises(ValueError, match="Circular dependency detected"):
        DerivedPointsManager(spec)
