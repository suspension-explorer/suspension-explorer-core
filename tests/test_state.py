import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.state import SuspensionState


def test_suspension_state_construction():
    # Test data setup.
    positions_data = {
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: Point3([4.0, 5.0, 6.0]),
    }
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create state.
    state = SuspensionState(positions=positions_data, free_points=free_points)

    # Verify positions are stored correctly.
    np.testing.assert_array_equal(
        state.positions[PointID.LOWER_WISHBONE_OUTBOARD].data,
        np.array([1.0, 2.0, 3.0]),
    )

    # Verify free points are stored.
    assert state.free_points == {PointID.LOWER_WISHBONE_OUTBOARD}

    # Verify fixed points are computed correctly.
    assert state.fixed_points == {PointID.UPPER_WISHBONE_OUTBOARD}


def test_suspension_state_array_conversion():
    """
    Test the array conversion methods in SuspensionState.
    """
    positions_data = {
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: Point3([4.0, 5.0, 6.0]),
    }
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD}

    # Create state
    state = SuspensionState(positions=positions_data, free_points=free_points)

    # Test extraction
    arr = state.get_free_array()
    # Should be sorted order: LOWER_WISHBONE_OUTBOARD, UPPER_WISHBONE_OUTBOARD
    expected_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(arr, expected_flat)

    # Test update from array
    new_array = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    state.update_from_array(new_array)

    # Verify positions were updated
    np.testing.assert_array_equal(
        state.positions[PointID.LOWER_WISHBONE_OUTBOARD].data,
        np.array([7.0, 8.0, 9.0]),
    )
    np.testing.assert_array_equal(
        state.positions[PointID.UPPER_WISHBONE_OUTBOARD].data,
        np.array([10.0, 11.0, 12.0]),
    )
