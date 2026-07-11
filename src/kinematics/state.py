"""
Core state management for suspension kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

import numpy as np

from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey


@dataclass
class SuspensionState:
    """
    Represents the complete state of a suspension system.

    This class manages point positions and solver metadata, providing methods
    for state manipulation, solver integration, and coordinate transformations.

    The point-key type is generic (:data:`~kinematics.core.point_ref.PointKey`):
    single-corner models key on ``PointID``, axle models key on ``PointRef``. The
    keys of a single state must be homogeneous so that ``free_points`` can be
    sorted for a stable variable ordering.

    Attributes:
        positions: Dictionary mapping point keys to 3D positions.
        free_points: Set of point keys that are free to move during solving.
        free_points_order: Sorted list of free point keys for consistent ordering.
    """

    positions: dict[PointKey, Point3]
    free_points: Set[PointKey]
    free_points_order: List[PointKey] = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize consistent ordering for free points.
        """
        self.free_points_order = sorted(list(self.free_points))

    @property
    def fixed_points(self) -> Set[PointKey]:
        """
        Points that are fixed (not free to move).
        """
        return set(self.positions.keys()) - self.free_points

    def get_free_array(self) -> np.ndarray:
        """
        Convert free points to flat array for solver.

        Extracts the positions of all free points and concatenates them into
        a single 1D array for use in the scipy loop. The order is determined
        by free_points_order for consistency.

        Returns:
            A flat numpy array containing [x1, y1, z1, x2, y2, z2, ...] coordinates
            for all free points in the consistent ordering.
        """
        arrays = [self.positions[pid].data for pid in self.free_points_order]
        return np.concatenate(arrays)

    def update_from_array(self, array: np.ndarray) -> None:
        """
        Update free points from solver array in-place.

        Takes a flat array from the numerical solver and updates the positions
        of free points in the suspension state. The array is reshaped and
        assigned according to the ordering in free_points_order.
        This modifies the state directly for performance.

        Args:
            array: Flat numpy array containing [x1, y1, z1, x2, y2, z2, ...]
                  coordinates for all free points. Must have length 3 * num_free_points.

        Raises:
            ValueError: If the array shape doesn't match the expected dimensions
                       based on the number of free points.
        """
        n_points = len(self.free_points_order)
        if array.shape != (n_points * 3,):
            raise ValueError(
                f"Array shape {array.shape} doesn't match expected ({n_points * 3},)"
            )

        positions_2d = array.reshape(n_points, 3)
        for i, point_id in enumerate(self.free_points_order):
            # Bind each Point3 directly to its slice of the solver parameter
            # buffer so least_squares can update positions without us copying
            # 3 floats per free point per iteration. from_trusted is the
            # explicit opt-in to this aliasing.
            self.positions[point_id] = Point3.from_trusted(positions_2d[i])

    def update_positions(self, new_positions: dict[PointKey, Point3]) -> None:
        """
        Replace positions dictionary in-place.

        Completely replaces the current positions dictionary with a new one.
        This modifies the state directly for performance, avoiding unnecessary
        copying when bulk updates are needed.

        Args:
            new_positions: Dictionary mapping point IDs to their new 3D coordinates.
                          Should contain entries for all points (both fixed and free).
        """
        self.positions = new_positions

    def copy(self) -> "SuspensionState":
        """
        Create a deep copy.
        """
        return SuspensionState(
            positions={pid: pos.copy() for pid, pos in self.positions.items()},
            free_points=self.free_points.copy(),
        )

    def get(self, point_id: PointKey) -> Point3:
        """
        Get position of a specific point.
        """
        return self.positions[point_id]

    def set(self, point_id: PointKey, position: Point3) -> None:
        """
        Set position of a specific point.
        """
        self.positions[point_id] = position.copy()

    def __getitem__(self, point_id: PointKey) -> Point3:
        """
        Allow dict-like access.
        """
        return self.positions[point_id]

    def __setitem__(self, point_id: PointKey, position: Point3) -> None:
        """
        Allow dict-like assignment.
        """
        self.positions[point_id] = position.copy()

    def __contains__(self, point_id: PointKey) -> bool:
        """
        Check if point exists.
        """
        return point_id in self.positions

    def items(self):
        """
        Iterate over (point_id, position) pairs.
        """
        return self.positions.items()

    def keys(self):
        """
        Iterate over point IDs.
        """
        return self.positions.keys()

    def values(self):
        """
        Iterate over positions.
        """
        return self.positions.values()
