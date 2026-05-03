"""
Composite type definitions for suspension kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple, Union

import numpy as np

from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.geometry import Direction3, Point3


def make_point3(data) -> Point3:
    """
    Creates a Point3 from input data.

    Handles passthrough for dual-number types (automatic differentiation).

    Args:
        data: Input data convertible to a 3-element array, or a Point3.

    Returns:
        A Point3 instance.

    Raises:
        ValueError: If the input cannot be shaped into a 3-element array.
    """
    # Passthrough for dual-number types (automatic differentiation).
    if hasattr(data, "deriv"):
        return data

    if isinstance(data, Point3):
        return data

    return Point3(data)


class WorldAxisSystem:
    """
    World coordinate system unit axis directions.

    Usage:
        WorldAxisSystem.X  # -> Direction3 along [1, 0, 0]
        WorldAxisSystem.Y  # -> Direction3 along [0, 1, 0]
        WorldAxisSystem.Z  # -> Direction3 along [0, 0, 1]
    """

    X: Final[Direction3] = Direction3.from_trusted(
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    Y: Final[Direction3] = Direction3.from_trusted(
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    Z: Final[Direction3] = Direction3.from_trusted(
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )


@dataclass
class SweepConfig:
    """
    Configuration for a parametric sweep over multiple target dimensions.

    Each inner list represents one sweep dimension (e.g., bump travel, steering angle).
    All dimensions must have the same length - the sweep will iterate through
    corresponding indices across all dimensions simultaneously.

    Example:
        bump_targets = [PointTarget(..., value=-30), ..., PointTarget(..., value=30)]
        steer_targets = [PointTarget(..., value=-10), ..., PointTarget(..., value=10)]
        config = SweepConfig([bump_targets, steer_targets])
    """

    target_sweeps: list[list["PointTarget"]]

    def __post_init__(self):
        if not self.target_sweeps:
            return

        lengths = [len(sweep) for sweep in self.target_sweeps]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All sweep dimensions must have the same length. Got: {lengths}"
            )

    @property
    def n_steps(self) -> int:
        """
        Number of steps in the sweep.
        """
        if not self.target_sweeps:
            return 0
        return len(self.target_sweeps[0])


class PointTarget(NamedTuple):
    """
    Defines a target constraint for a specific point during kinematic solving.

    The mode determines how the value is interpreted initially, but all targets
    are converted to absolute coordinates before solving begins.

    Attributes:
        point_id: The point to constrain
        direction: Direction along which to apply the target
        value: Target value (interpretation depends on mode)
        mode: Whether value is relative displacement or absolute coordinate
    """

    point_id: PointID
    direction: "PointTargetDirection"
    value: float
    mode: TargetPositionMode = TargetPositionMode.RELATIVE


@dataclass(slots=True, frozen=True)
class PointTargetAxis:
    """
    A target direction defined by one of the principal axes.

    Attributes:
        axis (Axis): The axis to use as the target direction.
    """

    axis: Axis


@dataclass(slots=True, frozen=True)
class PointTargetVector:
    """
    A target direction defined by an arbitrary vector.

    Attributes:
        vector (Direction3): The direction defining the target.
    """

    vector: Direction3


PointTargetDirection = Union[PointTargetAxis, PointTargetVector]
