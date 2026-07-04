"""
Composite type definitions for suspension kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple, Union

import numpy as np

from kinematics.core.enums import Axis, TargetPositionMode
from kinematics.core.geometry import Direction3
from kinematics.core.point_ref import PointKey


def frozen_unit_axis(values: tuple[float, float, float]) -> np.ndarray:
    """
    Create a frozen (immutable) unit-axis array from the given values.

    Args:
        values: A tuple of three floats representing the axis direction.

    Returns:
        A numpy array with writeable flag set to False to prevent mutation.
    """
    # Build a non-writeable unit-axis array so the shared WorldAxisSystem
    # constants cannot be mutated through their .data attribute.
    arr = np.array(values, dtype=np.float64)
    arr.flags.writeable = False
    return arr


class WorldAxisSystem:
    """
    World coordinate system unit axis directions.

    Usage:
        WorldAxisSystem.X  # -> Direction3 along [1, 0, 0]
        WorldAxisSystem.Y  # -> Direction3 along [0, 1, 0]
        WorldAxisSystem.Z  # -> Direction3 along [0, 0, 1]
    """

    X: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((1.0, 0.0, 0.0)))
    Y: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((0.0, 1.0, 0.0)))
    Z: Final[Direction3] = Direction3.from_trusted(frozen_unit_axis((0.0, 0.0, 1.0)))


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

    point_id: PointKey
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
