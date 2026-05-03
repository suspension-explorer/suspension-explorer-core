"""
Target resolution utilities for suspension kinematics.

This module provides functions to resolve target directions into world coordinate
directions.
"""

from kinematics.core.enums import Axis
from kinematics.core.geometry import Direction3
from kinematics.core.types import (
    PointTargetAxis,
    PointTargetDirection,
    PointTargetVector,
    WorldAxisSystem,
)


def resolve_target(target: PointTargetDirection) -> Direction3:
    """
    Resolves a target direction specification into a unit direction.

    Handles both axis-based directions (X, Y, Z) and arbitrary vector directions.

    Args:
        target: The target direction specification to resolve.

    Returns:
        A Direction3 representing the target direction.
    """
    if isinstance(target, PointTargetAxis):
        if target.axis is Axis.X:
            return WorldAxisSystem.X
        if target.axis is Axis.Y:
            return WorldAxisSystem.Y
        if target.axis is Axis.Z:
            return WorldAxisSystem.Z
        raise ValueError(f"Unsupported axis: {target.axis!r}")

    if isinstance(target, PointTargetVector):
        return Direction3(target.vector)

    raise TypeError(f"Unsupported target type: {type(target)!r}")
