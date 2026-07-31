"""Renderer-independent clipping helpers for visualization geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kinematics.core.primitives.geometry import (
    Direction3,
    Point3,
    Vector3,
    extract_array,
)

type Vector3Input = (
    Point3
    | Vector3
    | Direction3
    | NDArray[np.float64]
    | list[float]
    | tuple[float, float, float]
)


def _vector_array(value: Vector3Input) -> NDArray[np.float64]:
    """Copy a supported three-component value into a float array."""
    array = np.array(extract_array(value), dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"Expected a three-component value, got shape {array.shape}")
    return array


@dataclass(frozen=True, init=False, eq=False)
class Bounds3D:
    """Closed 3-D axis-aligned bounds."""

    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]

    def __init__(self, minimum: Vector3Input, maximum: Vector3Input) -> None:
        """Create bounds from array-like minimum and maximum corners."""
        minimum_array = _vector_array(minimum)
        maximum_array = _vector_array(maximum)

        if not np.all(np.isfinite(minimum_array)) or not np.all(
            np.isfinite(maximum_array)
        ):
            raise ValueError("Bounds must contain only finite coordinates")
        if np.any(minimum_array > maximum_array):
            raise ValueError("Bounds minimum must not exceed maximum")

        # Make the normalized values safe to share from a frozen value object.
        minimum_array.setflags(write=False)
        maximum_array.setflags(write=False)
        object.__setattr__(self, "minimum", minimum_array)
        object.__setattr__(self, "maximum", maximum_array)


def clip_infinite_line_to_bounds(
    point: Vector3Input,
    direction: Vector3Input,
    bounds: Bounds3D,
    *,
    direction_tolerance: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Clip an infinite 3-D line to an axis-aligned bounding box.

    The returned points are ordered by increasing parameter along ``direction``.
    A direction component whose magnitude is within ``direction_tolerance`` of
    the largest component is treated as parallel to that coordinate's slab.

    Args:
        point: Any point on the infinite line.
        direction: A non-zero line direction; it need not be normalized.
        bounds: Closed axis-aligned clipping bounds.
        direction_tolerance: Relative threshold for parallel components.

    Returns:
        The clipped segment endpoints, or ``None`` when the line is invalid or
        does not intersect the bounds.

    Raises:
        ValueError: If ``direction_tolerance`` is outside ``[0, 1)`` or either
            vector does not have three components.
    """
    if not np.isfinite(direction_tolerance) or not 0 <= direction_tolerance < 1:
        raise ValueError("direction_tolerance must be finite and in [0, 1)")

    point_array = _vector_array(point)
    direction_array = _vector_array(direction)
    if not np.all(np.isfinite(point_array)) or not np.all(np.isfinite(direction_array)):
        return None

    direction_scale = float(np.max(np.abs(direction_array)))
    if direction_scale == 0.0:
        return None

    # Normalizing by the largest component makes the parallel decision and the
    # clipping arithmetic invariant to arbitrary scaling of the direction.
    normalized_direction = direction_array / direction_scale
    parameter_min = -np.inf
    parameter_max = np.inf

    for coordinate in range(3):
        coordinate_direction = float(normalized_direction[coordinate])
        coordinate_point = float(point_array[coordinate])
        lower = float(bounds.minimum[coordinate])
        upper = float(bounds.maximum[coordinate])

        if abs(coordinate_direction) <= direction_tolerance:
            if coordinate_point < lower or coordinate_point > upper:
                return None
            continue

        first = (lower - coordinate_point) / coordinate_direction
        second = (upper - coordinate_point) / coordinate_direction
        slab_min, slab_max = min(first, second), max(first, second)
        parameter_min = max(parameter_min, slab_min)
        parameter_max = min(parameter_max, slab_max)
        if parameter_min > parameter_max:
            return None

    first_endpoint = point_array + parameter_min * normalized_direction
    second_endpoint = point_array + parameter_max * normalized_direction

    # Suppress harmless round-off beyond a face so renderers receive points
    # that are strictly contained by the requested box.
    clipped_first = np.clip(first_endpoint, bounds.minimum, bounds.maximum).astype(
        np.float64, copy=False
    )
    clipped_second = np.clip(second_endpoint, bounds.minimum, bounds.maximum).astype(
        np.float64, copy=False
    )
    return clipped_first, clipped_second
