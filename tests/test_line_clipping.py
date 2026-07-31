"""Tests for renderer-independent infinite-line clipping."""

from __future__ import annotations

import numpy as np
import pytest

from kinematics.cli.visualization.clipping import (
    Bounds3D,
    clip_infinite_line_to_bounds,
)
from kinematics.core.primitives.geometry import Point3, Vector3


@pytest.fixture
def unit_bounds() -> Bounds3D:
    return Bounds3D(np.zeros(3), np.full(3, 10.0))


def test_clips_line_crossing_all_three_dimensions(unit_bounds: Bounds3D) -> None:
    clipped = clip_infinite_line_to_bounds(
        Point3([5.0, 5.0, 5.0]),
        Vector3([1.0, 2.0, 3.0]),
        unit_bounds,
    )

    assert clipped is not None
    start, end = clipped
    assert start == pytest.approx([10.0 / 3.0, 5.0 / 3.0, 0.0])
    assert end == pytest.approx([20.0 / 3.0, 25.0 / 3.0, 10.0])


@pytest.mark.parametrize(
    ("point", "direction", "expected"),
    [
        ([5.0, 5.0, 5.0], [1.0, 0.0, 1.0], ([0.0, 5.0, 0.0], [10.0, 5.0, 10.0])),
        ([5.0, 4.0, 6.0], [1.0, 0.0, 0.0], ([0.0, 4.0, 6.0], [10.0, 4.0, 6.0])),
    ],
    ids=("parallel_to_one_slab_pair", "parallel_to_two_slab_pairs"),
)
def test_clips_line_parallel_to_slabs(
    unit_bounds: Bounds3D,
    point: list[float],
    direction: list[float],
    expected: tuple[list[float], list[float]],
) -> None:
    clipped = clip_infinite_line_to_bounds(point, direction, unit_bounds)

    assert clipped is not None
    assert clipped[0] == pytest.approx(expected[0])
    assert clipped[1] == pytest.approx(expected[1])


def test_clips_line_lying_on_slab(unit_bounds: Bounds3D) -> None:
    clipped = clip_infinite_line_to_bounds(
        [0.0, 5.0, 10.0],
        [0.0, 1.0, 0.0],
        unit_bounds,
    )

    assert clipped is not None
    assert clipped[0] == pytest.approx([0.0, 0.0, 10.0])
    assert clipped[1] == pytest.approx([0.0, 10.0, 10.0])


def test_rejects_line_outside_and_parallel_to_box(unit_bounds: Bounds3D) -> None:
    assert (
        clip_infinite_line_to_bounds(
            [-1.0, 5.0, 5.0],
            [0.0, 1.0, 0.0],
            unit_bounds,
        )
        is None
    )


def test_direction_reversal_reverses_endpoints(unit_bounds: Bounds3D) -> None:
    forward = clip_infinite_line_to_bounds(
        [5.0, 5.0, 5.0], [1.0, 2.0, 3.0], unit_bounds
    )
    reverse = clip_infinite_line_to_bounds(
        [5.0, 5.0, 5.0], [-1.0, -2.0, -3.0], unit_bounds
    )

    assert forward is not None
    assert reverse is not None
    assert reverse[0] == pytest.approx(forward[1])
    assert reverse[1] == pytest.approx(forward[0])


def test_treats_near_zero_direction_component_as_parallel(
    unit_bounds: Bounds3D,
) -> None:
    clipped = clip_infinite_line_to_bounds(
        [0.0, 5.0, 5.0],
        [1e-14, 1.0, 0.0],
        unit_bounds,
    )

    assert clipped is not None
    assert clipped[0] == pytest.approx([0.0, 0.0, 5.0])
    assert clipped[1] == pytest.approx([0.0, 10.0, 5.0])


@pytest.mark.parametrize(
    ("point", "direction"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]),
    ],
    ids=("zero_direction", "non_finite_point", "infinite_direction", "nan_direction"),
)
def test_rejects_non_finite_or_zero_line(
    unit_bounds: Bounds3D,
    point: list[float],
    direction: list[float],
) -> None:
    assert clip_infinite_line_to_bounds(point, direction, unit_bounds) is None


def test_bounds_accept_array_like_values_and_own_normalized_copies() -> None:
    minimum = np.array([-1, -2, -3])
    bounds = Bounds3D(minimum, [1.0, 2.0, 3.0])
    minimum[:] = 100

    assert bounds.minimum == pytest.approx([-1.0, -2.0, -3.0])
    assert bounds.maximum == pytest.approx([1.0, 2.0, 3.0])
    assert not bounds.minimum.flags.writeable
    assert not bounds.maximum.flags.writeable


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        ([0.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [np.inf, 1.0, 1.0]),
    ],
    ids=("wrong_shape", "reversed", "non_finite"),
)
def test_rejects_invalid_bounds(minimum: list[float], maximum: list[float]) -> None:
    with pytest.raises(ValueError):
        Bounds3D(minimum, maximum)
