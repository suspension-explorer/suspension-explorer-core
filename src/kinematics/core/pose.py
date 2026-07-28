"""Post-solve chassis placement in a flat, level WorldSpace.

Solver state and metrics remain in chassis space.  This module supplies the
rigid transform into WorldSpace, whose ground is always ``Z = 0`` and whose
gravity direction is always ``-Z``.  There are no gravity or inclined-ground
model variants.

A single axle supplies its current left and right wheel-ground tangent points.
The vehicle configuration supplies a chassis-fixed centreline point on the
unmodelled axle's lateral pivot axis.  That point remains fixed at its authored
World coordinates, closing the otherwise underdetermined chassis pitch.

The design contract is explicit: design chassis and World axes are aligned and
the design ground datum is horizontal.  A materially banked design datum is
therefore rejected instead of being silently reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Sequence

import numpy as np

from kinematics.core.enums import Axis, AxlePosition, PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointRef, Side

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension

__all__ = [
    "WorldSpace",
    "world_space_for_axle_state",
    "world_spaces_for_sweep",
]

_ORTHONORMAL_TOLERANCE = 1e-9
_DESIGN_ALIGNMENT_TOLERANCE = 1e-7
_CONSTRAINT_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class WorldSpace:
    """A chassis-to-World rigid transform.

    ``x``, ``y``, and ``z`` are the World axes expressed as chassis-space unit
    vectors and form the rows of :attr:`rotation_chassis_to_world`. ``origin``
    is the chassis-space point that maps to World ``(0, 0, 0)``.
    """

    x: Direction3
    y: Direction3
    z: Direction3
    origin: Point3

    def __post_init__(self) -> None:
        """Require a right-handed orthonormal triad."""
        axes = np.vstack((self.x.data, self.y.data, self.z.data))
        orthonormal = np.allclose(axes @ axes.T, np.eye(3), atol=_ORTHONORMAL_TOLERANCE)
        if not orthonormal or float(np.linalg.det(axes)) < 0.0:
            raise ValueError(
                "World-space axes must form a right-handed orthonormal triad"
            )

    @property
    def gravity(self) -> Direction3:
        """Return World ``-Z`` expressed as a chassis-space direction."""
        return -self.z

    @property
    def rotation_chassis_to_world(self) -> np.ndarray:
        """Return the 3x3 rotation taking chassis vectors to World vectors."""
        return np.vstack((self.x.data, self.y.data, self.z.data))

    def to_world(self, point: Point3) -> Point3:
        """Map a chassis-space point into World coordinates."""
        offset = (point - self.origin).data
        return Point3(self.rotation_chassis_to_world @ offset)


def _design_axle_x(state: SuspensionState) -> float | None:
    """Return the authored axle centre X from the two wheel centres."""
    left = float(state.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))[Axis.X])
    right = float(state.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))[Axis.X])
    axle_x = 0.5 * (left + right)
    return axle_x if isfinite(axle_x) else None


def _opposite_axle_pivot(
    axle: AxleSuspension,
) -> tuple[Point3, Point3] | None:
    """Return the fixed pivot reference in chassis and World coordinates."""
    config = axle.config
    design_ground = axle.design_ground
    if (
        config is None
        or config.axle_position is None
        or design_ground is None
        or not np.allclose(
            design_ground.normal.data,
            (0.0, 0.0, 1.0),
            atol=_DESIGN_ALIGNMENT_TOLERANCE,
        )
    ):
        return None

    ground_z = -design_ground.offset_mm
    axle_x = _design_axle_x(axle.initial_state())
    if axle_x is None:
        return None

    wheelbase = config.wheelbase
    height = config.opposite_axle_axis_height
    if not all(isfinite(value) for value in (ground_z, wheelbase, height)):
        return None

    longitudinal_sign = (
        -1.0 if config.axle_position is AxlePosition.FRONT else 1.0
    )
    pivot_x = axle_x + longitudinal_sign * wheelbase
    chassis_pivot = Point3((pivot_x, 0.0, ground_z + height))
    world_target = Point3((pivot_x, 0.0, height))
    return chassis_pivot, world_target


def _solve_world_up(
    left: Point3,
    right: Point3,
    pivot: Point3,
    height: float,
) -> Direction3 | None:
    """Solve the upward unit normal satisfying both ground contacts.

    Each tangent must map to World ``Z = 0`` while the fixed pivot maps to
    World height ``height``:

    ``n · (q_left - pivot) = n · (q_right - pivot) = -height``.

    The two linear equations and unit-length constraint ordinarily yield two
    candidates.  The candidate with the strongest positive chassis-Z component
    is the physical branch.  A tied upward branch is genuinely ambiguous.
    """
    matrix = np.vstack(((left - pivot).data, (right - pivot).data))
    if not np.isfinite(matrix).all() or not isfinite(height):
        return None

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if (
        singular_values.shape != (2,)
        or singular_values[0] < EPS_GEOMETRIC
        or singular_values[1]
        <= EPS_GEOMETRIC * max(1.0, float(singular_values[0]))
    ):
        return None

    rhs = np.array((-height, -height), dtype=np.float64)
    minimum_norm, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    minimum_norm_sq = float(np.dot(minimum_norm, minimum_norm))
    feasibility_tolerance = _CONSTRAINT_RELATIVE_TOLERANCE * max(
        1.0, minimum_norm_sq
    )
    if minimum_norm_sq > 1.0 + feasibility_tolerance:
        return None

    null = np.cross(matrix[0], matrix[1])
    null_magnitude = float(np.linalg.norm(null))
    if null_magnitude < EPS_GEOMETRIC:
        return None
    null /= null_magnitude
    branch_scale = np.sqrt(max(0.0, 1.0 - minimum_norm_sq))
    candidates = (
        minimum_norm + branch_scale * null,
        minimum_norm - branch_scale * null,
    )
    upward = [candidate for candidate in candidates if candidate[Axis.Z] > 0.0]
    if not upward:
        return None
    upward.sort(key=lambda candidate: float(candidate[Axis.Z]), reverse=True)
    if len(upward) > 1 and np.isclose(
        upward[0][Axis.Z],
        upward[1][Axis.Z],
        atol=_ORTHONORMAL_TOLERANCE,
        rtol=0.0,
    ):
        return None

    selected = upward[0]
    scale = max(1.0, abs(height), float(np.linalg.norm(matrix, ord=np.inf)))
    if not np.allclose(
        matrix @ selected,
        rhs,
        atol=_CONSTRAINT_RELATIVE_TOLERANCE * scale,
        rtol=0.0,
    ):
        return None
    return Direction3(selected)


def world_space_for_axle_state(
    axle: AxleSuspension,
    state: SuspensionState,
) -> WorldSpace | None:
    """Construct the flat, level WorldSpace for one accepted axle state."""
    pivot_pair = _opposite_axle_pivot(axle)
    if pivot_pair is None or axle.config is None:
        return None
    pivot, world_target = pivot_pair

    left = state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT))
    right = state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT))
    up = _solve_world_up(
        left,
        right,
        pivot,
        axle.config.opposite_axle_axis_height,
    )
    if up is None:
        return None

    chassis_forward = np.array((1.0, 0.0, 0.0))
    forward = chassis_forward - float(up[Axis.X]) * up.data
    forward_magnitude = float(np.linalg.norm(forward))
    if forward_magnitude < EPS_GEOMETRIC:
        return None
    forward /= forward_magnitude
    lateral = np.cross(up.data, forward)

    x = Direction3.from_trusted(forward)
    y = Direction3(lateral)
    z = up
    target_offset = (
        x * float(world_target[Axis.X])
        + y * float(world_target[Axis.Y])
        + z * float(world_target[Axis.Z])
    )
    origin = pivot - target_offset
    return WorldSpace(x=x, y=y, z=z, origin=origin)


def world_spaces_for_sweep(
    axle: AxleSuspension,
    states: Sequence[SuspensionState],
) -> list[WorldSpace | None]:
    """Construct one flat, level WorldSpace per accepted axle state."""
    return [world_space_for_axle_state(axle, state) for state in states]
