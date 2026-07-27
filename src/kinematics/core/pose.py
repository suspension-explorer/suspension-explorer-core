"""World-space axis directions for solved states, in chassis space.

World space is this project's name for what ISO 8855 and SAE J670 call the
earth-fixed axis system: Z up along the vertical, opposing gravity. The
standards distinguish it from the vehicle axis system (fixed to the sprung
mass; our chassis space). Suspension kinematics measures the chassis-to-ground
relation completely — the fitted
:class:`~kinematics.core.metrics.ground.GroundDatum` is exactly that — but no
single-axle state contains any information about where gravity points. World
space therefore cannot be derived; it must be supplied.

This module reports world space as its axis directions expressed in chassis
space, per solved state. Callers either pass an explicit chassis-space gravity
direction (from telemetry, a rig, or a full-vehicle model) or select a named
:class:`GravityModel` — a rule that computes gravity from the solved ground
datum under stated premises. Reporting basis vectors rather than composed
roll/pitch angles keeps the axis system exact — there is no rotation-order
convention and no small-angle approximation in the result; any approximation
lives entirely in the gravity vector and is recorded as its model.

Axis-system construction from a gravity direction is conventional:

- ``z`` (world up) is the negated gravity direction.
- ``x`` (world forward) is chassis forward projected into the
  world-horizontal plane, so heading is measured from the vehicle's own
  forward axis. The axis system is undefined when chassis forward is
  vertical in the world sense.
- ``y`` completes the right-handed triad.

Vocabulary, kept deliberately distinct:

- Suspension roll is the axle metric ``roll``, from wheel-centre travel.
- The chassis-to-ground angle is the axle metric ``ground_line_angle``.
- Chassis-to-world attitude is this module's output, and it is only as good
  as the gravity supplied. Under :attr:`GravityModel.ROAD_LEVEL` the ground
  datum is world-level, so the chassis-to-world attitude includes the full
  ground angle; under :attr:`GravityModel.CHASSIS_LEVEL` the chassis is
  world-level (the rig interpretation), so the same ground angle reads as
  road bank instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, cos, isfinite, sin
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from kinematics.core.enums import Axis, AxlePosition, PointID
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointRef, Side

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension

__all__ = [
    "GravityModel",
    "WorldSpace",
    "world_space_for_axle_state",
    "world_spaces_for_sweep",
]

_ORTHONORMAL_TOLERANCE = 1e-9


class GravityModel(StrEnum):
    """A rule for computing gravity from a solved state, under stated premises.

    One axle cannot measure gravity, so deriving world space from kinematics
    alone requires a model. Each member names its premises and the rule they
    justify:

    - ``ROAD_LEVEL`` — premise: the fitted ground plane is perpendicular to
      gravity. Rule: gravity is the negated ground-plane normal. Any
      ground-line angle then reads as chassis attitude over a level road.
    - ``OPPOSITE_AXLE_FIXED`` — premises: the road is laterally level, and the
      unmodelled axle's contact line stays fixed on it, so ride-height change
      against the design datum tips the chassis about that line. Rule: rotate
      the road-level gravity fore-aft by ``atan(compression / wheelbase)``,
      signed by the modelled axle's front/rear position.
    - ``CHASSIS_LEVEL`` — premise: the chassis vertical is aligned with
      gravity (the kinematics-rig reading). Rule: gravity is the negated
      chassis vertical, and any ground-line angle reads as road bank under a
      level vehicle.
    """

    ROAD_LEVEL = "road_level"
    OPPOSITE_AXLE_FIXED = "opposite_axle_fixed"
    CHASSIS_LEVEL = "chassis_level"


@dataclass(frozen=True)
class WorldSpace:
    """World-space axis directions expressed in chassis space.

    ``x``, ``y``, and ``z`` are the world axis directions (ISO 8855's
    earth-fixed axis system) as chassis-space unit vectors; together they are
    the rows of the chassis-to-world rotation. ``anchor`` is the chassis-space
    point that maps to the world origin: the modelled axle's ground-line
    centreline point. ``gravity_model`` records which model computed the
    gravity direction, or ``None`` when the caller supplied gravity
    explicitly, so a modelled attitude can never be mistaken for a measured
    one.
    """

    x: Direction3
    y: Direction3
    z: Direction3
    anchor: Point3
    gravity_model: GravityModel | None

    def __post_init__(self) -> None:
        """Require a right-handed orthonormal triad."""
        axes = np.vstack((self.x.data, self.y.data, self.z.data))
        orthonormal = np.allclose(axes @ axes.T, np.eye(3), atol=_ORTHONORMAL_TOLERANCE)
        if not orthonormal or float(np.linalg.det(axes)) < 0.0:
            raise ValueError(
                "World-space axes must form a right-handed orthonormal triad"
            )

    @classmethod
    def from_gravity(
        cls,
        gravity: Direction3 | Sequence[float] | np.ndarray,
        *,
        anchor: Point3,
        gravity_model: GravityModel | None = None,
    ) -> WorldSpace | None:
        """Build the axis system implied by a chassis-space gravity direction.

        World up opposes gravity; world forward is chassis forward projected
        into the world-horizontal plane. Returns ``None`` when the gravity
        vector is degenerate or chassis forward is vertical in the world
        sense, leaving heading undefined.
        """
        raw = np.asarray(
            gravity.data if isinstance(gravity, Direction3) else gravity,
            dtype=np.float64,
        )
        if raw.shape != (3,) or not np.isfinite(raw).all():
            return None
        magnitude = float(np.linalg.norm(raw))
        if magnitude < EPS_GEOMETRIC:
            return None
        up = -raw / magnitude

        forward = np.array((1.0, 0.0, 0.0)) - float(up[0]) * up
        forward_magnitude = float(np.linalg.norm(forward))
        if forward_magnitude < EPS_GEOMETRIC:
            return None
        forward /= forward_magnitude

        lateral = np.cross(up, forward)
        return cls(
            x=Direction3.from_trusted(forward),
            y=Direction3.from_trusted(lateral),
            z=Direction3.from_trusted(up),
            anchor=anchor,
            gravity_model=gravity_model,
        )

    @property
    def gravity(self) -> Direction3:
        """Return the chassis-space gravity direction the axis system encodes."""
        return Direction3.from_trusted(-self.z.data)

    @property
    def rotation_chassis_to_world(self) -> np.ndarray:
        """Return the 3x3 rotation taking chassis vectors to world vectors."""
        return np.vstack((self.x.data, self.y.data, self.z.data))

    def to_world(self, point: Point3) -> Point3:
        """Map a chassis-space point into world coordinates."""
        offset = (point - self.anchor).data
        return Point3(self.rotation_chassis_to_world @ offset)


def _modelled_gravity(
    axle: AxleSuspension,
    ground: GroundDatum,
    model: GravityModel,
) -> np.ndarray | None:
    """Compute the chassis-space gravity vector one model implies."""
    if model is GravityModel.CHASSIS_LEVEL:
        return np.array((0.0, 0.0, -1.0))

    road_gravity = -ground.normal.data

    if model is GravityModel.ROAD_LEVEL:
        return road_gravity

    # OPPOSITE_AXLE_FIXED: the road is laterally level, and ride-height
    # change against the design datum tips gravity fore-aft about the
    # unmodelled axle's contact line. Rotating gravity by the pitch angle
    # about the ground-line lateral axis reproduces the level-road case
    # exactly when the datum is level.
    design_ground = axle.design_ground
    config = axle.corners[Side.LEFT].config
    if design_ground is None or config is None or config.axle_position is None:
        return None
    current_z = ground.z_at(0.0)
    design_z = design_ground.z_at(0.0)
    if current_z is None or design_z is None:
        return None
    wheelbase = config.wheelbase
    if not isfinite(wheelbase) or wheelbase <= 0.0:
        return None

    # A compressing front axle drops the nose (positive, nose-down pitch); a
    # compressing rear axle raises it.
    compression = current_z - design_z
    sign = 1.0 if config.axle_position is AxlePosition.FRONT else -1.0
    pitch = sign * atan2(compression, wheelbase)
    lateral = np.array((0.0, ground.tangent_y, ground.tangent_z))
    return cos(pitch) * road_gravity + sin(pitch) * np.cross(road_gravity, lateral)


def world_space_for_axle_state(
    axle: AxleSuspension,
    state: SuspensionState,
    gravity: GravityModel | str | Direction3 | Sequence[float] | np.ndarray,
) -> WorldSpace | None:
    """Report the world-space axes for one solved axle state.

    This is a post-solve reading of solved outputs only; nothing here feeds
    back into the kinematic solve. ``gravity`` is either an explicit
    chassis-space gravity direction or a :class:`GravityModel` (string values
    coerce; unrecognised strings raise ``ValueError``).

    Returns ``None`` when the fitted ground datum needed for the anchor (or
    for a datum-based gravity model) is undefined, or the axis system itself
    is degenerate.
    """
    ground = GroundDatum.from_wheel_ground_tangents(
        state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)),
        state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT)),
    )
    if ground is None:
        return None
    anchor_z = ground.z_at(0.0)
    if anchor_z is None:
        return None

    left_center = state.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))
    right_center = state.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))
    axle_x = 0.5 * (float(left_center[Axis.X]) + float(right_center[Axis.X]))
    if not isfinite(axle_x):
        return None
    anchor = Point3((axle_x, 0.0, anchor_z))

    if isinstance(gravity, (GravityModel, str)):
        model = GravityModel(gravity)
        vector = _modelled_gravity(axle, ground, model)
        if vector is None:
            return None
        return WorldSpace.from_gravity(vector, anchor=anchor, gravity_model=model)
    return WorldSpace.from_gravity(gravity, anchor=anchor, gravity_model=None)


def world_spaces_for_sweep(
    axle: AxleSuspension,
    states: Sequence[SuspensionState],
    gravity: (
        GravityModel
        | str
        | Direction3
        | Sequence[float]
        | np.ndarray
        | Sequence[Direction3]
    ),
) -> list[WorldSpace | None]:
    """Report the world-space axes for every state of a solved sweep.

    ``gravity`` is a single model or direction applied to every state, or a
    sequence of per-state :class:`Direction3` gravity directions with one
    entry per state (for example from telemetry sampled along the sweep).
    """
    if (
        isinstance(gravity, Sequence)
        and not isinstance(gravity, str)
        and any(isinstance(item, Direction3) for item in gravity)
    ):
        directions = [item for item in gravity if isinstance(item, Direction3)]
        if len(directions) != len(gravity):
            raise ValueError(
                "Per-state gravity entries must all be Direction3 instances."
            )
        if len(directions) != len(states):
            raise ValueError(
                "Per-state gravity requires one direction per state: "
                f"{len(directions)} directions, {len(states)} states."
            )
        return [
            world_space_for_axle_state(axle, state, direction)
            for state, direction in zip(states, directions, strict=True)
        ]
    shared = cast(
        "GravityModel | str | Direction3 | Sequence[float] | np.ndarray",
        gravity,
    )
    return [world_space_for_axle_state(axle, state, shared) for state in states]
