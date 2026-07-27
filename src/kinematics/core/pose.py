"""Post-solve chassis pose relative to a world (road) frame.

Everything the solver produces lives in chassis space: ``+X`` forward, ``+Y``
left, ``+Z`` up, with fixed hardpoints stationary. The road, however, is only
known through the axle :class:`~kinematics.core.metrics.ground.GroundDatum`
fitted to the solved state. This module reinterprets that datum as a rigid
chassis pose relative to a world frame in which the road is the ``Z = 0``
plane and gravity points along ``-Z``.

A single modelled axle cannot know the whole vehicle's attitude, so the pitch
component of the pose is an *assumption*, chosen explicitly by the caller:

- :attr:`PoseAssumption.PURE_HEAVE` treats any ride-height change as whole-car
  heave. Pitch is zero.
- :attr:`PoseAssumption.OPPOSITE_AXLE_FIXED` treats the unmodelled axle's
  contact line as fixed on the road, so the modelled axle's ride-height change
  tips the chassis about it: ``pitch = atan(compression / wheelbase)``.

Roll and ground clearance need no assumption; they come directly from the
fitted ground datum. All angles are small-angle vehicle attitudes: the pose is
built by composing pitch about chassis ``+Y`` after roll about chassis ``+X``,
and the neglected cross terms (including the contact point migrating around
the tire as the chassis pitches) are second order in the angles.

Sign conventions (right-hand rule about chassis axes):

- Roll is positive when the left side of the chassis rises relative to the
  road (rotation about ``+X``).
- Pitch is positive nose-down (rotation about ``+Y``).

The world frame is anchored at the modelled axle: its origin is the chassis
ground-line centreline point ``(axle_x, 0, z_ground(0))``, its ``Z = 0`` plane
is the road, and its axes are the chassis axes de-rolled and de-pitched.
Because the fitted ground datum deliberately carries no longitudinal grade,
mapping it through a pitched pose leaves a residual grade of exactly the pitch
angle; this is the documented zero-grade approximation, not an error to be
iterated away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, cos, degrees, isfinite, radians, sin
from typing import TYPE_CHECKING

import numpy as np

from kinematics.core.enums import Axis, AxlePosition, PointID
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointRef, Side

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension

__all__ = [
    "ChassisPose",
    "PoseAssumption",
    "build_chassis_pose",
    "chassis_pose_for_axle_state",
]


class PoseAssumption(StrEnum):
    """Named interpretation of a single-axle state as a whole-car attitude."""

    PURE_HEAVE = "pure_heave"
    OPPOSITE_AXLE_FIXED = "opposite_axle_fixed"


@dataclass(frozen=True)
class ChassisPose:
    """Rigid chassis-to-world pose implied by a fitted axle ground datum.

    ``anchor`` is the chassis-space point that maps to the world origin: the
    modelled axle's ground-line centreline point. ``assumption`` records how
    pitch was obtained so consumers cannot mistake an interpreted attitude
    for a measured one.
    """

    roll_deg: float
    pitch_deg: float
    anchor: Point3
    assumption: PoseAssumption

    def __post_init__(self) -> None:
        """Reject non-finite pose components."""
        if not (isfinite(self.roll_deg) and isfinite(self.pitch_deg)):
            raise ValueError("Chassis pose angles must be finite")

    @property
    def rotation_chassis_to_world(self) -> np.ndarray:
        """Return the 3x3 rotation taking chassis vectors to world vectors.

        Composed as pitch about ``+Y`` after roll about ``+X``. The two
        rotations commute to first order in the (small) attitude angles.
        """
        roll = radians(self.roll_deg)
        pitch = radians(self.pitch_deg)
        roll_matrix = np.array(
            (
                (1.0, 0.0, 0.0),
                (0.0, cos(roll), -sin(roll)),
                (0.0, sin(roll), cos(roll)),
            )
        )
        pitch_matrix = np.array(
            (
                (cos(pitch), 0.0, sin(pitch)),
                (0.0, 1.0, 0.0),
                (-sin(pitch), 0.0, cos(pitch)),
            )
        )
        return pitch_matrix @ roll_matrix

    def to_world(self, point: Point3) -> Point3:
        """Map a chassis-space point into the world (road) frame."""
        rotated = self.rotation_chassis_to_world @ (point - self.anchor).data
        return Point3(rotated)

    def gravity_direction_chassis(self) -> Direction3:
        """Return the unit gravity direction ('down') expressed in chassis space."""
        down_world = np.array((0.0, 0.0, -1.0))
        return Direction3(self.rotation_chassis_to_world.T @ down_world)


def build_chassis_pose(
    ground: GroundDatum,
    design_ground: GroundDatum,
    *,
    wheelbase: float,
    axle_position: AxlePosition,
    assumption: PoseAssumption,
    axle_x: float,
) -> ChassisPose | None:
    """Build the chassis pose implied by a solved axle ground datum.

    Args:
        ground: Fitted ground datum for the solved state.
        design_ground: Fitted ground datum for the design state; the pitch
            reference. Ride-height change is measured between the two at the
            chassis centreline.
        wheelbase: Distance between axle lines in millimetres.
        axle_position: Which end of the vehicle the modelled axle is; sets the
            pitch sign under :attr:`PoseAssumption.OPPOSITE_AXLE_FIXED`.
        assumption: How to interpret ride-height change as pitch.
        axle_x: Chassis-space X station of the modelled axle (typically the
            mean wheel-centre X); locates the world-frame anchor.

    Returns:
        The pose, or ``None`` when either datum cannot evaluate the
        centreline height or the wheelbase is degenerate.
    """
    if not isfinite(axle_x):
        return None
    current_z = ground.z_at(0.0)
    design_z = design_ground.z_at(0.0)
    if current_z is None or design_z is None:
        return None

    roll_deg = -ground.angle_deg

    if assumption is PoseAssumption.PURE_HEAVE:
        pitch_deg = 0.0
    else:
        if not isfinite(wheelbase) or wheelbase <= 0.0:
            return None
        # Positive compression moves the ground line up in chassis space.
        compression = current_z - design_z
        # A compressing front axle drops the nose (positive, nose-down pitch);
        # a compressing rear axle raises it.
        sign = 1.0 if axle_position is AxlePosition.FRONT else -1.0
        pitch_deg = sign * degrees(atan2(compression, wheelbase))

    return ChassisPose(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        anchor=Point3((axle_x, 0.0, current_z)),
        assumption=assumption,
    )


def chassis_pose_for_axle_state(
    axle: AxleSuspension,
    state: SuspensionState,
    assumption: PoseAssumption,
) -> ChassisPose | None:
    """Reinterpret one solved axle state as a whole-vehicle attitude.

    This is the post-solve, assumption-labelled reading of a single-axle state:
    it is computed purely from solved outputs (the two stored wheel-ground
    tangents, the axle's design ground datum, and the vehicle configuration)
    and is never part of the kinematic solve. Nothing here feeds back into the
    constraint system, and the returned pose carries ``assumption`` so a
    consumer cannot mistake an interpreted pitch for a measured one.

    Args:
        axle: The composed axle; supplies the design ground datum and the
            corner configuration holding wheelbase and axle position.
        state: A solved axle state carrying both wheel-ground tangents.
        assumption: How to interpret ride-height change as pitch.

    Returns:
        The chassis pose, or ``None`` when the axle carries no configuration
        or axle position, or when either ground datum is undefined.
    """
    ground = GroundDatum.from_wheel_ground_tangents(
        state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)),
        state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT)),
    )
    design_ground = axle.design_ground
    if ground is None or design_ground is None:
        return None

    # Both corners share one vehicle configuration; the left is representative.
    config = axle.corners[Side.LEFT].config
    if config is None or config.axle_position is None:
        return None

    left_center = state.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER))
    right_center = state.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER))
    axle_x = 0.5 * (float(left_center[Axis.X]) + float(right_center[Axis.X]))

    return build_chassis_pose(
        ground,
        design_ground,
        wheelbase=config.wheelbase,
        axle_position=config.axle_position,
        assumption=assumption,
        axle_x=axle_x,
    )
