"""Post-solve placement in the supported ISO-aligned world frame.

Solver state remains entirely in chassis coordinates.  This module supplies a
presentation transform into world coordinates, following ISO 8855:2011
vehicle axes (§2.10) and earth-fixed axes (§2.8): vehicle X/Y/Z point
forward/left/up and earth-fixed Z points upwards, opposite gravity.

Suspension Explorer models only a straight, level road. The road and ground
planes therefore coincide with world ``Z = 0``. The axle's coupled wheel
contact centres define that plane in chassis coordinates. Contact closure
deliberately extrudes it parallel to chassis X, so this single-axle transform
represents local heave and roll but assigns zero pitch, yaw, and longitudinal
translation. Those unobservable whole-vehicle degrees of freedom are not
inferred from an opposite-axle proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from kinematics.core.enums import Axis, PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.road import RoadPlane

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension

__all__ = [
    "WorldSpace",
    "world_space_for_axle_state",
    "world_spaces_for_sweep",
]

_ORTHONORMAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class WorldSpace:
    """A rigid chassis-to-world transform for presentation.

    ``x``, ``y``, and ``z`` are world axes expressed as chassis-coordinate
    unit vectors and form the rows of :attr:`rotation_chassis_to_world`.
    ``origin`` is the chassis-coordinate point mapped to world ``(0, 0, 0)``.
    World Z is normal to the straight, level road and gravity is world ``-Z``.
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
        """Return world ``-Z`` expressed in chassis coordinates."""
        return -self.z

    @property
    def rotation_chassis_to_world(self) -> np.ndarray:
        """Return the matrix taking chassis-coordinate vectors to world."""
        return np.vstack((self.x.data, self.y.data, self.z.data))

    def to_world(self, point: Point3) -> Point3:
        """Map a chassis-coordinate point into world coordinates."""
        offset = (point - self.origin).data
        return Point3(self.rotation_chassis_to_world @ offset)


def world_space_for_axle_state(
    axle: AxleSuspension,
    state: SuspensionState,
) -> WorldSpace | None:
    """Construct the supported ISO earth-fixed frame for one axle state.

    The road plane comes from the same two output wheel contact centres as metric
    calculations.  Chassis +X remains world +X because the closure gives the
    plane no longitudinal gradient.  The origin is the intersection of the
    road plane with chassis ``X = 0`` and ``Y = 0``.

    Returns ``None`` when the authored design condition is not level or the
    current contact-centre pair cannot define the supported axle-local road plane.
    """
    if axle.design_road_plane is None:
        return None

    left = state.get(PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE))
    right = state.get(PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE))
    try:
        road = RoadPlane.from_axle_contact_centres(left, right)
    except ValueError:
        return None

    normal_z = float(road.normal[Axis.Z])
    if normal_z <= EPS_GEOMETRIC:
        return None

    x = Direction3((1.0, 0.0, 0.0))
    z = road.normal
    y = Direction3(z.cross(x))
    origin = Point3((0.0, 0.0, -road.offset_mm / normal_z))
    return WorldSpace(x=x, y=y, z=z, origin=origin)


def world_spaces_for_sweep(
    axle: AxleSuspension,
    states: Sequence[SuspensionState],
) -> list[WorldSpace | None]:
    """Construct one supported world transform per accepted axle state."""
    return [world_space_for_axle_state(axle, state) for state in states]
