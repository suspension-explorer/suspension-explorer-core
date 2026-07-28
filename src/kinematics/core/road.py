"""Road-plane geometry shared by contact closure, metrics, and presentation.

ISO 8855:2011 distinguishes the earth-fixed ``ground plane`` (§2.5) from the
local or equivalent ``road plane`` (§2.7) at a tyre contact.  Suspension
Explorer supports only a straight, level road, so those planes coincide in
world coordinates.
During a solve, however, points remain in chassis coordinates and the road
plane moves relative to that basis as the modelled axle heaves or rolls.

The axle contact closure is deliberately longitudinally invariant: its road
plane is extruded parallel to chassis X.  A single axle can therefore resolve
local heave and roll, but it cannot infer whole-vehicle pitch or yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

from kinematics.core.enums import Axis
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3

_INVARIANT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RoadPlane:
    """An ISO-style road plane represented in chassis coordinates.

    The Hessian plane equation is ``normal · point + offset_mm = 0``.
    ``normal`` points away from the road.  The representation is general,
    while :meth:`from_axle_tangents` records this application's single-axle
    convention by forcing the longitudinal normal component to zero.
    """

    normal: Direction3
    offset_mm: float

    def __post_init__(self) -> None:
        """Require finite, unit-length, upward-oriented plane data."""
        if not np.isfinite(self.normal.data).all() or not isfinite(self.offset_mm):
            raise ValueError("Road-plane fields must be finite")
        if abs(float(np.linalg.norm(self.normal.data)) - 1.0) > _INVARIANT_TOLERANCE:
            raise ValueError("Road-plane normal must be unit length")
        if float(self.normal[Axis.Z]) < -_INVARIANT_TOLERANCE:
            raise ValueError("Road-plane normal must use the upward orientation")

    @classmethod
    def through(cls, normal: Direction3, point: Point3) -> RoadPlane:
        """Construct a chassis-coordinate road plane through ``point``."""
        return cls(
            normal=normal,
            offset_mm=-float(np.dot(normal.data, point.data)),
        )

    @classmethod
    def horizontal_at(cls, point: Point3) -> RoadPlane:
        """Construct a chassis-horizontal road plane through ``point``."""
        return cls.through(Direction3((0.0, 0.0, 1.0)), point)

    @classmethod
    def from_axle_tangents(cls, left: Point3, right: Point3) -> RoadPlane:
        """Construct the axle-local road plane through both tyre tangents.

        The result is the unique upward plane through the two points whose
        normal has zero chassis-X component.  This is the same longitudinally
        extruded plane used by the coupled tyre-contact closure.  It supports
        axle heave and roll without pretending that one axle determines
        whole-vehicle pitch.

        Raises:
            ValueError: If the tangents do not define a usable lateral line.
        """
        dy = float(left[Axis.Y]) - float(right[Axis.Y])
        dz = float(left[Axis.Z]) - float(right[Axis.Z])
        magnitude = hypot(dy, dz)
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("Axle tyre tangents do not define a road plane")

        normal_y = -dz / magnitude
        normal_z = dy / magnitude
        if normal_z < 0.0:
            normal_y = -normal_y
            normal_z = -normal_z
        if normal_z < EPS_GEOMETRIC:
            raise ValueError("Axle tyre tangents imply a vertical road plane")

        return cls.through(Direction3((0.0, normal_y, normal_z)), left)

    @property
    def forward(self) -> Direction3:
        """Return ISO vehicle forward projected into the road plane."""
        chassis_forward = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        projected = chassis_forward - float(self.normal[Axis.X]) * self.normal.data
        magnitude = float(np.linalg.norm(projected))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("Road plane has no resolvable forward direction")
        return Direction3.from_trusted(projected / magnitude)

    @property
    def lateral(self) -> Direction3:
        """Return the in-plane lateral direction toward vehicle left."""
        return Direction3(self.normal.cross(self.forward))

    def signed_distance(self, point: Point3) -> float:
        """Return perpendicular distance, positive on the vehicle side."""
        return float(np.dot(self.normal.data, point.data) + self.offset_mm)
