"""Internal chassis-space representation of the current World ground plane."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3

_INVARIANT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class GroundDatum:
    """An upward unit normal and Hessian offset in chassis coordinates.

    The plane equation is ``normal · point + offset_mm = 0``.  This type is an
    internal metric primitive: axle callers derive it from the completed
    :class:`~kinematics.core.pose.WorldSpace`, while standalone corners use a
    horizontal plane through their tyre contact.
    """

    normal: Direction3
    offset_mm: float

    def __post_init__(self) -> None:
        """Reject non-finite or downward-oriented plane data."""
        if not np.isfinite(self.normal.data).all() or not isfinite(self.offset_mm):
            raise ValueError("Ground-datum fields must be finite")
        if abs(float(np.linalg.norm(self.normal.data)) - 1.0) > _INVARIANT_TOLERANCE:
            raise ValueError("Ground-datum normal must be unit length")
        if float(self.normal[2]) < -_INVARIANT_TOLERANCE:
            raise ValueError("Ground-datum normal must use the upward orientation")

    @classmethod
    def through(cls, normal: Direction3, point: Point3) -> GroundDatum:
        """Construct a plane with ``normal`` passing through ``point``."""
        return cls(
            normal=normal,
            offset_mm=-float(np.dot(normal.data, point.data)),
        )

    @classmethod
    def horizontal_at(cls, point: Point3) -> GroundDatum:
        """Construct the standalone-corner +Z datum through ``point``."""
        return cls.through(Direction3((0.0, 0.0, 1.0)), point)

    @property
    def forward(self) -> Direction3:
        """Return chassis-forward projected into the ground plane."""
        chassis_forward = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        projected = chassis_forward - float(self.normal[0]) * self.normal.data
        magnitude = float(np.linalg.norm(projected))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("Ground plane has no resolvable forward direction")
        return Direction3.from_trusted(projected / magnitude)

    @property
    def lateral(self) -> Direction3:
        """Return the ground-plane lateral direction toward vehicle left."""
        return Direction3(self.normal.cross(self.forward))

    def signed_distance(self, point: Point3) -> float:
        """Return signed perpendicular distance; positive is above the plane."""
        return float(np.dot(self.normal.data, point.data) + self.offset_mm)
