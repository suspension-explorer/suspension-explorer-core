"""Typed zero-grade ground datum for metric calculations.

Chassis space remains the coordinate reference. The datum stores the upward
normal and Hessian offset of a YZ ground line; its corresponding 3D plane is
that line extruded along chassis X, so it deliberately carries no grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

from kinematics.core.enums import Axis
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3

_INVARIANT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class GroundDatum:
    """A normalized upward YZ normal and Hessian offset.

    The line equation is ``normal_y * y + normal_z * z + offset_mm = 0``.
    Its derived tangent points from vehicle right to left (nonnegative Y).
    Extruding the line in chassis ±X yields the zero-grade ground plane used
    by metric calculations.
    """

    normal_y: float
    normal_z: float
    offset_mm: float

    def __post_init__(self) -> None:
        """Reject direct construction that violates the datum conventions."""
        if not all(
            isfinite(value) for value in (self.normal_y, self.normal_z, self.offset_mm)
        ):
            raise ValueError("Ground-datum fields must be finite")
        if abs(hypot(self.normal_y, self.normal_z) - 1.0) > _INVARIANT_TOLERANCE:
            raise ValueError("Ground-datum normal must be unit length")
        if self.normal_z < -_INVARIANT_TOLERANCE or (
            abs(self.normal_z) <= _INVARIANT_TOLERANCE
            and self.normal_y > _INVARIANT_TOLERANCE
        ):
            raise ValueError("Ground-datum normal must use the upward orientation")

    @classmethod
    def horizontal_at(cls, point: Point3) -> GroundDatum:
        """Construct the standalone-corner +Z datum through ``point``."""
        return cls(
            normal_y=0.0,
            normal_z=1.0,
            offset_mm=-float(point[Axis.Z]),
        )

    @classmethod
    def from_wheel_ground_tangents(
        cls, left: Point3, right: Point3
    ) -> GroundDatum | None:
        """Construct the datum from current wheel-ground tangents, ignoring X.

        Returns ``None`` when the lateral track or YZ projection has
        insufficient separation, or any input/intermediate is non-finite.
        Degenerate input is rejected by the guards below; anything that reaches
        construction satisfies the line invariants by construction, so a
        :class:`ValueError` from ``__post_init__`` signals a defect here rather
        than an undefined ground line.
        """
        left_y = left[Axis.Y]
        left_z = left[Axis.Z]
        right_y = right[Axis.Y]
        right_z = right[Axis.Z]
        if not all(isfinite(value) for value in (left_y, left_z, right_y, right_z)):
            return None

        tangent_y = left_y - right_y
        tangent_z = left_z - right_z
        if abs(tangent_y) <= EPS_GEOMETRIC:
            return None
        if tangent_y < 0.0:
            tangent_y = -tangent_y
            tangent_z = -tangent_z

        separation = hypot(tangent_y, tangent_z)
        if not isfinite(separation) or separation <= EPS_GEOMETRIC:
            return None

        # The guards above leave a strictly positive lateral component and
        # finite separation, so the upward normal is canonical.
        tangent_y /= separation
        tangent_z /= separation
        normal_y = -tangent_z
        normal_z = tangent_y
        offset_mm = -(normal_y * left_y + normal_z * left_z)
        if not isfinite(offset_mm):
            return None

        return cls(
            normal_y=normal_y,
            normal_z=normal_z,
            offset_mm=offset_mm,
        )

    @property
    def tangent_y(self) -> float:
        """Return the canonical ground-line tangent's lateral component."""
        return self.normal_z

    @property
    def tangent_z(self) -> float:
        """Return the canonical ground-line tangent's vertical component."""
        return -self.normal_y

    @property
    def angle_deg(self) -> float:
        """Return the signed ground-line roll angle, positive toward vehicle left."""
        return degrees(atan2(self.tangent_z, self.tangent_y))

    @property
    def normal(self) -> Direction3:
        """Return the upward normal of the X-extruded ground plane."""
        return Direction3((0.0, self.normal_y, self.normal_z))

    def z_at(self, y: float) -> float | None:
        """Return the line height at lateral coordinate ``y``, when defined."""
        if not isfinite(y) or abs(self.normal_z) < EPS_GEOMETRIC:
            return None
        z = -(self.normal_y * y + self.offset_mm) / self.normal_z
        return float(z) if isfinite(z) else None

    def signed_distance(self, point: Point3) -> float:
        """Return signed perpendicular distance; positive is above the plane."""
        return float(
            self.normal_y * point[Axis.Y]
            + self.normal_z * point[Axis.Z]
            + self.offset_mm
        )
