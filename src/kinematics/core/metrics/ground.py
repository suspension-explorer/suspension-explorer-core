"""Axle-level ground datum derived from two wheel-ground tangent points.

Chassis space remains the coordinate reference. This primitive
describes only the YZ ground line at one axle; its corresponding 3D plane is
that line extruded along chassis X, so it deliberately carries no grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

from kinematics.core.enums import Axis
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3

_INVARIANT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AxleGroundLine:
    """A normalized YZ tangent line and its upward Hessian normal.

    The line equation is ``normal_y * y + normal_z * z + offset_mm = 0``.
    Its tangent convention points from vehicle right to left (nonnegative Y),
    which makes the normal's Z component nonnegative.  Extruding the line in
    chassis ±X yields the axle-level, zero-grade ground plane.
    """

    tangent_y: float
    tangent_z: float
    normal_y: float
    normal_z: float
    offset_mm: float

    def __post_init__(self) -> None:
        """Reject direct construction that would violate the line conventions."""
        if not all(
            isfinite(value)
            for value in (
                self.tangent_y,
                self.tangent_z,
                self.normal_y,
                self.normal_z,
                self.offset_mm,
            )
        ):
            raise ValueError("Axle ground-line fields must be finite")

        if abs(hypot(self.tangent_y, self.tangent_z) - 1.0) > _INVARIANT_TOLERANCE:
            raise ValueError("Axle ground-line tangent must be unit length")

        # Pinning the normal to the tangent's upward rotation also fixes its
        # length and perpendicularity, so no separate checks are needed.
        if (
            abs(self.normal_y + self.tangent_z) > _INVARIANT_TOLERANCE
            or abs(self.normal_z - self.tangent_y) > _INVARIANT_TOLERANCE
        ):
            raise ValueError(
                "Axle ground-line normal must be the tangent's upward rotation"
            )

        if self.tangent_y < -_INVARIANT_TOLERANCE or (
            abs(self.tangent_y) <= _INVARIANT_TOLERANCE
            and self.tangent_z < -_INVARIANT_TOLERANCE
        ):
            raise ValueError("Axle ground-line tangent has a non-canonical orientation")

    @classmethod
    def from_wheel_ground_tangents(
        cls, left: Point3, right: Point3
    ) -> AxleGroundLine | None:
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

        # The guards above leave a strictly positive lateral component and a
        # finite separation, so normalization yields a canonically oriented unit
        # tangent and its upward rotation.
        tangent_y /= separation
        tangent_z /= separation
        normal_y = -tangent_z
        normal_z = tangent_y
        offset_mm = -(normal_y * left_y + normal_z * left_z)
        if not isfinite(offset_mm):
            return None

        return cls(
            tangent_y=tangent_y,
            tangent_z=tangent_z,
            normal_y=normal_y,
            normal_z=normal_z,
            offset_mm=offset_mm,
        )

    @property
    def angle_deg(self) -> float:
        """Return the signed ground-line roll angle, positive toward vehicle left."""
        return degrees(atan2(self.tangent_z, self.tangent_y))

    @property
    def plane_normal(self) -> tuple[float, float, float]:
        """Return the normal of the X-extruded axle ground plane."""
        return (0.0, self.normal_y, self.normal_z)

    def z_at(self, y: float) -> float | None:
        """Return the line height at lateral coordinate ``y``, when defined."""
        if not isfinite(y) or abs(self.normal_z) < EPS_GEOMETRIC:
            return None
        z = -(self.normal_y * y + self.offset_mm) / self.normal_z
        return float(z) if isfinite(z) else None

    def signed_distance_yz(self, point: Point3) -> float:
        """Return signed perpendicular YZ distance; positive is above the line."""
        return float(
            self.normal_y * point[Axis.Y]
            + self.normal_z * point[Axis.Z]
            + self.offset_mm
        )
