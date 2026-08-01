"""Steering-axis representation and source-independent steering metrics.

Physical pivots and motion-derived fits establish the same :class:`SteeringAxis`
line.  Once established, both sources use the calculations in this module; the
source only determines which axis is placed in the metric context.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np

from kinematics.core.enums import Axis
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3, Vector3
from kinematics.core.road import RoadPlane


@dataclass(frozen=True)
class SteeringAxis:
    """An immutable, oriented chassis-space steering-axis line.

    Axis-establishing code owns the direction convention: physical pivots use
    lower-to-upper, while an unoriented analytical line can be canonicalized by
    :meth:`from_unoriented_line`.
    """

    point: Point3
    direction: Direction3

    def __post_init__(self) -> None:
        """Validate, normalize, and freeze owned geometry copies."""
        point_data = np.array(self.point.data, dtype=np.float64)
        direction_data = np.array(self.direction.data, dtype=np.float64)
        if not (np.isfinite(point_data).all() and np.isfinite(direction_data).all()):
            raise ValueError("SteeringAxis point and direction must be finite")
        magnitude = float(np.linalg.norm(direction_data))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("SteeringAxis direction must be non-zero")
        normalized = direction_data / magnitude

        owned_point = Point3(point_data)
        owned_direction = Direction3.from_trusted(normalized)
        owned_point.data.setflags(write=False)
        owned_direction.data.setflags(write=False)
        object.__setattr__(self, "point", owned_point)
        object.__setattr__(self, "direction", owned_direction)

    @classmethod
    def from_point_direction(
        cls,
        point: Point3,
        direction: Vector3 | Direction3,
    ) -> SteeringAxis:
        """Construct an oriented axis from any point and non-zero direction."""
        raw_direction = direction.data
        if not np.isfinite(raw_direction).all():
            raise ValueError("SteeringAxis point and direction must be finite")
        magnitude = float(np.linalg.norm(raw_direction))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("SteeringAxis direction must be non-zero")
        return cls(
            point=point,
            direction=Direction3.from_trusted(raw_direction / magnitude),
        )

    @classmethod
    def from_unoriented_line(
        cls,
        point: Point3,
        direction: Vector3 | Direction3,
    ) -> SteeringAxis:
        """Construct an axis from a line whose input direction has no meaning.

        The sign is chosen chassis-up when possible, with deterministic X/Y
        tie-breakers for a horizontal line. This suits screw-axis fits, whose
        line direction may be reversed without changing their geometry.
        """
        axis = cls.from_point_direction(point, direction)
        canonical = axis.direction
        for component in (Axis.Z, Axis.X, Axis.Y):
            if abs(canonical[component]) >= EPS_GEOMETRIC:
                if canonical[component] < 0.0:
                    canonical = -canonical
                break
        return cls.from_point_direction(axis.point, canonical)

    @classmethod
    def from_pivots(
        cls,
        lower_pivot: Point3,
        upper_pivot: Point3,
    ) -> SteeringAxis:
        """Construct a lower-to-upper axis through physical steering pivots."""
        return cls.from_point_direction(lower_pivot, upper_pivot - lower_pivot)

    def intersect_road(self, road: RoadPlane) -> Point3 | None:
        """Return the finite intersection with ``road``, or ``None`` if parallel."""
        denominator = road.normal.dot(self.direction)
        if abs(denominator) < EPS_GEOMETRIC:
            return None
        parameter = -road.signed_distance(self.point) / denominator
        intersection = self.point.data + parameter * self.direction.data
        return Point3(intersection) if np.isfinite(intersection).all() else None


def calculate_caster(axis: SteeringAxis) -> float | None:
    """Return chassis-relative caster in degrees for ``axis``.

    This is ISO 8855:2011 castor angle (§7.2.2), exposed under the conventional
    project spelling ``caster``. The axis is resolved in the chassis XZ plane
    against chassis +Z; neither the road plane nor world vertical is involved.
    Positive caster means the established upward axis tilts rearward. A
    horizontal line has no resolvable lower-to-upper orientation and returns
    ``None``.
    """
    if abs(axis.direction[Axis.Z]) < EPS_GEOMETRIC:
        return None
    return degrees(atan2(-axis.direction[Axis.X], axis.direction[Axis.Z]))


def calculate_kpi(axis: SteeringAxis, side_sign: float) -> float | None:
    """Return inward-positive chassis-relative KPI for ``axis``.

    This is ISO 8855:2011 steering-axis inclination (§7.2.5), resolved in the
    chassis YZ plane against chassis +Z. Positive KPI means the top of the axis
    tilts inward toward the vehicle centreline.

    ``side_sign`` is ``+1`` for left and ``-1`` for right. ``None`` marks an
    unspecified vehicle side rather than inventing an inward direction.
    """
    if abs(axis.direction[Axis.Z]) < EPS_GEOMETRIC or abs(side_sign) < EPS_GEOMETRIC:
        return None
    return degrees(atan2(-side_sign * axis.direction[Axis.Y], axis.direction[Axis.Z]))


def calculate_scrub_radius(
    axis: SteeringAxis,
    road: RoadPlane,
    contact_centre: Point3,
) -> float | None:
    """Return unsigned road-plane scrub radius for ``axis``.

    Following ISO 8855:2011 scrub radius (§7.2.10), this is the distance in the
    local road plane from the tyre contact centre to the steering-axis
    intersection. It is not the signed lateral steering-axis offset. ``None``
    indicates that the axis does not intersect the road plane.
    """
    displacement = _road_displacement(axis, road, contact_centre)
    return displacement.norm() if displacement is not None else None


def calculate_steering_axis_offset_at_ground(
    axis: SteeringAxis,
    road: RoadPlane,
    contact_centre: Point3,
    wheel_axis: Vector3 | Direction3,
    side_sign: float,
) -> float | None:
    """Return inward-positive steering-axis offset at ground for ``axis``.

    Following ISO 8855:2011 steering-axis offset at ground (§7.2.6), this is
    the signed lateral component along tyre ``Y_T`` from the contact centre to
    the axis intersection with the local road plane. ``None`` indicates that
    the line or tyre-road basis cannot be resolved.
    """
    displacement = _road_displacement(axis, road, contact_centre)
    tyre_axes = _tyre_road_axes(wheel_axis, road, side_sign)
    if displacement is None or tyre_axes is None:
        return None
    _, lateral = tyre_axes
    return -side_sign * displacement.dot(lateral)


def calculate_mechanical_trail(
    axis: SteeringAxis,
    road: RoadPlane,
    contact_centre: Point3,
    wheel_axis: Vector3 | Direction3,
    side_sign: float,
) -> float | None:
    """Return ahead-positive wheel-relative mechanical trail for ``axis``.

    This is ISO 8855:2011 castor offset at ground (§7.2.3): the longitudinal
    component along tyre ``X_T`` from the contact centre to the axis-road
    intersection. It follows the steered tyre basis rather than chassis X.
    ``None`` indicates that the line or tyre-road basis cannot be resolved.
    """
    displacement = _road_displacement(axis, road, contact_centre)
    tyre_axes = _tyre_road_axes(wheel_axis, road, side_sign)
    if displacement is None or tyre_axes is None:
        return None
    longitudinal, _ = tyre_axes
    return displacement.dot(longitudinal)


def _road_displacement(
    axis: SteeringAxis,
    road: RoadPlane,
    contact_centre: Point3,
) -> Vector3 | None:
    """Return contact-to-axis-intersection displacement in the road plane."""
    intersection = axis.intersect_road(road)
    if intersection is None or not np.isfinite(contact_centre.data).all():
        return None
    return intersection - contact_centre


def _tyre_road_axes(
    wheel_axis: Vector3 | Direction3,
    road: RoadPlane,
    side_sign: float,
) -> tuple[Direction3, Direction3] | None:
    """Return tyre-forward and tyre-lateral directions within ``road``."""
    if abs(side_sign) < EPS_GEOMETRIC or not np.isfinite(wheel_axis.data).all():
        return None
    magnitude = float(np.linalg.norm(wheel_axis.data))
    if magnitude < EPS_GEOMETRIC:
        return None
    wheel_direction = Direction3.from_trusted(wheel_axis.data / magnitude)
    projected = wheel_direction.vector() - road.normal * wheel_direction.dot(
        road.normal
    )
    if projected.norm() < EPS_GEOMETRIC:
        return None
    outboard = projected.normalize()
    longitudinal = (side_sign * outboard.cross(road.normal)).normalize()
    lateral = road.normal.cross(longitudinal).normalize()
    return longitudinal, lateral
