"""
Geometric constraints for kinematic systems.

This module defines constraint classes that enforce geometric relationships between
points in suspension kinematics, such as distances, angles, and positional constraints.
Each constraint computes a residual value that the solver attempts to drive to zero.
"""

from abc import ABC, abstractmethod
from math import atan2
from typing import Set

import numpy as np

from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.soft_math import softnorm
from kinematics.core.vector_utils.geometric import (
    compute_point_to_plane_distance,
    compute_scalar_triple_product,
)


class Constraint(ABC):
    """
    Base class for all kinematics constraints.

    Constraints define geometric relationships that must be satisfied in the kinematic
    system. Each constraint computes a residual value representing the deviation from
    the desired condition. The solver minimizes these residuals to find valid
    configurations.
    """

    @property
    @abstractmethod
    def involved_points(self) -> Set[PointID]:
        """
        Returns a set of all PointIDs that this constraint operates on.
        """
        pass

    @abstractmethod
    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Calculate constraint residual.

        The solver's goal is to drive this value to zero. Returns a scalar float
        representing the constraint violation.
        """
        pass


class DistanceConstraint(Constraint):
    """
    Constrains the Euclidean distance between two points.

    This constraint enforces that the distance between two specified points remains
    constant at a target value, useful for rigid links or fixed separations in the
    suspension geometry.
    """

    def __init__(self, p1: PointID, p2: PointID, target_distance: float):
        """
        Initialize the distance constraint.

        Args:
            p1: First point identifier.
            p2: Second point identifier.
            target_distance: The required distance between the points.

        Raises:
            ValueError: If target_distance is negative.
        """
        if target_distance < 0:
            raise ValueError(
                f"Target distance must be non-negative, got {target_distance}"
            )

        self.p1 = p1
        self.p2 = p2
        self.target_distance = target_distance

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the distance residual.

        Returns the difference between the current distance and target distance.
        Uses softnorm(dx^2 + dy^2 + dz^2) to match the analytical Jacobian.
        """
        delta = positions[self.p2] - positions[self.p1]
        current_distance = softnorm(delta.squared_norm())
        return float(current_distance - self.target_distance)


class SphericalJointConstraint(Constraint):
    """
    Constrains two points to coincide (ball joint / spherical joint).

    This is a special case of DistanceConstraint with target_distance = 0, but made
    explicit for clarity when modeling ball joints in suspension systems.
    """

    def __init__(self, p1: PointID, p2: PointID):
        """
        Initialize the spherical joint constraint.

        Args:
            p1: First point identifier.
            p2: Second point identifier.
        """
        self.p1 = p1
        self.p2 = p2

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the distance between the two points.

        Returns softnorm of the squared separation (should be zero for a
        perfect joint). Uses softnorm to match the analytical Jacobian.
        """
        delta = positions[self.p2] - positions[self.p1]
        return float(softnorm(delta.squared_norm()))


class AngleConstraint(Constraint):
    """
    Constrains the angle between two vectors.

    This constraint enforces that the angle formed by two vectors (defined by point
    pairs) remains at a specified target angle. Useful for maintaining joint angles or
    geometric relationships in suspension linkages.
    """

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
        target_angle: float,
    ):
        """
        Initialize the angle constraint.

        Args:
            v1_start: Starting point of the first vector.
            v1_end: Ending point of the first vector.
            v2_start: Starting point of the second vector.
            v2_end: Ending point of the second vector.
            target_angle: The required angle between the vectors in radians.

        Raises:
            ValueError: If target_angle is outside [0, pi].
        """
        if not (0 <= target_angle <= np.pi):
            raise ValueError(f"Target angle must be in [0, pi], got {target_angle}")

        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end
        self.target_angle = target_angle

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the angle residual.

        Returns the difference between the current angle and target angle in radians.
        Uses raw (unnormalized) vectors with softnorm on the cross product magnitude,
        matching the analytical Jacobian: atan2(softnorm(|cross|^2), dot) - target.
        """
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]

        # Cross product components.
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]

        cross_mag = softnorm(cx * cx + cy * cy + cz * cz)
        dot_raw = v1.dot(v2)

        current_angle = atan2(cross_mag, dot_raw)
        return float(current_angle - self.target_angle)


class ThreePointAngleConstraint(Constraint):
    """
    Constrains the angle formed by three points (vertex at p2).

    This is often more intuitive than AngleConstraint for suspension geometry, as it
    directly specifies the angle at a joint formed by three connection points.
    """

    def __init__(
        self,
        p1: PointID,
        p2: PointID,  # vertex
        p3: PointID,
        target_angle: float,
    ):
        """
        Initialize the three-point angle constraint.

        Args:
            p1: First point.
            p2: Vertex point (angle is measured here).
            p3: Third point.
            target_angle: The required angle at p2 in radians.

        Raises:
            ValueError: If target_angle is outside [0, pi].
        """
        if not (0 <= target_angle <= np.pi):
            raise ValueError(f"Target angle must be in [0, pi], got {target_angle}")

        self.p1 = p1
        self.p2 = p2  # vertex
        self.p3 = p3
        self.target_angle = target_angle

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2, self.p3}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the angle residual at the vertex.

        Returns the difference between the current angle and target angle in radians.
        Uses raw (unnormalized) vectors from the vertex with softnorm on the cross
        product magnitude, matching the analytical Jacobian.
        """
        # Vectors from vertex to other points.
        v1 = positions[self.p1] - positions[self.p2]
        v2 = positions[self.p3] - positions[self.p2]

        # Cross product components.
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]

        cross_mag = softnorm(cx * cx + cy * cy + cz * cz)
        dot_raw = v1.dot(v2)

        current_angle = atan2(cross_mag, dot_raw)
        return float(current_angle - self.target_angle)


class VectorsParallelConstraint(Constraint):
    """
    Constrains two vectors to be parallel (or anti-parallel).

    Useful for parallel links, anti-roll bars, or ensuring vectors remain aligned in
    suspension systems.
    """

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
    ):
        """
        Initialize the parallel vectors constraint.

        Args:
            v1_start: Starting point of the first vector.
            v1_end: Ending point of the first vector.
            v2_start: Starting point of the second vector.
            v2_end: Ending point of the second vector.
        """
        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the parallel vectors constraint residual.

        Returns softnorm(|cross|^2) / (softnorm(|v1|^2) * softnorm(|v2|^2)),
        matching the analytical Jacobian. Zero indicates the vectors are
        parallel or anti-parallel.
        """
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]

        # Cross product components.
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]

        cross_mag = softnorm(cx * cx + cy * cy + cz * cz)
        v1_mag = softnorm(v1.squared_norm())
        v2_mag = softnorm(v2.squared_norm())

        return float(cross_mag / (v1_mag * v2_mag))


class VectorsPerpendicularConstraint(Constraint):
    """
    Constrains two vectors to be perpendicular.

    Useful for coordinate frames, orthogonal linkages, or maintaining perpendicular
    relationships in suspension geometry.
    """

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
    ):
        """
        Initialize the perpendicular vectors constraint.

        Args:
            v1_start: Starting point of the first vector.
            v1_end: Ending point of the first vector.
            v2_start: Starting point of the second vector.
            v2_end: Ending point of the second vector.
        """
        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the perpendicular constraint residual.

        Returns dot(v1, v2) / (softnorm(|v1|^2) * softnorm(|v2|^2)),
        matching the analytical Jacobian. Zero indicates the vectors are
        perpendicular.
        """
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]

        dot_raw = v1.dot(v2)
        v1_mag = softnorm(v1.squared_norm())
        v2_mag = softnorm(v2.squared_norm())

        return float(dot_raw / (v1_mag * v2_mag))


class EqualDistanceConstraint(Constraint):
    """Constrains two distances to be equal: |p1-p2| = |p3-p4|.

    Useful for symmetric linkages, equal-length links, or maintaining geometric
    relationships in suspension systems.
    """

    def __init__(
        self,
        p1: PointID,
        p2: PointID,
        p3: PointID,
        p4: PointID,
    ):
        """
        Initialize the equal distance constraint.

        Args:
            p1: First point of the first pair.
            p2: Second point of the first pair.
            p3: First point of the second pair.
            p4: Second point of the second pair.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2, self.p3, self.p4}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the equal distance residual.

        Returns softnorm(|d1|^2) - softnorm(|d2|^2), matching the analytical
        Jacobian. Zero indicates the distances are equal.
        """
        d1 = positions[self.p2] - positions[self.p1]
        d2 = positions[self.p4] - positions[self.p3]
        dist1 = softnorm(d1.squared_norm())
        dist2 = softnorm(d2.squared_norm())
        return float(dist1 - dist2)


class FixedAxisConstraint(Constraint):
    """
    Constrains a point's coordinate on a principal axis.

    This constraint fixes a specific coordinate (X, Y, or Z) of a point to a constant
    value, useful for ground-fixed points or symmetry constraints in the suspension
    system.
    """

    def __init__(self, point_id: PointID, axis: Axis, value: float):
        """
        Initialize the fixed axis constraint.

        Args:
            point_id: The point whose coordinate is constrained.
            axis: The axis (X, Y, or Z) to constrain.
            value: The fixed coordinate value on the specified axis.
        """
        self.point_id = point_id
        self.axis = axis
        self.value = value

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the axis coordinate residual.

        Returns the difference between the current coordinate and the fixed value.
        Positive values indicate the coordinate is above the target.
        """
        point_coord = positions[self.point_id][self.axis]
        return float(point_coord - self.value)


class PointOnLineConstraint(Constraint):
    """
    Constrains a point to lie on an arbitrary line.

    This constraint enforces that a point remains on a specified infinite line defined
    by a point and direction vector. Useful for guiding points along linear paths or
    maintaining alignment in suspension mechanisms.
    """

    def __init__(
        self,
        point_id: PointID,
        line_point: Point3,
        line_direction: Direction3,
    ):
        """
        Initialize the point-on-line constraint.

        Args:
            point_id: The point that must lie on the line.
            line_point: A point on the line.
            line_direction: The direction of the line (unit vector).

        Raises:
            ValueError: If line_direction has zero length.
        """
        if not isinstance(line_point, Point3):
            raise TypeError("line_point must be a Point3")
        if not isinstance(line_direction, Direction3):
            raise TypeError("line_direction must be a Direction3")

        self.point_id = point_id
        self.line_point = line_point.copy()
        self.line_direction = line_direction

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the point-to-line distance residual.

        Returns softnorm(|cross(p - line_point, line_direction)|^2), matching
        the analytical Jacobian. Zero indicates the point lies exactly on the
        line.
        """
        w = positions[self.point_id] - self.line_point
        ld = self.line_direction

        # Cross product components: w x line_direction.
        cx = w[1] * ld[2] - w[2] * ld[1]
        cy = w[2] * ld[0] - w[0] * ld[2]
        cz = w[0] * ld[1] - w[1] * ld[0]

        return float(softnorm(cx * cx + cy * cy + cz * cz))


class PointOnPlaneConstraint(Constraint):
    """
    Constrains a point to lie on a plane.

    Useful for restricting motion to a plane, enforcing symmetry constraints, or
    modeling planar mechanisms in suspension systems.
    """

    def __init__(
        self,
        point_id: PointID,
        plane_point: Point3,
        plane_normal: Direction3,
    ):
        """
        Initialize the point-on-plane constraint.

        Args:
            point_id: The point that must lie on the plane.
            plane_point: A point on the plane.
            plane_normal: The normal direction of the plane (unit vector).
        """
        if not isinstance(plane_point, Point3):
            raise TypeError("plane_point must be a Point3")
        if not isinstance(plane_normal, Direction3):
            raise TypeError("plane_normal must be a Direction3")

        self.point_id = point_id
        self.plane_point = plane_point.copy()
        self.plane_normal = plane_normal

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the signed distance from point to plane.

        Returns the signed distance (positive on the normal side, negative on the
        opposite side). Zero indicates the point lies exactly on the plane.
        """
        return compute_point_to_plane_distance(
            positions[self.point_id],
            self.plane_point,
            self.plane_normal,
        )


class CoplanarPointsConstraint(Constraint):
    """
    Constrains four points to be coplanar.

    Useful for ensuring points lie in the same plane or for modeling planar mechanisms
    in suspension systems.
    """

    def __init__(self, p1: PointID, p2: PointID, p3: PointID, p4: PointID):
        """
        Initialize the coplanar points constraint.

        Args:
            p1: First point.
            p2: Second point.
            p3: Third point.
            p4: Fourth point.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2, self.p3, self.p4}

    def residual(self, positions: dict[PointID, Point3]) -> float:
        """
        Compute the coplanarity residual using scalar triple product.

        Returns the scalar triple product v1 dot (v2 cross v3), where vectors are from
        p1 to the other points. Zero indicates the points are coplanar.
        """
        pos1 = positions[self.p1]
        v1 = positions[self.p2] - pos1
        v2 = positions[self.p3] - pos1
        v3 = positions[self.p4] - pos1
        return compute_scalar_triple_product(v1, v2, v3)
