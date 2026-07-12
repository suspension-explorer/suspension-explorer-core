"""
Tests for the generic vector utility functions.
"""

import numpy as np
import pytest

from kinematics.core.constants import EPS_NUMERICAL, TEST_TOLERANCE
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.vector_utils.generic import (
    compute_2d_vector_vector_intersection,
    normalize_vector,
    project_coordinate,
)


class TestNormalizeVector:
    """
    Tests for the normalize_vector function.
    """

    def test_normalize_unit_vector_x(self):
        """
        Test normalizing a unit vector in x direction.
        """
        v = np.array([1.0, 0.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_unit_vector_y(self):
        """
        Test normalizing a unit vector in y direction.
        """
        v = np.array([0.0, 1.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_unit_vector_z(self):
        """
        Test normalizing a unit vector in z direction.
        """
        v = np.array([0.0, 0.0, 1.0])
        result = normalize_vector(v)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_scaled_vector(self):
        """
        Test normalizing a scaled vector.
        """
        v = np.array([3.0, 4.0, 0.0])  # Magnitude = 5.
        result = normalize_vector(v)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_3d_vector(self):
        """
        Test normalizing a 3D vector.
        """
        v = np.array([1.0, 2.0, 2.0])  # Magnitude = 3.
        result = normalize_vector(v)
        expected = np.array([1 / 3, 2 / 3, 2 / 3])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_negative_vector(self):
        """
        Test normalizing a vector with negative components.
        """
        v = np.array([-3.0, -4.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([-0.6, -0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_mixed_sign_vector(self):
        """
        Test normalizing a vector with mixed positive/negative components.
        """
        v = np.array([3.0, -4.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([0.6, -0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_2d_vector(self):
        """
        Test normalizing a 2D vector.
        """
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_higher_dimension_vector(self):
        """
        Test normalizing a higher dimensional vector.
        """
        v = np.array([1.0, 1.0, 1.0, 1.0])  # Magnitude = 2.
        result = normalize_vector(v)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_very_small_vector(self):
        """
        Test normalizing a very small but non-zero vector.
        """
        v = np.array([TEST_TOLERANCE * 10, 0.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_zero_vector_raises_error(self):
        """
        Test that normalizing a zero vector raises ValueError.
        """
        v = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot normalize zero-length vector"):
            normalize_vector(v)

    def test_normalize_near_zero_vector_raises_error(self):
        """
        Test that normalizing a near-zero vector raises ValueError.
        """
        v = np.array([EPS_NUMERICAL / 2, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot normalize zero-length vector"):
            normalize_vector(v)

    def test_normalize_returns_float64(self):
        """
        Test that the result is of type float64.
        """
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = normalize_vector(v)
        assert result.dtype == np.float64


class TestProjectCoordinate:
    """
    Tests for the project_coordinate function.
    """

    def test_project_along_x_axis(self):
        """
        Test projection along x-axis.
        """
        position = Point3([3.0, 2.0, 1.0])
        direction = Direction3([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 3.0, atol=TEST_TOLERANCE)

    def test_project_along_y_axis(self):
        """
        Test projection along y-axis.
        """
        position = Point3([3.0, 2.0, 1.0])
        direction = Direction3([0.0, 1.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 2.0, atol=TEST_TOLERANCE)

    def test_project_along_z_axis(self):
        """
        Test projection along z-axis.
        """
        position = Point3([3.0, 2.0, 1.0])
        direction = Direction3([0.0, 0.0, 1.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 1.0, atol=TEST_TOLERANCE)

    def test_project_along_diagonal(self):
        """
        Test projection along a diagonal direction.
        """
        position = Point3([1.0, 1.0, 0.0])
        # Direction3 auto-normalizes; an unnormalized diagonal works fine.
        direction = Direction3([1.0, 1.0, 0.0])
        result = project_coordinate(position, direction)
        expected = np.sqrt(2)  # ||(1,1,0)|| projected onto (1/sqrt(2), 1/sqrt(2), 0).
        assert np.isclose(result, expected, atol=TEST_TOLERANCE)

    def test_project_negative_direction(self):
        """
        Test projection in negative direction.
        """
        position = Point3([3.0, 2.0, 1.0])
        direction = Direction3([-1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, -3.0, atol=TEST_TOLERANCE)

    def test_project_orthogonal_vectors(self):
        """
        Test projection of orthogonal vectors gives zero.
        """
        position = Point3([1.0, 0.0, 0.0])
        direction = Direction3([0.0, 1.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 0.0, atol=TEST_TOLERANCE)

    def test_project_zero_position(self):
        """
        Test projection of zero position vector.
        """
        position = Point3([0.0, 0.0, 0.0])
        direction = Direction3([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 0.0, atol=TEST_TOLERANCE)

    def test_project_arbitrary_unit_direction(self):
        """
        Test projection along an arbitrary unit direction.
        """
        position = Point3([2.0, 3.0, 6.0])
        # Direction3 auto-normalizes this raw vector.
        direction = Direction3([1.0, 2.0, 2.0])
        result = project_coordinate(position, direction)

        # Manual calculation: dot product against the normalized data.
        expected = np.dot(position.data, direction.data)
        assert np.isclose(result, expected, atol=TEST_TOLERANCE)

    def test_project_returns_float(self):
        """
        Test that the result is a float.
        """
        position = Point3([1.0, 2.0, 3.0])
        direction = Direction3([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert isinstance(result, float)

    def test_project_non_unit_direction_auto_normalizes(self):
        """
        Test that Direction3 auto-normalizes a non-unit input vector.

        The Direction3 constructor normalizes any non-zero vector, so passing
        a vector with magnitude != 1 produces a valid unit direction.
        """
        position = Point3([1.0, 2.0, 3.0])
        direction = Direction3([2.0, 0.0, 0.0])  # Magnitude = 2, auto-normalized.
        result = project_coordinate(position, direction)
        assert np.isclose(result, 1.0, atol=TEST_TOLERANCE)

    def test_project_zero_direction_raises_error(self):
        """
        Test that zero direction vector raises ValueError.

        The Direction3 constructor rejects zero-length vectors, which prevents
        constructing an invalid direction for projection.
        """
        with pytest.raises(ValueError, match="zero-length vector"):
            Direction3([0.0, 0.0, 0.0])

    def test_project_near_unit_direction_passes(self):
        """
        Test that nearly unit direction vector passes within tolerance.
        """
        position = Point3([1.0, 2.0, 3.0])
        # Create a direction vector that's almost but not exactly unit length;
        # Direction3 normalizes it on construction.
        direction = Direction3([1.0 + EPS_NUMERICAL / 2, 0.0, 0.0])
        result = project_coordinate(position, direction)
        # Should be close to projecting onto [1,0,0].
        assert np.isclose(result, 1.0, atol=1e-3)

    def test_project_slightly_off_unit_direction_normalizes(self):
        """
        Test that a slightly off-unit direction is auto-normalized by Direction3.
        """
        position = Point3([1.0, 2.0, 3.0])
        # Direction3 auto-normalizes, so 1.1 magnitude becomes unit length.
        direction = Direction3([1.1, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 1.0, atol=1e-3)


class TestCompute2dVectorVectorIntersection:
    """
    Tests for the compute_2d_vector_vector_intersection function.
    """

    def test_basic_intersection(self):
        """
        Test basic perpendicular line intersection.
        """
        # Horizontal line from (0,0) to (2,0)
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([2.0, 0.0])
        # Vertical line from (1,-1) to (1,1)
        line2_start = np.array([1.0, -1.0])
        line2_end = np.array([1.0, 1.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        np.testing.assert_allclose(result.point, [1.0, 0.0], atol=TEST_TOLERANCE)
        assert np.isclose(result.t1, 0.5, atol=TEST_TOLERANCE)  # Middle of line1
        assert np.isclose(result.t2, 0.5, atol=TEST_TOLERANCE)  # Middle of line2

    def test_diagonal_intersection(self):
        """
        Test intersection of diagonal lines.
        """
        # Line from (0,0) to (2,2)
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([2.0, 2.0])
        # Line from (0,2) to (2,0)
        line2_start = np.array([0.0, 2.0])
        line2_end = np.array([2.0, 0.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        np.testing.assert_allclose(result.point, [1.0, 1.0], atol=TEST_TOLERANCE)
        assert np.isclose(result.t1, 0.5, atol=TEST_TOLERANCE)
        assert np.isclose(result.t2, 0.5, atol=TEST_TOLERANCE)

    def test_parallel_lines_no_intersection(self):
        """
        Test that parallel lines return None.
        """
        # Two horizontal parallel lines
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([2.0, 0.0])
        line2_start = np.array([0.0, 1.0])
        line2_end = np.array([2.0, 1.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is None

    def test_degenerate_line_no_intersection(self):
        """
        Test that degenerate (zero-length) lines return None.
        """
        # First line is degenerate (point)
        line1_start = np.array([1.0, 1.0])
        line1_end = np.array([1.0, 1.0])
        line2_start = np.array([0.0, 0.0])
        line2_end = np.array([2.0, 2.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is None

    def test_intersection_outside_segments(self):
        """
        Test intersection that lies outside both line segments.
        """
        # Two lines that would intersect if extended, but don't intersect as segments
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([1.0, 0.0])
        line2_start = np.array([2.0, -1.0])
        line2_end = np.array([2.0, 1.0])

        # With segments_only=True (default), should return None
        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end, segments_only=True
        )
        assert result is None

        # With segments_only=False, should return intersection
        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end, segments_only=False
        )
        assert result is not None
        np.testing.assert_allclose(result.point, [2.0, 0.0], atol=TEST_TOLERANCE)

    def test_intersection_at_endpoint(self):
        """
        Test intersection at line segment endpoints.
        """
        # Lines meeting at endpoint
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([1.0, 1.0])
        line2_start = np.array([1.0, 1.0])
        line2_end = np.array([2.0, 0.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        np.testing.assert_allclose(result.point, [1.0, 1.0], atol=TEST_TOLERANCE)
        assert np.isclose(result.t1, 1.0, atol=TEST_TOLERANCE)  # End of line1
        assert np.isclose(result.t2, 0.0, atol=TEST_TOLERANCE)  # Start of line2

    def test_2d_input_basic(self):
        """
        Test basic 2D input handling.
        """
        # 2D points for basic intersection test
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([2.0, 0.0])
        line2_start = np.array([1.0, -1.0])
        line2_end = np.array([1.0, 1.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        # Result should be 2D
        assert result.point.shape == (2,)
        np.testing.assert_allclose(result.point, [1.0, 0.0], atol=TEST_TOLERANCE)

    def test_intersection_parameters(self):
        """
        Test that parameter values are calculated correctly.
        """
        line1_start = np.array([0.0, 0.0])
        line1_end = np.array([4.0, 0.0])
        line2_start = np.array([1.0, -2.0])
        line2_end = np.array([1.0, 2.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        # Intersection at (1,0)
        np.testing.assert_allclose(result.point, [1.0, 0.0], atol=TEST_TOLERANCE)
        # t1 = 1/4 (1 is 1/4 of the way from 0 to 4 on line1)
        assert np.isclose(result.t1, 0.25, atol=TEST_TOLERANCE)
        # t2 = 1/2 (0 is 1/2 of the way from -2 to 2 on line2)
        assert np.isclose(result.t2, 0.5, atol=TEST_TOLERANCE)

    def test_negative_coordinates(self):
        """
        Test intersection with negative coordinates.
        """
        line1_start = np.array([-2.0, -1.0])
        line1_end = np.array([0.0, 1.0])
        line2_start = np.array([-1.0, -2.0])
        line2_end = np.array([-1.0, 2.0])

        result = compute_2d_vector_vector_intersection(
            line1_start, line1_end, line2_start, line2_end
        )

        assert result is not None
        np.testing.assert_allclose(result.point, [-1.0, 0.0], atol=TEST_TOLERANCE)
