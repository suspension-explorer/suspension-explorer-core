"""
Tests for the affine geometry primitives: Point3, Vector3, Direction3.

These lock down the CGAL-style type algebra (which operations are legal and
what they return), the copy/aliasing semantics, the tolerant-comparison
contract, and the numpy interop layer.
"""

import numpy as np
import pytest

from kinematics.core.constants import EPS_GEOMETRIC, TEST_TOLERANCE
from kinematics.core.geometry import (
    NAN_POINT3,
    Direction3,
    Point3,
    Vector3,
    extract_array,
    midpoint,
)


class TestConstruction:
    """
    Construction, validation, and copy semantics.
    """

    def test_point_from_list(self):
        p = Point3([1.0, 2.0, 3.0])
        assert p[0] == 1.0
        assert p[1] == 2.0
        assert p[2] == 3.0

    def test_vector_from_tuple(self):
        v = Vector3((1.0, 2.0, 3.0))
        np.testing.assert_allclose(v.data, [1.0, 2.0, 3.0], atol=TEST_TOLERANCE)

    @pytest.mark.parametrize("cls", [Point3, Vector3])
    def test_wrong_shape_raises(self, cls):
        with pytest.raises(ValueError, match="shape"):
            cls([1.0, 2.0])

    def test_direction_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Direction3([1.0, 2.0, 3.0, 4.0])

    def test_constructor_owns_its_buffer(self):
        # Mutating the source array after construction must not leak in.
        src = np.array([1.0, 2.0, 3.0])
        p = Point3(src)
        src[0] = 99.0
        assert p[0] == 1.0

    def test_copy_is_independent(self):
        p = Point3([1.0, 2.0, 3.0])
        q = p.copy()
        q.data[0] = 99.0
        assert p[0] == 1.0

    def test_copy_constructor_from_same_type(self):
        p = Point3([1.0, 2.0, 3.0])
        q = Point3(p)
        q.data[0] = 99.0
        assert p[0] == 1.0

    def test_from_trusted_aliases_storage(self):
        # from_trusted intentionally shares the buffer (hot-path contract).
        buf = np.array([1.0, 2.0, 3.0])
        p = Point3.from_trusted(buf)
        buf[0] = 99.0
        assert p[0] == 99.0


class TestAffineAlgebra:
    """
    Legal affine operations and their return types.
    """

    def test_point_minus_point_is_vector(self):
        a = Point3([3.0, 2.0, 1.0])
        b = Point3([1.0, 1.0, 1.0])
        result = a - b
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(result.data, [2.0, 1.0, 0.0], atol=TEST_TOLERANCE)

    def test_point_plus_vector_is_point(self):
        p = Point3([1.0, 1.0, 1.0])
        v = Vector3([1.0, 2.0, 3.0])
        result = p + v
        assert isinstance(result, Point3)
        np.testing.assert_allclose(result.data, [2.0, 3.0, 4.0], atol=TEST_TOLERANCE)

    def test_vector_plus_point_is_point(self):
        v = Vector3([1.0, 2.0, 3.0])
        p = Point3([1.0, 1.0, 1.0])
        result = v + p
        assert isinstance(result, Point3)
        np.testing.assert_allclose(result.data, [2.0, 3.0, 4.0], atol=TEST_TOLERANCE)

    def test_point_minus_vector_is_point(self):
        p = Point3([2.0, 3.0, 4.0])
        v = Vector3([1.0, 1.0, 1.0])
        result = p - v
        assert isinstance(result, Point3)
        np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0], atol=TEST_TOLERANCE)

    def test_vector_plus_vector_is_vector(self):
        a = Vector3([1.0, 0.0, 0.0])
        b = Vector3([0.0, 2.0, 0.0])
        result = a + b
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(result.data, [1.0, 2.0, 0.0], atol=TEST_TOLERANCE)

    def test_vector_minus_vector_is_vector(self):
        a = Vector3([3.0, 2.0, 1.0])
        b = Vector3([1.0, 1.0, 1.0])
        result = a - b
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(result.data, [2.0, 1.0, 0.0], atol=TEST_TOLERANCE)

    def test_vector_scalar_multiplication(self):
        v = Vector3([1.0, 2.0, 3.0])
        assert isinstance(v * 2, Vector3)
        np.testing.assert_allclose((v * 2).data, [2.0, 4.0, 6.0], atol=TEST_TOLERANCE)
        np.testing.assert_allclose((2 * v).data, [2.0, 4.0, 6.0], atol=TEST_TOLERANCE)

    def test_vector_division(self):
        v = Vector3([2.0, 4.0, 6.0])
        np.testing.assert_allclose((v / 2).data, [1.0, 2.0, 3.0], atol=TEST_TOLERANCE)

    def test_vector_negation(self):
        v = Vector3([1.0, -2.0, 3.0])
        np.testing.assert_allclose((-v).data, [-1.0, 2.0, -3.0], atol=TEST_TOLERANCE)

    def test_midpoint(self):
        a = Point3([0.0, 0.0, 0.0])
        b = Point3([2.0, 4.0, 6.0])
        m = midpoint(a, b)
        assert isinstance(m, Point3)
        np.testing.assert_allclose(m.data, [1.0, 2.0, 3.0], atol=TEST_TOLERANCE)


class TestDisallowedOperations:
    """
    Affine-illegal operations must raise TypeError.
    """

    def test_point_plus_point_raises(self):
        a = Point3([1.0, 2.0, 3.0])
        b = Point3([4.0, 5.0, 6.0])
        # Type checker flags this at static time too; we assert the runtime guard.
        with pytest.raises(TypeError):
            _ = a + b  # type: ignore[unsupported-operator]

    def test_point_times_scalar_raises(self):
        p = Point3([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            _ = p * 2  # type: ignore[unsupported-operator]

    def test_vector_times_vector_raises(self):
        a = Vector3([1.0, 2.0, 3.0])
        b = Vector3([4.0, 5.0, 6.0])
        with pytest.raises(TypeError):
            _ = a * b  # type: ignore[unsupported-operator]


class TestDirection3:
    """
    Direction3 normalization and orientation algebra.
    """

    def test_normalizes_on_construction(self):
        d = Direction3([0.0, 0.0, 5.0])
        np.testing.assert_allclose(d.data, [0.0, 0.0, 1.0], atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(d.data), 1.0, atol=TEST_TOLERANCE)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            Direction3([0.0, 0.0, 0.0])

    def test_below_epsilon_raises(self):
        tiny = EPS_GEOMETRIC / 10.0
        with pytest.raises(ValueError, match="zero-length"):
            Direction3([tiny, 0.0, 0.0])

    def test_scaling_produces_vector(self):
        d = Direction3([1.0, 0.0, 0.0])
        result = d * 3.0
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(result.data, [3.0, 0.0, 0.0], atol=TEST_TOLERANCE)
        np.testing.assert_allclose((2.0 * d).data, [2.0, 0.0, 0.0], atol=TEST_TOLERANCE)

    def test_negation_produces_direction(self):
        d = Direction3([1.0, 0.0, 0.0])
        result = -d
        assert isinstance(result, Direction3)
        np.testing.assert_allclose(result.data, [-1.0, 0.0, 0.0], atol=TEST_TOLERANCE)

    def test_vector_method_returns_unit_vector(self):
        d = Direction3([0.0, 3.0, 4.0])
        v = d.vector()
        assert isinstance(v, Vector3)
        assert np.isclose(v.norm(), 1.0, atol=TEST_TOLERANCE)

    def test_direction_from_vector(self):
        d = Direction3(Vector3([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(d.data, [0.0, 0.0, 1.0], atol=TEST_TOLERANCE)


class TestVectorMethods:
    """
    Linear-algebra methods on Vector3.
    """

    def test_dot(self):
        a = Vector3([1.0, 2.0, 3.0])
        b = Vector3([4.0, 5.0, 6.0])
        assert np.isclose(a.dot(b), 32.0, atol=TEST_TOLERANCE)

    def test_cross(self):
        a = Vector3([1.0, 0.0, 0.0])
        b = Vector3([0.0, 1.0, 0.0])
        result = a.cross(b)
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(result.data, [0.0, 0.0, 1.0], atol=TEST_TOLERANCE)

    def test_norm_and_squared_norm(self):
        v = Vector3([0.0, 3.0, 4.0])
        assert np.isclose(v.norm(), 5.0, atol=TEST_TOLERANCE)
        assert np.isclose(v.squared_norm(), 25.0, atol=TEST_TOLERANCE)

    def test_normalize_returns_direction(self):
        v = Vector3([0.0, 0.0, 10.0])
        d = v.normalize()
        assert isinstance(d, Direction3)
        np.testing.assert_allclose(d.data, [0.0, 0.0, 1.0], atol=TEST_TOLERANCE)

    def test_normalize_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            Vector3([0.0, 0.0, 0.0]).normalize()


class TestEquality:
    """
    Exact (__eq__) and tolerant (almost_equals) comparison.
    """

    def test_exact_equality(self):
        assert Point3([1.0, 2.0, 3.0]) == Point3([1.0, 2.0, 3.0])
        assert Vector3([1.0, 2.0, 3.0]) == Vector3([1.0, 2.0, 3.0])
        assert Direction3([1.0, 0.0, 0.0]) == Direction3([1.0, 0.0, 0.0])

    def test_inequality(self):
        assert Point3([1.0, 2.0, 3.0]) != Point3([1.0, 2.0, 3.1])

    def test_eq_across_types_is_not_equal(self):
        # __eq__ returns NotImplemented for foreign types; Python falls back to
        # identity, so a Point3 and a Vector3 with identical data are not equal.
        assert Point3([1.0, 2.0, 3.0]) != Vector3([1.0, 2.0, 3.0])

    def test_almost_equals_within_tolerance(self):
        a = Point3([1.0, 2.0, 3.0])
        b = Point3([1.0 + 1e-9, 2.0, 3.0])
        assert a.almost_equals(b)
        assert not a.almost_equals(Point3([1.0, 2.0, 3.5]))

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            (Point3([1.0, 2.0, 3.0]), Vector3([1.0, 2.0, 3.0])),
            (Vector3([1.0, 2.0, 3.0]), Point3([1.0, 2.0, 3.0])),
            (Direction3([1.0, 0.0, 0.0]), Vector3([1.0, 0.0, 0.0])),
        ],
    )
    def test_almost_equals_type_mismatch_raises(self, lhs, rhs):
        with pytest.raises(TypeError, match="almost_equals expects"):
            lhs.almost_equals(rhs)


class TestNumpyInterop:
    """
    The __array_function__ / __array_ufunc__ dispatch layer.
    """

    def test_np_dot(self):
        a = Vector3([1.0, 2.0, 3.0])
        b = Vector3([4.0, 5.0, 6.0])
        assert np.isclose(np.dot(a, b), 32.0, atol=TEST_TOLERANCE)

    def test_np_cross_returns_vector3(self):
        a = Vector3([1.0, 0.0, 0.0])
        b = Vector3([0.0, 1.0, 0.0])
        result = np.cross(a, b)
        assert isinstance(result, Vector3)
        # Read via extract_array (-> ndarray); result.data narrows against the
        # numpy stub's ndarray return type and would resolve to a memoryview.
        np.testing.assert_allclose(
            extract_array(result), [0.0, 0.0, 1.0], atol=TEST_TOLERANCE
        )

    def test_np_linalg_norm(self):
        v = Vector3([0.0, 3.0, 4.0])
        assert np.isclose(np.linalg.norm(v), 5.0, atol=TEST_TOLERANCE)

    def test_ufunc_scalar_multiply(self):
        v = Vector3([1.0, 2.0, 3.0])
        result = np.multiply(2.0, v)
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(
            extract_array(result), [2.0, 4.0, 6.0], atol=TEST_TOLERANCE
        )

    def test_ufunc_ndarray_add_and_subtract(self):
        v = Vector3([1.0, 2.0, 3.0])
        arr = np.array([1.0, 1.0, 1.0])
        added = np.add(arr, v)
        subtracted = np.subtract(arr, v)
        assert isinstance(added, Vector3)
        assert isinstance(subtracted, Vector3)
        np.testing.assert_allclose(
            extract_array(added), [2.0, 3.0, 4.0], atol=TEST_TOLERANCE
        )
        np.testing.assert_allclose(
            extract_array(subtracted), [0.0, -1.0, -2.0], atol=TEST_TOLERANCE
        )

    def test_direction_ufunc_scalar_multiply(self):
        d = Direction3([1.0, 0.0, 0.0])
        result = np.multiply(d, 3.0)
        assert isinstance(result, Vector3)
        np.testing.assert_allclose(
            extract_array(result), [3.0, 0.0, 0.0], atol=TEST_TOLERANCE
        )

    def test_extract_array_unwraps_wrappers(self):
        np.testing.assert_array_equal(
            extract_array(Point3([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            extract_array(np.array([4.0, 5.0, 6.0])), [4.0, 5.0, 6.0]
        )


def test_nan_point_sentinel():
    assert isinstance(NAN_POINT3, Point3)
    assert np.all(np.isnan(NAN_POINT3.data))
