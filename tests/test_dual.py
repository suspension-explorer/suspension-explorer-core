"""Tests for forward-mode automatic differentiation via dual numbers."""

import numpy as np
import pytest

from kinematics.core.dual import (
    DualScalar,
    DualVec3,
    atan2,
    cross,
    degrees,
    seed_positions,
    seed_positions_with_tangent,
    sqrt,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointKey
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.points.derived.definitions import get_wheel_center

# Finite-difference step and tolerance for comparing autodiff vs FD.
FD_STEP = 1e-7
FD_TOL = 1e-6


class TestDualScalar:
    """Tests for DualScalar arithmetic and comparisons."""

    def test_add_dual(self):
        a = DualScalar(2.0, 1.0)
        b = DualScalar(3.0, 0.5)
        c = a + b
        assert c.val == pytest.approx(5.0)
        assert c.deriv == pytest.approx(1.5)

    def test_add_float(self):
        a = DualScalar(2.0, 1.0)
        c = a + 3.0
        assert c.val == pytest.approx(5.0)
        assert c.deriv == pytest.approx(1.0)

    def test_radd_float(self):
        a = DualScalar(2.0, 1.0)
        c = 3.0 + a
        assert c.val == pytest.approx(5.0)
        assert c.deriv == pytest.approx(1.0)

    def test_sub_dual(self):
        a = DualScalar(5.0, 2.0)
        b = DualScalar(3.0, 0.5)
        c = a - b
        assert c.val == pytest.approx(2.0)
        assert c.deriv == pytest.approx(1.5)

    def test_mul_dual(self):
        a = DualScalar(2.0, 1.0)
        b = DualScalar(3.0, 0.5)
        c = a * b
        assert c.val == pytest.approx(6.0)
        # Product rule: 1.0*3.0 + 2.0*0.5 = 4.0.
        assert c.deriv == pytest.approx(4.0)

    def test_mul_float(self):
        a = DualScalar(2.0, 3.0)
        c = a * 5.0
        assert c.val == pytest.approx(10.0)
        assert c.deriv == pytest.approx(15.0)

    def test_div_dual(self):
        a = DualScalar(6.0, 1.0)
        b = DualScalar(3.0, 0.5)
        c = a / b
        assert c.val == pytest.approx(2.0)
        # Quotient rule: (1.0*3.0 - 6.0*0.5) / 9.0 = 0.0.
        assert c.deriv == pytest.approx(0.0)

    def test_div_float(self):
        a = DualScalar(6.0, 4.0)
        c = a / 2.0
        assert c.val == pytest.approx(3.0)
        assert c.deriv == pytest.approx(2.0)

    def test_neg(self):
        a = DualScalar(2.0, 1.0)
        c = -a
        assert c.val == pytest.approx(-2.0)
        assert c.deriv == pytest.approx(-1.0)

    def test_abs_positive(self):
        a = DualScalar(3.0, 2.0)
        c = abs(a)
        assert c.val == pytest.approx(3.0)
        assert c.deriv == pytest.approx(2.0)

    def test_abs_negative(self):
        a = DualScalar(-3.0, 2.0)
        c = abs(a)
        assert c.val == pytest.approx(3.0)
        assert c.deriv == pytest.approx(-2.0)

    def test_comparison_lt(self):
        a = DualScalar(1.0, 100.0)
        assert a < 2.0
        assert not (a < 0.5)

    def test_comparison_gt(self):
        a = DualScalar(3.0, 0.0)
        assert a > 2.0

    def test_float_conversion(self):
        a = DualScalar(4.5, 1.0)
        assert float(a) == pytest.approx(4.5)

    def test_rsub_float(self):
        a = DualScalar(2.0, 1.0)
        c = 5.0 - a
        assert c.val == pytest.approx(3.0)
        assert c.deriv == pytest.approx(-1.0)

    def test_rtruediv_float(self):
        a = DualScalar(2.0, 1.0)
        c = 6.0 / a
        assert c.val == pytest.approx(3.0)
        # d(6/a)/da = -6 * 1.0 / 4.0 = -1.5.
        assert c.deriv == pytest.approx(-1.5)

    def test_mul_with_dual_vec3(self):
        s = DualScalar(2.0, 1.0)
        v = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = s * v
        assert isinstance(result, DualVec3)
        np.testing.assert_allclose(result.val, [2.0, 4.0, 6.0])
        # Product rule: 1.0*[1,2,3] + 2.0*[0.1,0.2,0.3] = [1.2, 2.4, 3.6].
        np.testing.assert_allclose(result.deriv, [1.2, 2.4, 3.6])


class TestDualVec3:
    """Tests for DualVec3 arithmetic and indexing."""

    def test_add_dual_vec3(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        b = DualVec3(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 0.0]))
        c = a + b
        np.testing.assert_allclose(c.val, [5.0, 7.0, 9.0])
        np.testing.assert_allclose(c.deriv, [1.0, 1.0, 0.0])

    def test_sub_dual_vec3(self):
        a = DualVec3(np.array([4.0, 5.0, 6.0]), np.array([1.0, 0.0, 0.0]))
        b = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]))
        c = a - b
        np.testing.assert_allclose(c.val, [3.0, 3.0, 3.0])
        np.testing.assert_allclose(c.deriv, [1.0, -1.0, 0.0])

    def test_add_ndarray(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        b = np.array([10.0, 20.0, 30.0])
        c = a + b
        np.testing.assert_allclose(c.val, [11.0, 22.0, 33.0])
        np.testing.assert_allclose(c.deriv, [1.0, 0.0, 0.0])

    def test_radd_ndarray(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        b = np.array([10.0, 20.0, 30.0])
        c = b + a
        np.testing.assert_allclose(c.val, [11.0, 22.0, 33.0])
        np.testing.assert_allclose(c.deriv, [1.0, 0.0, 0.0])

    def test_mul_scalar_float(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        c = a * 3.0
        np.testing.assert_allclose(c.val, [3.0, 6.0, 9.0])
        np.testing.assert_allclose(c.deriv, [0.3, 0.6, 0.9])

    def test_rmul_scalar_float(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        c = 3.0 * a
        np.testing.assert_allclose(c.val, [3.0, 6.0, 9.0])
        np.testing.assert_allclose(c.deriv, [0.3, 0.6, 0.9])

    def test_rmul_int(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]))
        c = -1 * a
        np.testing.assert_allclose(c.val, [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(c.deriv, [0.0, 0.0, -1.0])

    def test_mul_dual_scalar(self):
        v = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        s = DualScalar(2.0, 0.5)
        c = v * s
        np.testing.assert_allclose(c.val, [2.0, 4.0, 6.0])
        # Product rule: [0.1,0.2,0.3]*2.0 + [1,2,3]*0.5.
        np.testing.assert_allclose(c.deriv, [0.7, 1.4, 2.1])

    def test_div_dual_scalar(self):
        v = DualVec3(np.array([6.0, 4.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        s = DualScalar(2.0, 0.5)
        c = v / s
        np.testing.assert_allclose(c.val, [3.0, 2.0, 1.0])
        # Quotient rule: ([1,1,1]*2 - [6,4,2]*0.5) / 4.
        np.testing.assert_allclose(c.deriv, [-0.25, 0.0, 0.25])

    def test_div_float(self):
        v = DualVec3(np.array([6.0, 4.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        c = v / 2.0
        np.testing.assert_allclose(c.val, [3.0, 2.0, 1.0])
        np.testing.assert_allclose(c.deriv, [0.5, 1.0, 1.5])

    def test_neg(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        c = -a
        np.testing.assert_allclose(c.val, [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(c.deriv, [-0.1, -0.2, -0.3])

    def test_getitem_int(self):
        v = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        s = v[1]
        assert isinstance(s, DualScalar)
        assert s.val == pytest.approx(2.0)
        assert s.deriv == pytest.approx(0.2)

    def test_getitem_axis_enum(self):
        v = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        s = v[Axis.Z]
        assert isinstance(s, DualScalar)
        assert s.val == pytest.approx(3.0)
        assert s.deriv == pytest.approx(0.3)

    def test_default_zero_deriv(self):
        v = DualVec3(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(v.deriv, [0.0, 0.0, 0.0])


class TestNumpyDispatch:
    """Tests for np.dot and np.linalg.norm dispatch to dual implementations."""

    def test_dot_dual_dual(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        b = DualVec3(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 0.0]))
        result = np.dot(a, b)  # ty: ignore[invalid-argument-type]  # __array_function__ protocol
        assert isinstance(result, DualScalar)
        # dot(val) = 4 + 10 + 18 = 32.
        assert result.val == pytest.approx(32.0)
        # dot(a', b) + dot(a, b') = 4 + 2 = 6.
        assert result.deriv == pytest.approx(6.0)

    def test_dot_dual_ndarray(self):
        a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        b = np.array([4.0, 5.0, 6.0])
        result = np.dot(a, b)  # ty: ignore[invalid-argument-type]  # __array_function__ protocol
        assert isinstance(result, DualScalar)
        assert result.val == pytest.approx(32.0)
        # dot([1,0,0], [4,5,6]) = 4.
        assert result.deriv == pytest.approx(4.0)

    def test_norm_dual(self):
        v = DualVec3(np.array([3.0, 4.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        result = np.linalg.norm(v)  # ty: ignore[no-matching-overload]  # __array_function__ protocol
        assert isinstance(result, DualScalar)
        assert result.val == pytest.approx(5.0)
        # d(||v||) = dot(v, v') / ||v|| = 3*1 / 5 = 0.6.
        assert result.deriv == pytest.approx(0.6)

    def test_norm_zero_vector(self):
        v = DualVec3(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        with pytest.raises(
            ValueError, match="Dual vector norm derivative is undefined at zero length"
        ):
            np.linalg.norm(v)  # ty: ignore[no-matching-overload]  # __array_function__ protocol


class TestSeedPositions:
    """Tests for the seed_positions utility function."""

    def test_seed_creates_dual_dict(self):
        positions = {
            PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
            PointID.AXLE_OUTBOARD: np.array([0.0, 100.0, 0.0]),
        }
        dual = seed_positions(positions, PointID.AXLE_OUTBOARD, 1)

        # All entries should be DualVec3.
        for pid in positions:
            assert isinstance(dual[pid], DualVec3)

        # The seeded point has deriv = e_1 (y-axis).
        np.testing.assert_allclose(dual[PointID.AXLE_OUTBOARD].deriv, [0.0, 1.0, 0.0])

        # The other point has zero derivative.
        np.testing.assert_allclose(dual[PointID.AXLE_INBOARD].deriv, [0.0, 0.0, 0.0])

    def test_seed_preserves_values(self):
        positions = {
            PointID.AXLE_INBOARD: np.array([1.0, 2.0, 3.0]),
            PointID.AXLE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
        }
        dual = seed_positions(positions, PointID.AXLE_INBOARD, 0)

        np.testing.assert_allclose(dual[PointID.AXLE_INBOARD].val, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(dual[PointID.AXLE_OUTBOARD].val, [4.0, 5.0, 6.0])


class TestAnalyticalDerivatives:
    """
    Verify autodiff produces exact derivatives by comparing against
    results derived by hand from calculus rules.
    """

    def test_scalar_polynomial(self):
        # f(x) = x^3 - 2x + 1, so f'(x) = 3x^2 - 2.
        # At x = 3: f = 27 - 6 + 1 = 22, f' = 27 - 2 = 25.
        x = DualScalar(3.0, 1.0)
        f = x * x * x - 2.0 * x + 1.0
        assert f.val == pytest.approx(22.0)
        assert f.deriv == pytest.approx(25.0)

    def test_scalar_rational_function(self):
        # f(x) = (x^2 + 1) / (x - 1).
        # By the quotient rule, f'(x) = (x^2 - 2x - 1) / (x - 1)^2.
        # At x = 3: f = 10/2 = 5, f' = (9 - 6 - 1) / 4 = 1/2.
        x = DualScalar(3.0, 1.0)
        f = (x * x + 1.0) / (x - 1.0)
        assert f.val == pytest.approx(5.0)
        assert f.deriv == pytest.approx(0.5)

    def test_scalar_chain_rule(self):
        # f(x) = 1 / (x^2 + 1), so f'(x) = -2x / (x^2 + 1)^2.
        # At x = 2: f = 1/5 = 0.2, f' = -4/25 = -0.16.
        x = DualScalar(2.0, 1.0)
        f = 1.0 / (x * x + 1.0)
        assert f.val == pytest.approx(0.2)
        assert f.deriv == pytest.approx(-0.16)

    def test_squared_norm(self):
        # f(v) = dot(v, v), so df/dv_i = 2 * v_i.
        # At v = (3, 4, 0): f = 25, df/dv = (6, 8, 0).
        v_val = np.array([3.0, 4.0, 0.0])
        for i, expected in enumerate([6.0, 8.0, 0.0]):
            v = DualVec3(v_val.copy(), np.eye(3)[i])
            f = np.dot(v, v)  # ty: ignore[invalid-argument-type]  # __array_function__ protocol
            assert f.val == pytest.approx(25.0)
            assert f.deriv == pytest.approx(expected)

    def test_unit_vector_projection(self):
        # f(v) = v_x / ||v||, the x-component of the unit vector.
        # df/dv_x = (||v||^2 - v_x^2) / ||v||^3.
        # df/dv_y = -v_x * v_y / ||v||^3.
        # df/dv_z = -v_x * v_z / ||v||^3.
        # At v = (3, 4, 0) with ||v|| = 5 and ||v||^3 = 125:
        #   df/dv_x = (25 - 9) / 125 = 16/125.
        #   df/dv_y = -12 / 125.
        #   df/dv_z = 0.
        v_val = np.array([3.0, 4.0, 0.0])
        e_x = np.array([1.0, 0.0, 0.0])

        expected_derivs = [16.0 / 125.0, -12.0 / 125.0, 0.0]
        for i, expected in enumerate(expected_derivs):
            v = DualVec3(v_val.copy(), np.eye(3)[i])
            unit = v / np.linalg.norm(v)  # ty: ignore[no-matching-overload]  # __array_function__ protocol
            f = np.dot(unit, e_x)  # __array_function__ protocol
            assert f.val == pytest.approx(3.0 / 5.0)
            assert f.deriv == pytest.approx(expected)

    def test_inverse_norm(self):
        # f(v) = 1 / ||v||, so df/dv_i = -v_i / ||v||^3.
        # At v = (3, 4, 0) with ||v||^3 = 125:
        #   df/dv_x = -3/125, df/dv_y = -4/125, df/dv_z = 0.
        v_val = np.array([3.0, 4.0, 0.0])

        expected_derivs = [-3.0 / 125.0, -4.0 / 125.0, 0.0]
        for i, expected in enumerate(expected_derivs):
            v = DualVec3(v_val.copy(), np.eye(3)[i])
            norm = np.linalg.norm(v)  # ty: ignore[no-matching-overload]  # __array_function__ protocol
            assert isinstance(norm, DualScalar)
            f = 1.0 / norm
            assert isinstance(f, DualScalar)
            assert f.val == pytest.approx(1.0 / 5.0)
            assert f.deriv == pytest.approx(expected)

    def test_squared_distance(self):
        # f(a) = dot(a - b, a - b) with b constant, so df/da_i = 2 * (a_i - b_i).
        # At a = (5, 1, 3), b = (2, 1, 0): delta = (3, 0, 3), f = 18, df/da = (6, 0, 6).
        a_val = np.array([5.0, 1.0, 3.0])
        b = np.array([2.0, 1.0, 0.0])

        expected_derivs = [6.0, 0.0, 6.0]
        for i, expected in enumerate(expected_derivs):
            a = DualVec3(a_val.copy(), np.eye(3)[i])
            delta = a - b
            f = np.dot(delta, delta)  # ty: ignore[invalid-argument-type]  # __array_function__ protocol
            assert f.val == pytest.approx(18.0)
            assert f.deriv == pytest.approx(expected)

    def test_normalize_vector_analytical(self):
        # normalize(v) = v / ||v||.
        # The Jacobian is d(hat_j)/dv_i = (delta_ij - hat_i * hat_j) / ||v||,
        # where hat = v / ||v||.
        # At v = (3, 4, 0) with ||v|| = 5 and hat = (3/5, 4/5, 0),
        # seeding dv_x = 1 (i = 0):
        #   d(hat_x)/dv_x = (1 - 9/25) / 5 = 16/125.
        #   d(hat_y)/dv_x = (0 - 12/25) / 5 = -12/125.
        #   d(hat_z)/dv_x = 0.
        v = DualVec3(np.array([3.0, 4.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        result = normalize_vector(v)

        np.testing.assert_allclose(result.val, [0.6, 0.8, 0.0])
        np.testing.assert_allclose(
            result.deriv,
            [16.0 / 125.0, -12.0 / 125.0, 0.0],
        )


class TestWheelCenterAutodiff:
    """
    Tests that the autodiff Jacobian for get_wheel_center matches finite
    differences.
    """

    def test_jacobian_matches_finite_differences(self):
        positions: dict[PointKey, np.ndarray] = {
            PointID.AXLE_INBOARD: np.array([0.0, -200.0, 300.0]),
            PointID.AXLE_OUTBOARD: np.array([0.0, -600.0, 300.0]),
        }
        wheel_offset = 25.0

        base = get_wheel_center(positions, wheel_offset)

        for pid in [PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD]:
            for d in range(3):
                # Autodiff derivative.
                dual_pos = seed_positions(positions, pid, d)
                result = get_wheel_center(dual_pos, wheel_offset)
                ad_deriv = result.deriv  # returns DualVec3 at runtime

                # Finite-difference derivative.
                saved = positions[pid].copy()
                positions[pid][d] += FD_STEP
                perturbed = get_wheel_center(positions, wheel_offset)
                fd_deriv = (perturbed - base) / FD_STEP
                positions[pid] = saved

                np.testing.assert_allclose(
                    ad_deriv,
                    fd_deriv,
                    atol=FD_TOL,
                    err_msg=f"Mismatch for {pid.name}[{d}]",
                )


def test_dual_cross_product_rule() -> None:
    a = DualVec3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
    b = DualVec3(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 0.0]))

    result = cross(a, b)

    np.testing.assert_allclose(result.val, [-3.0, 6.0, -3.0])
    np.testing.assert_allclose(result.deriv, [-3.0, -6.0, 6.0])


def test_dual_scalar_math() -> None:
    root = sqrt(DualScalar(9.0, 1.0))
    angle = atan2(DualScalar(1.0, 1.0), DualScalar(2.0, 0.0))
    angle_degrees = degrees(DualScalar(np.pi / 2.0, 1.0))

    assert isinstance(root, DualScalar)
    assert root.val == pytest.approx(3.0)
    assert root.deriv == pytest.approx(1.0 / 6.0)
    assert isinstance(angle, DualScalar)
    assert angle.val == pytest.approx(np.arctan2(1.0, 2.0))
    assert angle.deriv == pytest.approx(2.0 / 5.0)
    assert isinstance(angle_degrees, DualScalar)
    assert angle_degrees.val == pytest.approx(90.0)
    assert angle_degrees.deriv == pytest.approx(180.0 / np.pi)


def test_seed_positions_with_tangent_copies_derivatives() -> None:
    source_derivative = np.array([1.0, 2.0, 3.0])
    positions = {
        PointID.AXLE_INBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.AXLE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }

    dual_positions = seed_positions_with_tangent(
        positions,
        {PointID.AXLE_INBOARD: source_derivative},
    )
    source_derivative[0] = 99.0

    np.testing.assert_allclose(
        dual_positions[PointID.AXLE_INBOARD].deriv,
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        dual_positions[PointID.AXLE_OUTBOARD].deriv,
        [0.0, 0.0, 0.0],
    )
