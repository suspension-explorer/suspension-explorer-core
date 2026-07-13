"""
Analytical Jacobian row functions for kinematic constraints.

Each function computes one row of the Jacobian matrix: the partial derivatives
of a single scalar residual with respect to the coordinates of the involved
points. The solver assembles these rows into the full Jacobian in
`ResidualComputer.compute_jacobian`.

The CSE (common-subexpression elimination) expressions inside each function
body (t0, t1, ...) were produced by SymPy via `tools/generate_jacobians.py`.
To regenerate them (e.g. after changing a residual formula), run:

    uv run python tools/generate_jacobians.py

and paste the printed temporaries + return expression into the relevant
function below.

All norm-based residuals use `sqrt(s + EPS_SQ) - EPS` so that the
derivative remains finite when a constraint is exactly satisfied. The
`- EPS` bias correction is constant and vanishes under differentiation,
so the CSE expressions below only contain the `sqrt(s + EPS_SQ)` term.
See `kinematics.core.primitives.soft_math` for details.
"""

import math

import numpy as np

# The CSE expressions reference EPS_SQ by name, so we alias the public
# constant to keep the generated code untouched.
from kinematics.core.primitives.soft_math import EPS_SQ


# Residual: softnorm(|p2 - p1|^2) - L   (L is constant, drops out of the derivative)
def jac_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Jacobian row for DistanceConstraint / SphericalJointConstraint.

    Residual: `sqrt(|p2 - p1|^2) - L`.
    Returns `[dR/dp1_x, dR/dp1_y, dR/dp1_z, dR/dp2_x, dR/dp2_y, dR/dp2_z]`.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    t0 = x1 - x2
    t1 = -t0
    t2 = y1 - y2
    t3 = -t2
    t4 = z1 - z2
    t5 = -t4
    t6 = 1 / math.sqrt(EPS_SQ + t1**2 + t3**2 + t5**2)
    return np.array([t0 * t6, t2 * t6, t4 * t6, t1 * t6, t3 * t6, t5 * t6])


# Residual: atan2(|v1 × v2|, v1 · v2) - α   where v1 = p2 - p1, v2 = p4 - p3
def jac_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for AngleConstraint (4-point, two vectors).

    Points: v1_start(p1), v1_end(p2), v2_start(p3), v2_end(p4).
    Returns 12-element array of partials in point order.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    x4, y4, z4 = float(p4[0]), float(p4[1]), float(p4[2])
    t0 = x3 - x4
    t1 = x1 - x2
    t2 = -t1
    t3 = y3 - y4
    t4 = -t3
    t5 = -t0
    t6 = y1 - y2
    t7 = -t6
    t8 = t2 * t4 - t5 * t7
    t9 = z3 - z4
    t10 = -t9
    t11 = z1 - z2
    t12 = -t11
    t13 = -t10 * t2 + t12 * t5
    t14 = t10 * t7 - t12 * t4
    t15 = EPS_SQ + t13**2 + t14**2 + t8**2
    t16 = math.sqrt(t15)
    t17 = t10 * t12 + t2 * t5 + t4 * t7
    t18 = 1 / (t15 + t17**2)
    t19 = t16 * t18
    t20 = 2 * y3 - 2 * y4
    t21 = (1 / 2) * t8
    t22 = 2 * z3 - 2 * z4
    t23 = -t22
    t24 = (1 / 2) * t13
    t25 = 1 / t16
    t26 = 2 * x3 - 2 * x4
    t27 = -t26
    t28 = (1 / 2) * t14
    t29 = -t20
    t30 = 2 * y1 - 2 * y2
    t31 = -t30
    t32 = 2 * z1 - 2 * z2
    t33 = 2 * x1 - 2 * x2
    t34 = -t32
    t35 = -t33
    return np.array(
        [
            -t0 * t19 + t17 * t18 * t25 * (t20 * t21 + t23 * t24),
            t17 * t18 * t25 * (t21 * t27 + t22 * t28) - t19 * t3,
            t17 * t18 * t25 * (t24 * t26 + t28 * t29) - t19 * t9,
            t17 * t18 * t25 * (t21 * t29 + t22 * t24) - t19 * t5,
            t17 * t18 * t25 * (t21 * t26 + t23 * t28) - t19 * t4,
            -t10 * t19 + t17 * t18 * t25 * (t20 * t28 + t24 * t27),
            -t1 * t19 + t17 * t18 * t25 * (t21 * t31 + t24 * t32),
            t17 * t18 * t25 * (t21 * t33 + t28 * t34) - t19 * t6,
            -t11 * t19 + t17 * t18 * t25 * (t24 * t35 + t28 * t30),
            t17 * t18 * t25 * (t21 * t30 + t24 * t34) - t19 * t2,
            t17 * t18 * t25 * (t21 * t35 + t28 * t32) - t19 * t7,
            -t12 * t19 + t17 * t18 * t25 * (t24 * t33 + t28 * t31),
        ]
    )  # noqa: E501


# Residual: atan2(|v1 × v2|, v1 · v2) - α
# v1 = p1 - p2, v2 = p3 - p2 (vertex at p2)
def jac_three_point_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Jacobian row for ThreePointAngleConstraint (vertex at p2).

    Returns 9-element array: [dR/dp1, dR/dp2, dR/dp3].
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    t0 = -x2 + x3
    t1 = x1 - x2
    t2 = -y2 + y3
    t3 = y1 - y2
    t4 = -t0 * t3 + t1 * t2
    t5 = -z2 + z3
    t6 = z1 - z2
    t7 = t0 * t6 - t1 * t5
    t8 = -t2 * t6 + t3 * t5
    t9 = EPS_SQ + t4**2 + t7**2 + t8**2
    t10 = math.sqrt(t9)
    t11 = t0 * t1 + t2 * t3 + t5 * t6
    t12 = 1 / (t11**2 + t9)
    t13 = t10 * t12
    t14 = 2 * y2
    t15 = -2 * y3
    t16 = t14 + t15
    t17 = (1 / 2) * t4
    t18 = 2 * z2
    t19 = -2 * z3
    t20 = t18 + t19
    t21 = (1 / 2) * t7
    t22 = 1 / t10
    t23 = 2 * x2
    t24 = -2 * x3
    t25 = t23 + t24
    t26 = (1 / 2) * t8
    t27 = -t23
    t28 = 2 * y1
    t29 = t15 + t28
    t30 = 2 * z1
    t31 = t19 + t30
    t32 = t11 * t12 * t22
    t33 = -t14
    t34 = 2 * x1
    t35 = t24 + t34
    t36 = -t18
    t37 = t28 + t33
    t38 = t30 + t36
    t39 = t27 + t34
    return np.array(
        [
            -t0 * t13 + t11 * t12 * t22 * (-t16 * t17 + t20 * t21),
            t11 * t12 * t22 * (t17 * t25 - t20 * t26) - t13 * t2,
            t11 * t12 * t22 * (t16 * t26 - t21 * t25) - t13 * t5,
            -t13 * (-t27 - x1 - x3) + t32 * (t17 * t29 - t21 * t31),
            -t13 * (-t33 - y1 - y3) + t32 * (-t17 * t35 + t26 * t31),
            -t13 * (-t36 - z1 - z3) + t32 * (t21 * t35 - t26 * t29),
            -t1 * t13 + t11 * t12 * t22 * (-t17 * t37 + t21 * t38),
            t11 * t12 * t22 * (t17 * t39 - t26 * t38) - t13 * t3,
            t11 * t12 * t22 * (-t21 * t39 + t26 * t37) - t13 * t6,
        ]
    )  # noqa: E501


# Residual: |v1_hat × v2_hat| = |v1 × v2| / (|v1| · |v2|)
def jac_vectors_parallel(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for VectorsParallelConstraint.

    Residual: `|v1_hat x v2_hat|`.
    Points: v1_start(p1), v1_end(p2), v2_start(p3), v2_end(p4).
    Returns 12-element array.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    x4, y4, z4 = float(p4[0]), float(p4[1]), float(p4[2])
    t0 = x1 - x2
    t1 = -t0
    t2 = y1 - y2
    t3 = -t2
    t4 = z1 - z2
    t5 = -t4
    t6 = EPS_SQ + t1**2 + t3**2 + t5**2
    t7 = x3 - x4
    t8 = -t7
    t9 = y3 - y4
    t10 = -t9
    t11 = z3 - z4
    t12 = -t11
    t13 = EPS_SQ + t10**2 + t12**2 + t8**2
    t14 = 1 / math.sqrt(t13)
    t15 = t1 * t10 - t3 * t8
    t16 = -t1 * t12 + t5 * t8
    t17 = -t10 * t5 + t12 * t3
    t18 = math.sqrt(EPS_SQ + t15**2 + t16**2 + t17**2)
    t19 = t14 * t18 / t6 ** (3 / 2)
    t20 = 2 * y3 - 2 * y4
    t21 = (1 / 2) * t15
    t22 = 2 * z3 - 2 * z4
    t23 = -t22
    t24 = (1 / 2) * t16
    t25 = 1 / math.sqrt(t6)
    t26 = t14 * t25 / t18
    t27 = 2 * x3 - 2 * x4
    t28 = -t27
    t29 = (1 / 2) * t17
    t30 = -t20
    t31 = t18 * t25 / t13 ** (3 / 2)
    t32 = 2 * y1 - 2 * y2
    t33 = -t32
    t34 = 2 * z1 - 2 * z2
    t35 = 2 * x1 - 2 * x2
    t36 = -t34
    t37 = -t35
    return np.array(
        [
            t1 * t19 + t26 * (t20 * t21 + t23 * t24),
            t19 * t3 + t26 * (t21 * t28 + t22 * t29),
            t19 * t5 + t26 * (t24 * t27 + t29 * t30),
            t0 * t19 + t26 * (t21 * t30 + t22 * t24),
            t19 * t2 + t26 * (t21 * t27 + t23 * t29),
            t19 * t4 + t26 * (t20 * t29 + t24 * t28),
            t26 * (t21 * t33 + t24 * t34) + t31 * t8,
            t10 * t31 + t26 * (t21 * t35 + t29 * t36),
            t12 * t31 + t26 * (t24 * t37 + t29 * t32),
            t26 * (t21 * t32 + t24 * t36) + t31 * t7,
            t26 * (t21 * t37 + t29 * t34) + t31 * t9,
            t11 * t31 + t26 * (t24 * t35 + t29 * t33),
        ]
    )  # noqa: E501


# Residual: v1_hat · v2_hat = (v1 · v2) / (|v1| · |v2|)
def jac_vectors_perpendicular(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for VectorsPerpendicularConstraint.

    Residual: `v1_hat . v2_hat`.
    Points: v1_start(p1), v1_end(p2), v2_start(p3), v2_end(p4).
    Returns 12-element array.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    x4, y4, z4 = float(p4[0]), float(p4[1]), float(p4[2])
    t0 = x3 - x4
    t1 = x1 - x2
    t2 = -t1
    t3 = y1 - y2
    t4 = -t3
    t5 = z1 - z2
    t6 = -t5
    t7 = EPS_SQ + t2**2 + t4**2 + t6**2
    t8 = 1 / math.sqrt(t7)
    t9 = -t0
    t10 = y3 - y4
    t11 = -t10
    t12 = z3 - z4
    t13 = -t12
    t14 = EPS_SQ + t11**2 + t13**2 + t9**2
    t15 = 1 / math.sqrt(t14)
    t16 = t15 * t8
    t17 = t11 * t4 + t13 * t6 + t2 * t9
    t18 = t15 * t17 / t7 ** (3 / 2)
    t19 = t17 * t8 / t14 ** (3 / 2)
    return np.array(
        [
            t0 * t16 + t18 * t2,
            t10 * t16 + t18 * t4,
            t12 * t16 + t18 * t6,
            t1 * t18 + t16 * t9,
            t11 * t16 + t18 * t3,
            t13 * t16 + t18 * t5,
            t1 * t16 + t19 * t9,
            t11 * t19 + t16 * t3,
            t13 * t19 + t16 * t5,
            t0 * t19 + t16 * t2,
            t10 * t19 + t16 * t4,
            t12 * t19 + t16 * t6,
        ]
    )  # noqa: E501


# Residual: softnorm(|p1 - p2|^2) - softnorm(|p3 - p4|^2)
def jac_equal_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for EqualDistanceConstraint.

    Residual: `|p1-p2| - |p3-p4|`.
    Returns 12-element array.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    x4, y4, z4 = float(p4[0]), float(p4[1]), float(p4[2])
    t0 = x1 - x2
    t1 = -t0
    t2 = y1 - y2
    t3 = -t2
    t4 = z1 - z2
    t5 = -t4
    t6 = 1 / math.sqrt(EPS_SQ + t1**2 + t3**2 + t5**2)
    t7 = x3 - x4
    t8 = -t7
    t9 = y3 - y4
    t10 = -t9
    t11 = z3 - z4
    t12 = -t11
    t13 = 1 / math.sqrt(EPS_SQ + t10**2 + t12**2 + t8**2)
    return np.array(
        [
            t0 * t6,
            t2 * t6,
            t4 * t6,
            t1 * t6,
            t3 * t6,
            t5 * t6,
            -t13 * t7,
            -t13 * t9,
            -t11 * t13,
            -t13 * t8,
            -t10 * t13,
            -t12 * t13,
        ]
    )  # noqa: E501


# Residual: softnorm(|cross(p - line_point, line_direction)|^2)
# line_point and line_direction are fixed; only p has free coordinates.
def jac_point_on_line(
    p: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for PointOnLineConstraint.

    Residual: `|cross(p - line_point, line_direction)|`.
    Returns `[dR/dp_x, dR/dp_y, dR/dp_z]`.
    """
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    lpx = float(line_point[0])
    lpy = float(line_point[1])
    lpz = float(line_point[2])
    ldx = float(line_direction[0])
    ldy = float(line_direction[1])
    ldz = float(line_direction[2])
    t0 = -lpy + py
    t1 = -lpx + px
    t2 = -t0 * ldx + t1 * ldy
    t3 = -lpz + pz
    t4 = -t1 * ldz + t3 * ldx
    t5 = t0 * ldz - t3 * ldy
    t6 = 1 / math.sqrt(EPS_SQ + t2**2 + t4**2 + t5**2)
    return np.array(
        [
            t6 * (t2 * ldy - t4 * ldz),
            t6 * (-t2 * ldx + t5 * ldz),
            t6 * (t4 * ldx - t5 * ldy),
        ]
    )  # noqa: E501


# Residual: dot(p - plane_point, plane_normal)
# Linear residual — the Jacobian is simply the plane normal.
def jac_point_on_plane(
    p: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for PointOnPlaneConstraint.

    Residual: `dot(p - plane_point, plane_normal)`.
    Returns `[dR/dp_x, dR/dp_y, dR/dp_z]`  (= plane_normal components).
    """
    nx = float(plane_normal[0])
    ny = float(plane_normal[1])
    nz = float(plane_normal[2])
    return np.array([nx, ny, nz])


# Residual: v1 · (v2 × v3)   where vi = p(i+1) - p1  (scalar triple product)
def jac_coplanar(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """
    Jacobian row for CoplanarPointsConstraint.

    Residual: `v1 . (v2 x v3)` where vi = p(i+1) - p1.
    Returns 12-element array.
    """
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
    x3, y3, z3 = float(p3[0]), float(p3[1]), float(p3[2])
    x4, y4, z4 = float(p4[0]), float(p4[1]), float(p4[2])
    t0 = -y1 + y2
    t1 = -z4
    t2 = t1 + z3
    t3 = -y4
    t4 = t3 + y1
    t5 = -t4
    t6 = z1 - z3
    t7 = -t6
    t8 = t5 * t7
    t9 = t3 + y3
    t10 = -z1 + z2
    t11 = y1 - y3
    t12 = -t11
    t13 = t1 + z1
    t14 = -t13
    t15 = t12 * t14
    t16 = -x1 + x2
    t17 = -x4
    t18 = t17 + x3
    t19 = x1 - x3
    t20 = -t19
    t21 = t17 + x1
    t22 = -t21
    t23 = t14 * t20 - t22 * t7
    t24 = t12 * t22
    t25 = t20 * t5
    return np.array(
        [
            -t0 * t2 + t10 * t9 - t15 + t8,
            -t10 * t18 + t16 * t2 + t23,
            t0 * t18 - t16 * t9 + t24 - t25,
            t15 - t8,
            -t23,
            -t24 + t25,
            t0 * t13 + t10 * t5,
            t10 * t21 + t14 * t16,
            t0 * t22 + t16 * t4,
            t0 * t7 + t10 * t11,
            t10 * t20 + t16 * t6,
            t0 * t19 + t12 * t16,
        ]
    )  # noqa: E501
