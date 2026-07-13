"""Validate analytical Jacobian functions against central finite differences.

Each Jacobian function is tested by comparing its output to numerical
derivatives computed via central differences:

    dR/dx_i ≈ (R(x + h*e_i) - R(x - h*e_i)) / (2h)
"""

import math

import numpy as np
import pytest

from kinematics.core.constraints import DistanceConstraint, FixedAxisConstraint
from kinematics.core.jacobians import (
    jac_angle,
    jac_coplanar,
    jac_distance,
    jac_equal_distance,
    jac_point_on_line,
    jac_point_on_plane,
    jac_three_point_angle,
    jac_vectors_parallel,
    jac_vectors_perpendicular,
)
from kinematics.core.primitives.enums import Axis, PointID

# Central-difference step size.  Small enough for accuracy, large enough to
# avoid cancellation error with float64.
STEP_SIZE = 1e-7

# Tolerance for analytical-vs-numerical comparison.  Central differences have
# O(h²) truncation error ≈ 1e-14, but accumulated rounding pushes it up.
TOLERANCE = 1e-6

P1 = PointID.LOWER_WISHBONE_INBOARD_FRONT
P2 = PointID.LOWER_WISHBONE_INBOARD_REAR
P3 = PointID.LOWER_WISHBONE_OUTBOARD
P4 = PointID.UPPER_WISHBONE_INBOARD_FRONT


def numerical_jacobian_row(
    residual_fn,
    positions: dict[PointID, np.ndarray],
    free_point_ids: list[PointID],
) -> np.ndarray:
    """
    Compute numerical Jacobian row via central finite differences.

    Args:
        residual_fn: Callable(positions_dict) -> float (scalar residual).
        positions: Full positions dict (will be mutated and restored).
        free_point_ids: Ordered list of points whose coordinates are varied.

    Returns:
        1-D array of partial derivatives, 3 per free point, in the order
        given by *free_point_ids*.
    """
    n = len(free_point_ids) * 3
    jac_row = np.empty(n)

    for k, pid in enumerate(free_point_ids):
        original = positions[pid].copy()
        for axis in range(3):
            positions[pid] = original.copy()
            positions[pid][axis] += STEP_SIZE
            r_plus = residual_fn(positions)

            positions[pid] = original.copy()
            positions[pid][axis] -= STEP_SIZE
            r_minus = residual_fn(positions)

            jac_row[3 * k + axis] = (r_plus - r_minus) / (2.0 * STEP_SIZE)

        positions[pid] = original

    return jac_row


# Non-aligned, non-degenerate geometry so that all components of the
# derivatives are exercised.
def sample_positions() -> dict[PointID, np.ndarray]:
    """Non-trivial positions for four points in general position."""
    return {
        P1: np.array([1.0, 2.0, 3.0]),
        P2: np.array([4.0, 6.0, 5.0]),
        P3: np.array([7.0, 1.0, 9.0]),
        P4: np.array([2.0, 8.0, 4.0]),
    }


class TestJacDistance:
    """Analytical vs. numerical derivatives for jac_distance."""

    def test_matches_numerical(self):
        """Central-difference check at a generic operating point."""
        pos = sample_positions()
        analytical = jac_distance(pos[P1], pos[P2])

        def residual(p):
            return float(np.linalg.norm(p[P2] - p[P1]))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_antisymmetry(self):
        """dR/dp1 = -dR/dp2 for the distance constraint."""
        pos = sample_positions()
        j = jac_distance(pos[P1], pos[P2])
        np.testing.assert_allclose(j[:3], -j[3:], atol=1e-14)

    def test_unit_magnitude(self):
        """Each 3-block should be a unit vector (since d(||v||)/dv = v/||v||)."""
        pos = sample_positions()
        j = jac_distance(pos[P1], pos[P2])
        assert math.isclose(np.linalg.norm(j[:3]), 1.0, abs_tol=1e-14)
        assert math.isclose(np.linalg.norm(j[3:]), 1.0, abs_tol=1e-14)

    @pytest.mark.parametrize(
        "p1, p2",
        [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 5.0, 0.0])),
            (np.array([1.0, 1.0, 1.0]), np.array([4.0, 5.0, 6.0])),
            (np.array([10.0, -3.0, 7.0]), np.array([-2.0, 4.0, 1.0])),
        ],
    )
    def test_multiple_configurations(self, p1, p2):
        """Parametric check across several configurations."""
        pos = {P1: p1.copy(), P2: p2.copy()}
        analytical = jac_distance(pos[P1], pos[P2])

        def residual(p):
            return float(np.linalg.norm(p[P2] - p[P1]))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_spherical_joint_same_jacobian(self):
        """SphericalJoint uses the same derivative as Distance (target L=0)."""
        pos = sample_positions()
        j_dist = jac_distance(pos[P1], pos[P2])

        def residual(p):
            return float(np.linalg.norm(p[P2] - p[P1]))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2])
        np.testing.assert_allclose(j_dist, numerical, atol=TOLERANCE)


class TestJacAngle:
    """Analytical vs. numerical derivatives for jac_angle."""

    def test_matches_numerical(self):
        """Central-difference check for the 4-point angle constraint."""
        pos = sample_positions()
        analytical = jac_angle(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P4] - p[P3]
            cross_mag = np.linalg.norm(np.cross(v1, v2))
            dot = np.dot(v1, v2)
            return float(np.arctan2(cross_mag, dot))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_negation_symmetry(self):
        """dR/d(v1_start) = -dR/d(v1_end) and likewise for v2.

        This holds because the angle depends on (p2-p1) and (p4-p3), so the
        partials w.r.t. each pair's start and end are negatives of each other.
        """
        pos = sample_positions()
        j = jac_angle(pos[P1], pos[P2], pos[P3], pos[P4])
        np.testing.assert_allclose(j[0:3], -j[3:6], atol=1e-12)
        np.testing.assert_allclose(j[6:9], -j[9:12], atol=1e-12)

    @pytest.mark.parametrize("angle_deg", [30, 45, 60, 90, 120, 150])
    def test_various_angles(self, angle_deg):
        """Check at several known angles (vectors in XY plane)."""
        angle_rad = math.radians(angle_deg)
        pos = {
            P1: np.array([0.0, 0.0, 0.0]),
            P2: np.array([1.0, 0.0, 0.0]),
            P3: np.array([0.0, 0.0, 0.0]),
            P4: np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0]),
        }
        analytical = jac_angle(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P4] - p[P3]
            cross_mag = np.linalg.norm(np.cross(v1, v2))
            dot = np.dot(v1, v2)
            return float(np.arctan2(cross_mag, dot))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)


class TestJacThreePointAngle:
    """Analytical vs. numerical derivatives for jac_three_point_angle."""

    def test_matches_numerical(self):
        """Central-difference check for the 3-point angle constraint."""
        pos = sample_positions()
        analytical = jac_three_point_angle(pos[P1], pos[P2], pos[P3])

        def residual(p):
            v1 = p[P1] - p[P2]
            v2 = p[P3] - p[P2]
            cross_mag = np.linalg.norm(np.cross(v1, v2))
            dot = np.dot(v1, v2)
            return float(np.arctan2(cross_mag, dot))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_vertex_derivatives_sum(self):
        """Translating all three points by the same vector doesn't change the
        angle, so (dR/dp1 + dR/dp2 + dR/dp3) = 0 for each axis.
        """
        pos = sample_positions()
        j = jac_three_point_angle(pos[P1], pos[P2], pos[P3])
        for axis in range(3):
            total = j[axis] + j[3 + axis] + j[6 + axis]
            assert abs(total) < 1e-12, (
                f"Translation invariance violated on axis {axis}: {total}"
            )


class TestJacVectorsParallel:
    """Analytical vs. numerical derivatives for jac_vectors_parallel."""

    def test_matches_numerical(self):
        """Central-difference check for the parallel-vectors constraint."""
        pos = sample_positions()
        analytical = jac_vectors_parallel(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P4] - p[P3]
            v1n = v1 / np.linalg.norm(v1)
            v2n = v2 / np.linalg.norm(v2)
            return float(np.linalg.norm(np.cross(v1n, v2n)))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_negation_symmetry(self):
        """dR/d(start) = -dR/d(end) for each vector pair."""
        pos = sample_positions()
        j = jac_vectors_parallel(pos[P1], pos[P2], pos[P3], pos[P4])
        np.testing.assert_allclose(j[0:3], -j[3:6], atol=1e-12)
        np.testing.assert_allclose(j[6:9], -j[9:12], atol=1e-12)


class TestJacVectorsPerpendicular:
    """Analytical vs. numerical derivatives for jac_vectors_perpendicular."""

    def test_matches_numerical(self):
        """Central-difference check for perpendicular constraint."""
        pos = sample_positions()
        analytical = jac_vectors_perpendicular(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P4] - p[P3]
            v1n = v1 / np.linalg.norm(v1)
            v2n = v2 / np.linalg.norm(v2)
            return float(np.dot(v1n, v2n))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_negation_symmetry(self):
        """dR/d(start) = -dR/d(end) for each vector pair."""
        pos = sample_positions()
        j = jac_vectors_perpendicular(pos[P1], pos[P2], pos[P3], pos[P4])
        np.testing.assert_allclose(j[0:3], -j[3:6], atol=1e-12)
        np.testing.assert_allclose(j[6:9], -j[9:12], atol=1e-12)


class TestJacEqualDistance:
    """Analytical vs. numerical derivatives for jac_equal_distance."""

    def test_matches_numerical(self):
        """Central-difference check for equal-distance constraint."""
        pos = sample_positions()
        analytical = jac_equal_distance(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            d1 = float(np.linalg.norm(p[P2] - p[P1]))
            d2 = float(np.linalg.norm(p[P4] - p[P3]))
            return d1 - d2

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_structure(self):
        """First 6 entries = jac_distance(p1,p2), last 6 = -jac_distance(p3,p4)."""
        pos = sample_positions()
        j_eq = jac_equal_distance(pos[P1], pos[P2], pos[P3], pos[P4])
        j_d1 = jac_distance(pos[P1], pos[P2])
        j_d2 = jac_distance(pos[P3], pos[P4])
        np.testing.assert_allclose(j_eq[:6], j_d1, atol=1e-14)
        np.testing.assert_allclose(j_eq[6:], -j_d2, atol=1e-14)


class TestJacPointOnLine:
    """Analytical vs. numerical derivatives for jac_point_on_line."""

    def test_matches_numerical(self):
        """Central-difference check for point-on-line constraint."""
        line_point = np.array([1.0, 2.0, 3.0])
        line_dir = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)

        # Point NOT on the line (to avoid the singularity).
        pos = {P1: np.array([5.0, 3.0, 1.0])}
        analytical = jac_point_on_line(pos[P1], line_point, line_dir)

        def residual(p):
            w = p[P1] - line_point
            cross = np.cross(w, line_dir)
            return float(np.linalg.norm(cross))

        numerical = numerical_jacobian_row(residual, pos, [P1])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    @pytest.mark.parametrize(
        "line_dir",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
            np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0),
        ],
    )
    def test_various_line_directions(self, line_dir):
        """Check derivatives for several line orientations."""
        line_point = np.array([0.0, 0.0, 0.0])
        pos = {P1: np.array([3.0, -2.0, 5.0])}
        analytical = jac_point_on_line(pos[P1], line_point, line_dir)

        def residual(p):
            w = p[P1] - line_point
            cross = np.cross(w, line_dir)
            return float(np.linalg.norm(cross))

        numerical = numerical_jacobian_row(residual, pos, [P1])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)


class TestJacPointOnPlane:
    """Analytical vs. numerical derivatives for jac_point_on_plane."""

    def test_matches_numerical(self):
        """Central-difference check for point-on-plane constraint."""
        plane_point = np.array([0.0, 0.0, 5.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        pos = {P1: np.array([3.0, -2.0, 7.0])}

        analytical = jac_point_on_plane(pos[P1], plane_point, plane_normal)

        def residual(p):
            return float(np.dot(p[P1] - plane_point, plane_normal))

        numerical = numerical_jacobian_row(residual, pos, [P1])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_derivatives_are_normal_components(self):
        """The derivatives should exactly equal the plane normal."""
        plane_point = np.array([1.0, 2.0, 3.0])
        normal = np.array([0.3, 0.4, 0.5]) / np.linalg.norm([0.3, 0.4, 0.5])
        pos = {P1: np.array([10.0, 20.0, 30.0])}

        j = jac_point_on_plane(pos[P1], plane_point, normal)
        np.testing.assert_allclose(j, normal, atol=1e-14)


class TestJacCoplanar:
    """Analytical vs. numerical derivatives for jac_coplanar."""

    def test_matches_numerical(self):
        """Central-difference check for coplanarity constraint."""
        pos = sample_positions()
        analytical = jac_coplanar(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P3] - p[P1]
            v3 = p[P4] - p[P1]
            return float(np.dot(v1, np.cross(v2, v3)))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_translation_invariance(self):
        """Translating all 4 points by the same vector doesn't change the
        triple product, so the sum of all partials per axis should be zero.
        """
        pos = sample_positions()
        j = jac_coplanar(pos[P1], pos[P2], pos[P3], pos[P4])
        for axis in range(3):
            total = j[axis] + j[3 + axis] + j[6 + axis] + j[9 + axis]
            assert abs(total) < 1e-10, (
                f"Translation invariance violated on axis {axis}: {total}"
            )

    @pytest.mark.parametrize("scale", [0.01, 1.0, 100.0, 10000.0])
    def test_scale_independence(self, scale):
        """Check that derivatives track correctly across scales."""
        pos = {k: v * scale for k, v in sample_positions().items()}
        analytical = jac_coplanar(pos[P1], pos[P2], pos[P3], pos[P4])

        def residual(p):
            v1 = p[P2] - p[P1]
            v2 = p[P3] - p[P1]
            v3 = p[P4] - p[P1]
            return float(np.dot(v1, np.cross(v2, v3)))

        numerical = numerical_jacobian_row(residual, pos, [P1, P2, P3, P4])
        np.testing.assert_allclose(
            analytical, numerical, atol=TOLERANCE * scale**2, rtol=1e-5
        )


class TestFullJacobianAssembly:
    """End-to-end test: assemble the full Jacobian for a small constrained
    system and compare against central-difference evaluation of the full
    residual vector.
    """

    def test_jacobian_matches_numerical_for_solver_system(self):
        from kinematics.core.points.derived.manager import (
            DerivedPointsManager,
            DerivedPointsSpec,
        )
        from kinematics.core.primitives.geometry import Point3
        from kinematics.core.solver import ResidualComputer
        from kinematics.core.state import SuspensionState
        from kinematics.core.targeting import (
            PointTarget,
            PointTargetAxis,
            TargetPositionMode,
        )

        positions = {
            P1: Point3([0.0, 0.0, 0.0]),
            P2: Point3([3.0, 4.0, 0.0]),
            P3: Point3([1.0, 5.0, 2.0]),
        }
        free_points = {P1, P2, P3}
        state = SuspensionState(
            positions={k: v.copy() for k, v in positions.items()},
            free_points=free_points,
        )

        constraints: list = [
            DistanceConstraint(P1, P2, target_distance=5.0),
            DistanceConstraint(P2, P3, target_distance=3.5),
            FixedAxisConstraint(P1, Axis.Z, value=0.0),
        ]

        targets = [
            PointTarget(
                point_id=P1,
                direction=PointTargetAxis(axis=Axis.X),
                value=0.0,
                mode=TargetPositionMode.ABSOLUTE,
            ),
        ]

        derived_spec = DerivedPointsSpec(functions={}, dependencies={})
        derived_mgr = DerivedPointsManager(derived_spec)

        rc = ResidualComputer(
            constraints=constraints,
            derived_manager=derived_mgr,
            state_buffer=state,
            n_target_variables=len(targets),
        )

        x0 = state.get_free_array()

        analytical = rc.compute_jacobian(x0, targets)

        n_res = analytical.shape[0]
        n_vars = len(x0)
        numerical = np.zeros((n_res, n_vars))

        for j in range(n_vars):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[j] += STEP_SIZE
            x_minus[j] -= STEP_SIZE

            r_plus = rc.compute(x_plus, targets)
            r_minus = rc.compute(x_minus, targets)
            numerical[:, j] = (r_plus - r_minus) / (2.0 * STEP_SIZE)

        np.testing.assert_allclose(analytical, numerical, atol=TOLERANCE)

    def test_residual_computer_rejects_target_count_changes(self):
        from kinematics.core.points.derived.manager import (
            DerivedPointsManager,
            DerivedPointsSpec,
        )
        from kinematics.core.primitives.geometry import Point3
        from kinematics.core.solver import ResidualComputer
        from kinematics.core.state import SuspensionState
        from kinematics.core.targeting import (
            PointTarget,
            PointTargetAxis,
            TargetPositionMode,
        )

        positions = {
            P1: Point3([0.0, 0.0, 0.0]),
            P2: Point3([3.0, 4.0, 0.0]),
        }
        state = SuspensionState(
            positions={k: v.copy() for k, v in positions.items()},
            free_points={P1, P2},
        )

        derived_spec = DerivedPointsSpec(functions={}, dependencies={})
        derived_mgr = DerivedPointsManager(derived_spec)
        rc = ResidualComputer(
            constraints=[DistanceConstraint(P1, P2, target_distance=5.0)],
            derived_manager=derived_mgr,
            state_buffer=state,
            n_target_variables=1,
        )

        x0 = state.get_free_array()
        bad_targets = [
            PointTarget(
                point_id=P1,
                direction=PointTargetAxis(axis=Axis.X),
                value=0.0,
                mode=TargetPositionMode.ABSOLUTE,
            ),
            PointTarget(
                point_id=P2,
                direction=PointTargetAxis(axis=Axis.Y),
                value=0.0,
                mode=TargetPositionMode.ABSOLUTE,
            ),
        ]

        with pytest.raises(ValueError, match="fixed number of targets"):
            rc.compute(x0, bad_targets)

        with pytest.raises(ValueError, match="fixed number of targets"):
            rc.compute_jacobian(x0, bad_targets)
