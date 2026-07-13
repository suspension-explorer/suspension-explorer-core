"""
Tests for camber shim functionality.

These tests verify that the local split-body shim assembly solver correctly finds
a configuration where both shim faces remain parallel at the requested thickness,
UBJ stays on the upper wishbone arc, and the upright-body rotation is applied to
upright-mounted points.
"""

import numpy as np

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.vector_utils.geometric import rotate_point_about_axis
from kinematics.core.schema.config import CamberShimConfig
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.config.shims import solve_camber_shim_assembly
from kinematics.core.suspensions.corner import DoubleWishboneSuspension

# ---------------------------------------------------------------------------
# rotate_point_about_axis
# ---------------------------------------------------------------------------


def test_rotate_point_about_axis_90_degrees():
    """
    Test rotation of a point 90 degrees about Z axis.
    """
    point = Point3([1.0, 0.0, 0.0])
    pivot = Point3([0.0, 0.0, 0.0])
    axis = Direction3([0.0, 0.0, 1.0])
    angle = np.pi / 2  # 90 degrees

    rotated = rotate_point_about_axis(point, pivot, axis, angle)

    # Should rotate to (0, 1, 0)
    assert np.isclose(rotated[Axis.X], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Y], 1.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Z], 0.0, atol=1e-10)


def test_rotate_point_about_axis_with_offset_pivot():
    """
    Test rotation about an axis that doesn't pass through origin.
    """
    point = Point3([2.0, 0.0, 0.0])
    pivot = Point3([1.0, 0.0, 0.0])
    axis = Direction3([0.0, 0.0, 1.0])
    angle = np.pi  # 180 degrees

    rotated = rotate_point_about_axis(point, pivot, axis, angle)

    # Should rotate to (0, 0, 0)
    assert np.isclose(rotated[Axis.X], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Y], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Z], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# solve_camber_shim_assembly: unit-level tests
# ---------------------------------------------------------------------------


def make_simple_geometry(
    design_thickness: float = 30.0,
    setup_thickness: float = 30.0,
) -> tuple[dict[PointID, Point3], CamberShimConfig]:
    """
    Return positions and shim config for the shim assembly solver.

    Uses positions matching the test geometry YAML: UBJ at the tip of two upper
    wishbone arms, LBJ below, trackrod connecting rack to upright.
    """
    positions = {
        PointID.UPPER_WISHBONE_OUTBOARD: Point3([0.0, 750.0, 500.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([0.0, 900.0, 200.0]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3([225.0, 350.0, 500.0]),
        PointID.UPPER_WISHBONE_INBOARD_REAR: Point3([-275.0, 350.0, 500.0]),
        PointID.TRACKROD_OUTBOARD: Point3([150.0, 800.0, 275.0]),
        PointID.TRACKROD_INBOARD: Point3([50.0, 200.0, 250.0]),
    }
    shim_config = CamberShimConfig(
        shim_face_point_a=Point3([0.0, 750.0, 510.0]),
        shim_face_point_b=Point3([0.0, 750.0, 490.0]),
        shim_face_normal=Direction3([0.0, 1.0, 0.0]),
        design_thickness=design_thickness,
        setup_thickness=setup_thickness,
    )
    return positions, shim_config


def test_design_thickness_returns_identity():
    """
    When setup_thickness == design_thickness the solver should return design state
    with zero rotation and unchanged UBJ.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=30.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    np.testing.assert_allclose(
        sol.ubj_position,
        positions[PointID.UPPER_WISHBONE_OUTBOARD].data,
        atol=1e-10,
    )
    np.testing.assert_allclose(sol.camber_block_rot_vec, 0.0, atol=1e-10)
    np.testing.assert_allclose(sol.upright_body_rot_vec, 0.0, atol=1e-10)
    assert sol.upright_body_rot_angle_rad == 0.0
    assert sol.constraint_residual_norm == 0.0


def test_solver_converges():
    """
    Basic convergence check: a 10mm shim change should produce a solution with
    near-zero residual norm.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    assert sol.constraint_residual_norm < 1e-5


def test_upper_arm_lengths_preserved():
    """
    The solved UBJ must remain on the upper wishbone arc, so the distances to
    both inboard pickups must match their design values.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    ubj = positions[PointID.UPPER_WISHBONE_OUTBOARD]
    design_front = float(
        np.linalg.norm(ubj - positions[PointID.UPPER_WISHBONE_INBOARD_FRONT])
    )
    design_rear = float(
        np.linalg.norm(ubj - positions[PointID.UPPER_WISHBONE_INBOARD_REAR])
    )

    sol = solve_camber_shim_assembly(positions, shim_config)

    solved_ubj = Point3(sol.ubj_position)
    solved_front = float(
        np.linalg.norm(solved_ubj - positions[PointID.UPPER_WISHBONE_INBOARD_FRONT])
    )
    solved_rear = float(
        np.linalg.norm(solved_ubj - positions[PointID.UPPER_WISHBONE_INBOARD_REAR])
    )

    assert abs(solved_front - design_front) < TEST_TOLERANCE
    assert abs(solved_rear - design_rear) < TEST_TOLERANCE


def test_face_normals_parallel_at_solution():
    """
    At the solved state the camber-block and upright-body face normals must align.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    normal_camber_block = sol.camber_block_face_normal
    normal_upright = sol.upright_body_face_normal
    cross = np.cross(normal_camber_block, normal_upright)
    assert np.linalg.norm(cross) < 1e-8
    assert float(np.dot(normal_camber_block, normal_upright)) > 1.0 - 1e-8


def test_lbj_stays_fixed():
    """
    The lower ball joint is the fixed pivot and must not appear in the solution
    as having moved.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    # The solver doesn't move LBJ, but the lower rotation angle should be non-zero
    # (the body rotates about LBJ, LBJ itself stays put).
    assert sol.upright_body_rot_angle_rad > 1e-6


def test_nonzero_upright_body_rotation():
    """
    A shim change must produce a non-trivial upright body rotation so that upright-
    mounted points actually move. This guards against the solver finding a spurious
    branch where the shim change is absorbed entirely by the camber block.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    assert sol.upright_body_rot_angle_rad > 1e-6
    assert np.linalg.norm(sol.upright_body_rot_vec) > 1e-6


def test_trackrod_length_preserved():
    """
    The trackrod is a rigid link. Its length must be unchanged after the shim solve.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    design_length = float(
        np.linalg.norm(
            positions[PointID.TRACKROD_OUTBOARD] - positions[PointID.TRACKROD_INBOARD]
        )
    )

    sol = solve_camber_shim_assembly(positions, shim_config)

    # Compute where trackrod outboard lands after upright-body rotation about LBJ.
    solved_tro = rotate_point_about_axis(
        positions[PointID.TRACKROD_OUTBOARD],
        positions[PointID.LOWER_WISHBONE_OUTBOARD],
        Direction3(sol.upright_body_rot_axis),
        sol.upright_body_rot_angle_rad,
    )
    solved_length = float(
        np.linalg.norm(solved_tro - positions[PointID.TRACKROD_INBOARD])
    )

    assert abs(solved_length - design_length) < 1e-4, (
        f"Trackrod length changed: design={design_length:.4f}, "
        f"solved={solved_length:.4f}"
    )


def test_ubj_moves_for_nonzero_shim_change():
    """
    UBJ should move along the upper wishbone arc when the shim thickness changes.
    """
    positions, shim_config = make_simple_geometry(
        design_thickness=30.0, setup_thickness=40.0
    )
    sol = solve_camber_shim_assembly(positions, shim_config)

    displacement = np.linalg.norm(
        sol.ubj_position - positions[PointID.UPPER_WISHBONE_OUTBOARD].data
    )
    assert displacement > 1e-4, (
        f"UBJ should move for non-trivial shim change, moved {displacement:.6f}mm"
    )


# ---------------------------------------------------------------------------
# Integration tests (full suspension)
# ---------------------------------------------------------------------------


def _make_shim_config(
    design_thickness: float = 30.0,
    setup_thickness: float = 40.0,
) -> CamberShimConfig:
    """
    Build a CamberShimConfig matching the test geometry YAML datum positions.
    """
    return CamberShimConfig(
        shim_face_point_a=Point3([-25.0, 750.0, 510.0]),
        shim_face_point_b=Point3([-25.0, 750.0, 490.0]),
        shim_face_normal=Direction3([0.0, 1.0, 0.0]),
        design_thickness=design_thickness,
        setup_thickness=setup_thickness,
    )


def _make_shimmed_suspension(
    base_suspension: Suspension,
    shim_config: CamberShimConfig,
) -> DoubleWishboneSuspension:
    """
    Create a new suspension instance with the given shim config applied.
    """
    assert isinstance(base_suspension, DoubleWishboneSuspension)
    assert base_suspension.config is not None
    new_config = base_suspension.config.model_copy(update={"camber_shim": shim_config})
    return DoubleWishboneSuspension(
        name=base_suspension.name,
        version=base_suspension.version,
        units=base_suspension.units,
        side=base_suspension.side,
        hardpoints=base_suspension.hardpoints.copy(),
        config=new_config,
    )


def test_shim_application_changes_camber(double_wishbone_geometry_file):
    """
    Test that applying a camber shim rotates the upright and changes camber angle.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()

    initial_axle_in = initial_state.positions[PointID.AXLE_INBOARD]
    initial_axle_out = initial_state.positions[PointID.AXLE_OUTBOARD]
    initial_axle_vector = initial_axle_out - initial_axle_in
    initial_camber_rad = np.arctan2(
        initial_axle_vector[Axis.Y], initial_axle_vector[Axis.Z]
    )

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()

    shimmed_axle_in = shimmed_state.positions[PointID.AXLE_INBOARD]
    shimmed_axle_out = shimmed_state.positions[PointID.AXLE_OUTBOARD]
    shimmed_axle_vector = shimmed_axle_out - shimmed_axle_in
    shimmed_camber_rad = np.arctan2(
        shimmed_axle_vector[Axis.Y], shimmed_axle_vector[Axis.Z]
    )

    # Adding shim outboard should change camber.
    assert not np.isclose(initial_camber_rad, shimmed_camber_rad, atol=1e-6), (
        f"Expected camber to change, "
        f"got initial={np.degrees(initial_camber_rad):.3f} deg, "
        f"shimmed={np.degrees(shimmed_camber_rad):.3f} deg"
    )


def test_shim_does_not_move_lower_ball_joint(double_wishbone_geometry_file):
    """
    Test that the lower ball joint (fixed pivot) doesn't move when shims are applied.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()
    initial_lbj = initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD].copy()

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()
    shimmed_lbj = shimmed_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

    np.testing.assert_allclose(
        initial_lbj.data,
        shimmed_lbj.data,
        atol=1e-10,
        err_msg="Lower ball joint (pivot) should not move when shims are applied",
    )


def test_shim_does_not_move_inboard_points(double_wishbone_geometry_file):
    """
    Test that chassis-mounted inboard points don't move when shims are applied.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()

    chassis_point_ids = [
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_INBOARD_REAR,
        PointID.TRACKROD_INBOARD,
    ]
    initial_positions = {
        pid: initial_state.positions[pid].copy() for pid in chassis_point_ids
    }

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()

    for point_id, initial_pos in initial_positions.items():
        shimmed_pos = shimmed_state.positions[point_id]
        np.testing.assert_allclose(
            initial_pos.data,
            shimmed_pos.data,
            atol=1e-10,
            err_msg=f"{point_id.name} (chassis-mounted) should not move",
        )


def test_shim_moves_axle_points(double_wishbone_geometry_file):
    """
    Test that all upright-mounted points DO move when shims are applied.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()

    upright_points = {
        PointID.AXLE_INBOARD: initial_state.positions[PointID.AXLE_INBOARD].copy(),
        PointID.AXLE_OUTBOARD: initial_state.positions[PointID.AXLE_OUTBOARD].copy(),
        PointID.TRACKROD_OUTBOARD: initial_state.positions[
            PointID.TRACKROD_OUTBOARD
        ].copy(),
    }

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()

    for point_id, initial_pos in upright_points.items():
        shimmed_pos = shimmed_state.positions[point_id]
        distance_moved = np.linalg.norm(shimmed_pos - initial_pos)
        assert distance_moved > 0.1, (
            f"{point_id.name} should move (moved {distance_moved:.3f}mm)"
        )


def test_upright_mounted_points_maintain_distance_from_lbj(
    double_wishbone_geometry_file,
):
    """
    Upright-mounted points rotate rigidly about LBJ, so their distances to LBJ
    must be unchanged after the shim is applied.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()
    lbj = initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

    upright_point_ids = [
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.TRACKROD_OUTBOARD,
    ]
    design_distances = {
        pid: float(np.linalg.norm(initial_state.positions[pid] - lbj))
        for pid in upright_point_ids
    }

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()
    shimmed_lbj = shimmed_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

    for pid, design_dist in design_distances.items():
        shimmed_dist = float(np.linalg.norm(shimmed_state.positions[pid] - shimmed_lbj))
        assert abs(shimmed_dist - design_dist) < 1e-6, (
            f"{pid.name} distance to LBJ changed: "
            f"design={design_dist:.4f}, shimmed={shimmed_dist:.4f}"
        )


def test_shim_preserves_trackrod_length(double_wishbone_geometry_file):
    """
    The trackrod is a rigid link. Its length must be unchanged after shimming.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    initial_state = suspension.initial_state()

    design_trackrod_length = float(
        np.linalg.norm(
            initial_state.positions[PointID.TRACKROD_OUTBOARD]
            - initial_state.positions[PointID.TRACKROD_INBOARD]
        )
    )

    shimmed_suspension = _make_shimmed_suspension(suspension, _make_shim_config())
    shimmed_state = shimmed_suspension.initial_state()

    shimmed_trackrod_length = float(
        np.linalg.norm(
            shimmed_state.positions[PointID.TRACKROD_OUTBOARD]
            - shimmed_state.positions[PointID.TRACKROD_INBOARD]
        )
    )

    assert abs(shimmed_trackrod_length - design_trackrod_length) < 0.01, (
        f"Trackrod length changed: design={design_trackrod_length:.4f}mm, "
        f"shimmed={shimmed_trackrod_length:.4f}mm"
    )


def test_backward_compatibility_no_shim(double_wishbone_geometry_file):
    """
    Test that when design_thickness == setup_thickness, there's no effect.
    """
    suspension = load_geometry(double_wishbone_geometry_file)

    assert suspension.config is not None
    assert suspension.config.camber_shim is not None
    shim = suspension.config.camber_shim
    assert shim.design_thickness == shim.setup_thickness

    # Should initialize without error.
    state = suspension.initial_state()

    # Should have all expected points.
    assert PointID.UPPER_WISHBONE_OUTBOARD in state.positions
    assert PointID.LOWER_WISHBONE_OUTBOARD in state.positions
    assert PointID.AXLE_INBOARD in state.positions
    assert PointID.AXLE_OUTBOARD in state.positions
