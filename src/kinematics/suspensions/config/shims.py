"""
Camber shim geometry calculations.

This module solves suspension pose for the specified split body (outboard) camber shim.
The camber block rotates about the upper ball joint, the upright body rotates
about the lower ball joint, and the two shim faces remain separated by the
requested setup thickness.

The solver uses a 7 variable overdetermined least-squares formulation:
    - wishbone_angle (1): Rotation angle of UBJ about the upper wishbone axis.
      UBJ position is computed exactly on the wishbone arc by construction.
    - camber_block_rotvec (3): Rotation vector for the camber block about UBJ.
    - upright_body_rotvec (3): Rotation vector for the upright body about LBJ.

With 10 residuals:
    - 3 scalar datum A closure
      (upright body face A - camber block face A = thickness * normal).
    - 3 scalar datum B closure
      (upright body face B - camber block face B = thickness * normal).
    - 3 scalar normal alignment
      (camber block and upright body face normals must match).
    - 1 scalar trackrod length (preserves design trackrod length through shim change)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.core.constants import EPS_GEOMETRIC, EPS_NUMERICAL
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3, Vector3, extract_array
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.core.vector_utils.geometric import rotate_vector_rodrigues
from kinematics.solver import SolverConfig, solve_least_squares_problem
from kinematics.suspensions.config.settings import CamberShimConfig

CAMBER_SHIM_N_VARS = 7
CAMBER_SHIM_N_RESIDUALS = 10


def _rotate_array_rodrigues(v: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """
    Thin wrapper: rotate raw numpy arrays via Vector3-based Rodrigues.

    This avoids Vector3 wrapping at every call site in the residual function.
    """
    return rotate_vector_rodrigues(Vector3(v), Vector3(rotvec)).data


@dataclass(frozen=True)
class CamberShimAssemblySolution:
    """
    Solved split-body shim assembly state.

    The local assembly solve determines how both the camber block and upright
    body rotate to accommodate the setup shim thickness. The suspension
    integration consumes the solved UBJ position and upright body rotation to update
    the global suspension state.
    """

    ubj_position: np.ndarray
    camber_block_rot_vec: np.ndarray
    upright_body_rot_vec: np.ndarray
    camber_block_face_normal: np.ndarray
    upright_body_face_normal: np.ndarray
    upright_body_rot_axis: np.ndarray
    upright_body_rot_angle_rad: float
    constraint_residual_norm: float


@dataclass(frozen=True)
class CamberShimAssemblyContext:
    """
    Fixed geometry and invariants for a single shim assembly solve.

    This is the local equivalent of the main solver's pre-built residual context:
    all design-state offsets and invariant lengths are computed once, then reused
    on every residual evaluation.
    """

    shim_setup_thickness: float
    shim_face_normal_design: np.ndarray
    wishbone_axis: np.ndarray
    trackrod_inboard_position: np.ndarray
    trackrod_length: float
    lbj_position: np.ndarray
    uwb_inboard_front_position: np.ndarray
    uwb_inboard_front_to_ubj_design: np.ndarray
    ubj_to_camber_block_datum_a: np.ndarray
    ubj_to_camber_block_datum_b: np.ndarray
    lbj_to_upright_body_datum_a: np.ndarray
    lbj_to_upright_body_datum_b: np.ndarray
    lbj_to_trackrod_outboard: np.ndarray


def compute_camber_shim_assembly_residuals(
    x: np.ndarray,
    assembly_context: CamberShimAssemblyContext,
) -> np.ndarray:
    """
    Compute the residual vector for the local camber shim assembly solve.

    Variables (7):
        x[0]   - Wishbone angle: rotation of UBJ about the upper wishbone axis.
        x[1:4] - Camber block rotation vector about UBJ.
        x[4:7] - Upright body rotation vector about LBJ.

    Residuals (10):
        [0:3]  - Shim contact/datum A closure: p_lA - p_uA - t * n_u.
        [3:6]  - Shim contact/datum B closure: p_lB - p_uB - t * n_u.
        [6:9]  - Normal alignment: n_l - n_u.
        [9]    - Trackrod length: |rotated_tro - tri| - L_trackrod.
    """
    wishbone_angle = x[0]
    camber_block_rot_vec = x[1:4]
    upright_body_rot_vec = x[4:7]

    # Compute UBJ position exactly on the wishbone arc by rotating the
    # design-state offset about the wishbone axis (front-to-rear inboard line).
    wishbone_rot_vec = assembly_context.wishbone_axis * wishbone_angle
    solved_ubj_position = (
        assembly_context.uwb_inboard_front_position
        + _rotate_array_rodrigues(
            assembly_context.uwb_inboard_front_to_ubj_design,
            wishbone_rot_vec,
        )
    )

    # Rotate the design-state pivot-to-datum vectors on each rigid half of the
    # split upright.
    rotated_ubj_to_camber_block_datum_a = _rotate_array_rodrigues(
        assembly_context.ubj_to_camber_block_datum_a,
        camber_block_rot_vec,
    )
    rotated_ubj_to_camber_block_datum_b = _rotate_array_rodrigues(
        assembly_context.ubj_to_camber_block_datum_b,
        camber_block_rot_vec,
    )
    solved_camber_block_face_normal = _rotate_array_rodrigues(
        assembly_context.shim_face_normal_design,
        camber_block_rot_vec,
    )

    rotated_lbj_to_upright_body_datum_a = _rotate_array_rodrigues(
        assembly_context.lbj_to_upright_body_datum_a,
        upright_body_rot_vec,
    )
    rotated_lbj_to_upright_body_datum_b = _rotate_array_rodrigues(
        assembly_context.lbj_to_upright_body_datum_b,
        upright_body_rot_vec,
    )

    solved_upright_body_face_normal = _rotate_array_rodrigues(
        assembly_context.shim_face_normal_design,
        upright_body_rot_vec,
    )

    # Reconstruct the world positions of the A/B interface datums from the solved
    # rigid-body pose of each half.
    solved_camber_block_datum_a = (
        solved_ubj_position + rotated_ubj_to_camber_block_datum_a
    )
    solved_camber_block_datum_b = (
        solved_ubj_position + rotated_ubj_to_camber_block_datum_b
    )
    solved_upright_body_datum_a = (
        assembly_context.lbj_position + rotated_lbj_to_upright_body_datum_a
    )
    solved_upright_body_datum_b = (
        assembly_context.lbj_position + rotated_lbj_to_upright_body_datum_b
    )

    # Closure residual: the opposing datum points on each shim face must have coaxial
    # normals (parallel with the main face normal) and separated by exactly the setup
    # shim thickness. Two dowel datums (A, B) clock the interface orientation; no
    # relative rotation is allowed.
    datum_a_closure_residual = (
        solved_upright_body_datum_a
        - solved_camber_block_datum_a
        - assembly_context.shim_setup_thickness * solved_camber_block_face_normal
    )
    datum_b_closure_residual = (
        solved_upright_body_datum_b
        - solved_camber_block_datum_b
        - assembly_context.shim_setup_thickness * solved_camber_block_face_normal
    )

    # The two faces must keep the same orientation, not just be parallel up to a
    # sign flip. Using the vector difference rejects the anti-parallel branch.
    face_normal_alignment_residual = (
        solved_upright_body_face_normal - solved_camber_block_face_normal
    )

    # The trackrod remains a rigid link while its outboard pickup rotates with the
    # lower body about LBJ.
    rotated_trackrod_offset = _rotate_array_rodrigues(
        assembly_context.lbj_to_trackrod_outboard,
        upright_body_rot_vec,
    )
    solved_trackrod_outboard = assembly_context.lbj_position + rotated_trackrod_offset
    trackrod_length_residual = (
        np.linalg.norm(
            solved_trackrod_outboard - assembly_context.trackrod_inboard_position
        )
        - assembly_context.trackrod_length
    )

    return np.concatenate(
        [
            datum_a_closure_residual,
            datum_b_closure_residual,
            face_normal_alignment_residual,
            np.array([trackrod_length_residual]),
        ]
    )


# Kinematic hardpoints required to run the shim solve. Shim face geometry
# (datum points and normal) is read from shim_config, not from the positions
# dict, since the normal is a Direction3 rather than a Point3.
REQUIRED_POINT_IDS = frozenset(
    {
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_INBOARD_REAR,
        PointID.TRACKROD_OUTBOARD,
        PointID.TRACKROD_INBOARD,
    }
)


def solve_camber_shim_assembly(
    positions: dict[PointID, Point3],
    shim_config: CamberShimConfig,
    solver_config: SolverConfig = SolverConfig(),
) -> CamberShimAssemblySolution:
    """
    Solve the suspension pose for the specified split body (outboard) camber shim.

    Finds the configuration where the camber block (rotating about the UBJ) and
    upright body (rotating about the LBJ) produce parallel shim faces separated
    by the setup thickness, while the UBJ remains on the upper wishbone arc,
    with trackrod length remaining equal to design condition.

    Args:
        positions: Dict mapping PointID to Point3 positions.
        shim_config: Shim thickness configuration (design and setup thicknesses).
        solver_config: Solver configuration (tolerances, verbosity, etc.).

    Returns:
        Solved assembly state with UBJ position, rotation vectors, and convergence
        info.

    Raises:
        RuntimeError: If the solver fails to converge.
        KeyError: If a required PointID is missing from positions.
    """
    missing = REQUIRED_POINT_IDS - positions.keys()
    if missing:
        names = sorted(p.name for p in missing)
        raise KeyError(f"Missing required PointIDs: {names}")

    upper_ball_joint_design = positions[PointID.UPPER_WISHBONE_OUTBOARD].data
    lower_ball_joint = positions[PointID.LOWER_WISHBONE_OUTBOARD].data
    upper_wishbone_pickup_front = positions[PointID.UPPER_WISHBONE_INBOARD_FRONT].data
    upper_wishbone_pickup_rear = positions[PointID.UPPER_WISHBONE_INBOARD_REAR].data
    trackrod_outboard_design = positions[PointID.TRACKROD_OUTBOARD].data
    trackrod_inboard = positions[PointID.TRACKROD_INBOARD].data

    # Shim geometry: datum points are Point3, normal is a unit Direction3.
    shim_face_datum_a = shim_config.shim_face_point_a.data
    shim_face_datum_b = shim_config.shim_face_point_b.data
    design_face_normal = shim_config.shim_face_normal.data

    # Early exit when there is no shim thickness change.
    if abs(shim_config.setup_thickness - shim_config.design_thickness) < EPS_GEOMETRIC:
        return CamberShimAssemblySolution(
            ubj_position=upper_ball_joint_design.copy(),
            camber_block_rot_vec=np.zeros(3),
            upright_body_rot_vec=np.zeros(3),
            camber_block_face_normal=design_face_normal.copy(),
            upright_body_face_normal=design_face_normal.copy(),
            upright_body_rot_axis=np.array([0.0, 0.0, 1.0]),
            upright_body_rot_angle_rad=0.0,
            constraint_residual_norm=0.0,
        )

    half_design_thickness = 0.5 * shim_config.design_thickness

    # Design-state face datum positions. The camber block face is on the inboard
    # side (toward UBJ), and the upright body face is on the outboard side
    # (toward the main upright body).
    camber_block_datum_a_design = (
        shim_face_datum_a - half_design_thickness * design_face_normal
    )
    camber_block_datum_b_design = (
        shim_face_datum_b - half_design_thickness * design_face_normal
    )
    upright_body_datum_a_design = (
        shim_face_datum_a + half_design_thickness * design_face_normal
    )
    upright_body_datum_b_design = (
        shim_face_datum_b + half_design_thickness * design_face_normal
    )

    # Upper wishbone axis: the line through the two inboard pickups about which
    # the wishbone rotates. UBJ traces a circular arc about this axis.
    wishbone_axis = extract_array(
        normalize_vector(upper_wishbone_pickup_rear - upper_wishbone_pickup_front)
    )

    # Design-state offset from the front inboard pickup to UBJ. Rotating this
    # vector about the wishbone axis by the solved wishbone angle recovers the
    # solved UBJ position exactly on the arc.
    upper_wishbone_pickup_front_to_ubj_design = (
        upper_ball_joint_design - upper_wishbone_pickup_front
    )

    # Design-state trackrod length. The trackrod is a rigid link so this distance
    # must be preserved through the shim change.
    trackrod_length = float(np.linalg.norm(trackrod_outboard_design - trackrod_inboard))

    # Design-state pivot-to-datum vectors. These are the local vectors rotated by
    # the respective rotation vectors during the solve.
    ubj_to_camber_block_datum_a = camber_block_datum_a_design - upper_ball_joint_design
    ubj_to_camber_block_datum_b = camber_block_datum_b_design - upper_ball_joint_design
    lbj_to_upright_body_datum_a = upright_body_datum_a_design - lower_ball_joint
    lbj_to_upright_body_datum_b = upright_body_datum_b_design - lower_ball_joint

    # Trackrod outboard vector from LBJ (rotates with the upright body).
    lbj_to_trackrod_outboard = trackrod_outboard_design - lower_ball_joint

    assembly_context = CamberShimAssemblyContext(
        lbj_position=lower_ball_joint,
        uwb_inboard_front_position=upper_wishbone_pickup_front,
        wishbone_axis=wishbone_axis,
        uwb_inboard_front_to_ubj_design=upper_wishbone_pickup_front_to_ubj_design,
        trackrod_inboard_position=trackrod_inboard,
        shim_face_normal_design=design_face_normal,
        ubj_to_camber_block_datum_a=ubj_to_camber_block_datum_a,
        ubj_to_camber_block_datum_b=ubj_to_camber_block_datum_b,
        lbj_to_upright_body_datum_a=lbj_to_upright_body_datum_a,
        lbj_to_upright_body_datum_b=lbj_to_upright_body_datum_b,
        lbj_to_trackrod_outboard=lbj_to_trackrod_outboard,
        shim_setup_thickness=shim_config.setup_thickness,
        trackrod_length=trackrod_length,
    )

    # Seed from design condition: zero wishbone angle, zero rotations.
    x_0 = np.zeros(CAMBER_SHIM_N_VARS)

    result = solve_least_squares_problem(
        residual_function=compute_camber_shim_assembly_residuals,
        x_0=x_0,
        args=(assembly_context,),
        solver_config=solver_config,
        n_residuals=CAMBER_SHIM_N_RESIDUALS,
    )

    if not result.success:
        raise RuntimeError(
            f"Camber shim assembly solve failed to converge.\nMessage: {result.message}"
        )

    # Extract solution. Recover UBJ position from the solved wishbone angle.
    solved_wishbone_angle_rad = result.x[0]
    camber_block_rot_vec = result.x[1:4].copy()
    upright_body_rot_vec = result.x[4:7].copy()

    solved_ubj_position = upper_wishbone_pickup_front + _rotate_array_rodrigues(
        upper_wishbone_pickup_front_to_ubj_design,
        wishbone_axis * solved_wishbone_angle_rad,
    )

    # Compute solved face normals.
    solved_camber_block_face_normal = _rotate_array_rodrigues(
        design_face_normal, result.x[1:4]
    )
    solved_upright_body_face_normal = _rotate_array_rodrigues(
        design_face_normal, result.x[4:7]
    )

    # Extract upright body rotation axis and angle for suspension integration.
    upright_body_rot_angle_rad = float(np.linalg.norm(upright_body_rot_vec))
    if upright_body_rot_angle_rad > EPS_NUMERICAL:
        upright_body_rot_axis = upright_body_rot_vec / upright_body_rot_angle_rad
    else:
        upright_body_rot_axis = np.array([0.0, 0.0, 1.0])

    return CamberShimAssemblySolution(
        ubj_position=solved_ubj_position,
        camber_block_rot_vec=camber_block_rot_vec,
        upright_body_rot_vec=upright_body_rot_vec,
        camber_block_face_normal=solved_camber_block_face_normal,
        upright_body_face_normal=solved_upright_body_face_normal,
        upright_body_rot_axis=upright_body_rot_axis,
        upright_body_rot_angle_rad=upright_body_rot_angle_rad,
        constraint_residual_norm=float(np.linalg.norm(result.fun)),
    )
