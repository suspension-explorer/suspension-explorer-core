"""Renderer-neutral rigid-body twist and instantaneous screw-axis fitting.

The functions here consume a single analytical tangent field.  They do not
perturb a solved configuration or infer motion from neighbouring sweep states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

import numpy as np

from kinematics.core.primitives.geometry import Point3, Vector3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.sensitivity import TangentField

# These relative tolerances deliberately scale with the supplied geometry and
# tangent magnitudes.  The absolute floors only protect the zero-scale case.
_GEOMETRY_RANK_RELATIVE_TOLERANCE = 1e-10
_ANGULAR_RATE_RELATIVE_TOLERANCE = 1e-8
_FIT_RELATIVE_TOLERANCE = 1e-5
_FIT_ABSOLUTE_TOLERANCE = 1e-10

VelocityField = TangentField | Mapping[PointKey, np.ndarray]


class ScrewAxisStatus(StrEnum):
    """Validity status for a fitted upright screw axis."""

    VALID = "valid"
    NO_STEERING_ACTUATOR = "no_steering_actuator"
    TANGENT_UNAVAILABLE = "tangent_unavailable"
    DEGENERATE_UPRIGHT = "degenerate_upright"
    RANK_DEFICIENT = "rank_deficient"
    NEAR_PURE_TRANSLATION = "near_pure_translation"
    EXCESSIVE_FIT_ERROR = "excessive_fit_error"
    NON_FINITE = "non_finite"


@dataclass(frozen=True)
class RigidBodyTwist:
    """Least-squares rigid-body twist about a selected reference point.

    ``reference_point`` is the centroid of the points used in the fit.  The
    velocity law is ``velocity = reference_velocity + angular_velocity x r``.
    Invalid fits leave the three kinematic vectors as ``None`` and carry a
    non-``VALID`` status rather than raising from a sweep.
    """

    reference_point: Point3 | None
    reference_velocity: Vector3 | None
    angular_velocity: Vector3 | None
    fit_rms: float
    fit_max: float
    fit_rank: int
    point_count: int
    velocity_scale: float
    geometry_scale: float
    status: ScrewAxisStatus = ScrewAxisStatus.VALID
    message: str | None = None

    @property
    def valid(self) -> bool:
        """Whether a full-rank, finite twist was recovered."""
        return self.status is ScrewAxisStatus.VALID


@dataclass(frozen=True)
class InstantaneousScrewAxis:
    """Finite representation of an instantaneous screw axis line."""

    point: Point3
    direction: Vector3
    pitch: float
    angular_rate: float
    fit_rms: float
    fit_max: float


@dataclass(frozen=True)
class UprightScrewAxisResult:
    """Axis result and diagnostics for one upright at one solved state."""

    upright_label: str
    point_keys: tuple[PointKey, ...]
    axis: InstantaneousScrewAxis | None
    status: ScrewAxisStatus
    twist: RigidBodyTwist | None = None
    message: str | None = None

    @property
    def point_count(self) -> int:
        """Return the number of points that participated in the fit."""
        return self.twist.point_count if self.twist is not None else 0


@dataclass(frozen=True)
class _SelectedPoints:
    """Internally collected position and tangent arrays in stable key order."""

    keys: tuple[PointKey, ...]
    positions: np.ndarray
    velocities: np.ndarray
    status: ScrewAxisStatus
    message: str | None


def fit_rigid_body_twist(
    positions: Mapping[PointKey, Point3],
    tangent: VelocityField,
    point_keys: Sequence[PointKey],
) -> RigidBodyTwist:
    """Fit a rigid-body twist to analytical velocities at selected points.

    ``tangent`` may be a :class:`TangentField` or a plain velocity mapping,
    such as the output of ``combine_tangents``. Missing position or tangent
    entries are omitted. Duplicate keys are removed while retaining their
    first occurrence. Geometrically collinear,
    non-finite, and rank-deficient inputs return diagnostic twists instead of
    throwing, allowing one bad frame to remain local to that frame.
    """
    selected = _select_points(positions, tangent, point_keys)
    if selected.status is not ScrewAxisStatus.VALID:
        return _invalid_twist(
            selected.status,
            selected.message,
            point_count=len(selected.keys),
        )

    point_positions = selected.positions
    point_velocities = selected.velocities
    reference = np.mean(point_positions, axis=0)
    offsets = point_positions - reference
    geometry_scale = float(np.sqrt(np.mean(np.sum(offsets**2, axis=1))))
    velocity_scale = float(np.sqrt(np.mean(np.sum(point_velocities**2, axis=1))))

    design = np.empty((3 * len(selected.keys), 6), dtype=np.float64)
    for index, offset in enumerate(offsets):
        rows = slice(3 * index, 3 * index + 3)
        design[rows, :3] = np.eye(3)
        design[rows, 3:] = -_skew_symmetric(offset)

    solution, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        point_velocities.reshape(-1),
        rcond=None,
    )
    fitted_velocities = (design @ solution).reshape(-1, 3)
    residual_norms = np.linalg.norm(fitted_velocities - point_velocities, axis=1)
    fit_rms = float(np.sqrt(np.mean(residual_norms**2)))
    fit_max = float(np.max(residual_norms))

    if not (
        np.isfinite(solution).all() and np.isfinite(fit_rms) and np.isfinite(fit_max)
    ):
        return _invalid_twist(
            ScrewAxisStatus.NON_FINITE,
            "Rigid-body least-squares fit produced non-finite values.",
            point_count=len(selected.keys),
            velocity_scale=velocity_scale,
            geometry_scale=geometry_scale,
            fit_rank=int(rank),
        )
    if rank < 6:
        return _invalid_twist(
            ScrewAxisStatus.RANK_DEFICIENT,
            f"Rigid-body twist matrix rank {rank} is below the required rank 6.",
            point_count=len(selected.keys),
            velocity_scale=velocity_scale,
            geometry_scale=geometry_scale,
            fit_rank=int(rank),
            fit_rms=fit_rms,
            fit_max=fit_max,
        )

    return RigidBodyTwist(
        reference_point=Point3(reference),
        reference_velocity=Vector3(solution[:3]),
        angular_velocity=Vector3(solution[3:]),
        fit_rms=fit_rms,
        fit_max=fit_max,
        fit_rank=int(rank),
        point_count=len(selected.keys),
        velocity_scale=velocity_scale,
        geometry_scale=geometry_scale,
    )


def extract_screw_axis(
    twist: RigidBodyTwist,
    *,
    angular_tolerance: float | None = None,
    fit_tolerance: float | None = None,
) -> tuple[InstantaneousScrewAxis | None, ScrewAxisStatus, str | None]:
    """Extract a screw axis from a rigid-body twist.

    ``angular_tolerance`` and ``fit_tolerance`` are optional absolute floors;
    the effective thresholds also adapt to the velocity and geometry scale of
    the fit.  This prevents a very small angular residual of a translation from
    becoming an arbitrary, distant plotted axis.
    """
    if not twist.valid:
        return None, twist.status, twist.message
    if (
        twist.reference_point is None
        or twist.reference_velocity is None
        or twist.angular_velocity is None
    ):
        return None, ScrewAxisStatus.NON_FINITE, "Twist vectors are unavailable."

    angular_velocity = twist.angular_velocity.data
    reference_velocity = twist.reference_velocity.data
    if not (
        np.isfinite(angular_velocity).all()
        and np.isfinite(reference_velocity).all()
        and np.isfinite(twist.reference_point.data).all()
    ):
        return None, ScrewAxisStatus.NON_FINITE, "Twist contains non-finite values."

    absolute_fit_tolerance = (
        _FIT_ABSOLUTE_TOLERANCE if fit_tolerance is None else fit_tolerance
    )
    if absolute_fit_tolerance < 0.0:
        raise ValueError("fit_tolerance must be non-negative")
    effective_fit_tolerance = max(
        absolute_fit_tolerance,
        _FIT_RELATIVE_TOLERANCE * max(twist.velocity_scale, 0.0),
    )
    if twist.fit_max > effective_fit_tolerance:
        return (
            None,
            ScrewAxisStatus.EXCESSIVE_FIT_ERROR,
            "Rigid-body fit maximum residual "
            f"{twist.fit_max:.6g} exceeds scale-aware tolerance "
            f"{effective_fit_tolerance:.6g}.",
        )

    angular_rate = float(np.linalg.norm(angular_velocity))
    scale_rate = twist.velocity_scale / max(twist.geometry_scale, 1e-15)
    absolute_angular_tolerance = 0.0 if angular_tolerance is None else angular_tolerance
    if absolute_angular_tolerance < 0.0:
        raise ValueError("angular_tolerance must be non-negative")
    effective_angular_tolerance = max(
        absolute_angular_tolerance,
        _ANGULAR_RATE_RELATIVE_TOLERANCE * scale_rate,
    )
    if angular_rate <= effective_angular_tolerance:
        return (
            None,
            ScrewAxisStatus.NEAR_PURE_TRANSLATION,
            "Angular rate "
            f"{angular_rate:.6g} is below scale-aware translation threshold "
            f"{effective_angular_tolerance:.6g}.",
        )

    angular_rate_squared = angular_rate**2
    direction = angular_velocity / angular_rate
    closest_point = twist.reference_point.data + (
        np.cross(angular_velocity, reference_velocity) / angular_rate_squared
    )
    pitch = float(np.dot(angular_velocity, reference_velocity) / angular_rate_squared)
    if not (np.isfinite(closest_point).all() and np.isfinite(pitch)):
        return None, ScrewAxisStatus.NON_FINITE, "Screw-axis extraction was non-finite."

    return (
        InstantaneousScrewAxis(
            point=Point3(closest_point),
            direction=Vector3(direction),
            pitch=pitch,
            angular_rate=angular_rate,
            fit_rms=twist.fit_rms,
            fit_max=twist.fit_max,
        ),
        ScrewAxisStatus.VALID,
        None,
    )


def compute_upright_screw_axis(
    upright_label: str,
    positions: Mapping[PointKey, Point3],
    tangent: VelocityField | None,
    point_keys: Sequence[PointKey],
    *,
    tangent_rank_deficient: bool = False,
    angular_tolerance: float | None = None,
    fit_tolerance: float | None = None,
) -> UprightScrewAxisResult:
    """Return a diagnostic screw-axis result for one upright.

    This small adapter keeps generic fitting independent of suspension and
    rendering code while giving callers one result shape for every outcome.
    """
    stable_keys = tuple(dict.fromkeys(point_keys))
    if tangent is None:
        return UprightScrewAxisResult(
            upright_label=upright_label,
            point_keys=stable_keys,
            axis=None,
            status=ScrewAxisStatus.TANGENT_UNAVAILABLE,
            message="No steering tangent field is available for this state.",
        )
    if tangent_rank_deficient:
        return UprightScrewAxisResult(
            upright_label=upright_label,
            point_keys=stable_keys,
            axis=None,
            status=ScrewAxisStatus.RANK_DEFICIENT,
            message="The analytical tangent solve is rank-deficient.",
        )

    twist = fit_rigid_body_twist(positions, tangent, stable_keys)
    axis, status, message = extract_screw_axis(
        twist,
        angular_tolerance=angular_tolerance,
        fit_tolerance=fit_tolerance,
    )
    return UprightScrewAxisResult(
        upright_label=upright_label,
        point_keys=stable_keys,
        axis=axis,
        status=status,
        twist=twist,
        message=message,
    )


def _select_points(
    positions: Mapping[PointKey, Point3],
    tangent: VelocityField,
    point_keys: Sequence[PointKey],
) -> _SelectedPoints:
    """Collect finite position/velocity pairs and validate their geometry."""
    stable_keys = tuple(dict.fromkeys(point_keys))
    usable_keys: list[PointKey] = []
    point_positions: list[np.ndarray] = []
    point_velocities: list[np.ndarray] = []
    missing_tangent: list[PointKey] = []
    velocity_mapping = _velocity_mapping(tangent)

    for key in stable_keys:
        position = positions.get(key)
        if position is None:
            continue
        velocity = velocity_mapping.get(key)
        if velocity is None:
            missing_tangent.append(key)
            continue
        position_array = np.asarray(position.data, dtype=np.float64)
        velocity_array = np.asarray(velocity, dtype=np.float64)
        if position_array.shape != (3,) or velocity_array.shape != (3,):
            return _SelectedPoints(
                keys=tuple(usable_keys),
                positions=np.empty((0, 3)),
                velocities=np.empty((0, 3)),
                status=ScrewAxisStatus.NON_FINITE,
                message=f"Point '{key}' has a position or velocity outside shape (3,).",
            )
        if not (
            np.isfinite(position_array).all() and np.isfinite(velocity_array).all()
        ):
            return _SelectedPoints(
                keys=tuple(usable_keys),
                positions=np.empty((0, 3)),
                velocities=np.empty((0, 3)),
                status=ScrewAxisStatus.NON_FINITE,
                message=f"Point '{key}' has non-finite position or tangent velocity.",
            )
        usable_keys.append(key)
        point_positions.append(position_array)
        point_velocities.append(velocity_array)

    if len(usable_keys) < 3:
        status = (
            ScrewAxisStatus.TANGENT_UNAVAILABLE
            if missing_tangent
            else ScrewAxisStatus.DEGENERATE_UPRIGHT
        )
        return _SelectedPoints(
            keys=tuple(usable_keys),
            positions=np.empty((0, 3)),
            velocities=np.empty((0, 3)),
            status=status,
            message=(
                f"Only {len(usable_keys)} usable rigid points were available; "
                "at least three non-collinear points are required."
            ),
        )

    selected_positions = np.vstack(point_positions)
    selected_velocities = np.vstack(point_velocities)
    centered = selected_positions - np.mean(selected_positions, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest_singular_value = float(singular_values[0])
    second_singular_value = float(singular_values[1])
    if (
        not np.isfinite(singular_values).all()
        or largest_singular_value == 0.0
        or second_singular_value
        <= _GEOMETRY_RANK_RELATIVE_TOLERANCE * largest_singular_value
    ):
        return _SelectedPoints(
            keys=tuple(usable_keys),
            positions=selected_positions,
            velocities=selected_velocities,
            status=ScrewAxisStatus.DEGENERATE_UPRIGHT,
            message="Upright points are coincident or collinear.",
        )

    return _SelectedPoints(
        keys=tuple(usable_keys),
        positions=selected_positions,
        velocities=selected_velocities,
        status=ScrewAxisStatus.VALID,
        message=None,
    )


def _invalid_twist(
    status: ScrewAxisStatus,
    message: str | None,
    *,
    point_count: int,
    velocity_scale: float = 0.0,
    geometry_scale: float = 0.0,
    fit_rank: int = 0,
    fit_rms: float = float("nan"),
    fit_max: float = float("nan"),
) -> RigidBodyTwist:
    """Build a consistent diagnostic twist without invented kinematics."""
    return RigidBodyTwist(
        reference_point=None,
        reference_velocity=None,
        angular_velocity=None,
        fit_rms=fit_rms,
        fit_max=fit_max,
        fit_rank=fit_rank,
        point_count=point_count,
        velocity_scale=velocity_scale,
        geometry_scale=geometry_scale,
        status=status,
        message=message,
    )


def _velocity_mapping(tangent: VelocityField) -> Mapping[PointKey, np.ndarray]:
    """Return a raw point-velocity mapping from either supported field shape."""
    return tangent.velocities if isinstance(tangent, TangentField) else tangent


def _skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return ``[vector]x``, satisfying ``[vector]x @ other = vector x other``."""
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
