"""Fit parameterised rigid-body twists and extract instantaneous screw axes.

This module is deliberately ignorant of suspension and sweep semantics.  Its
input is a point-rate field for a collection of points on one rigid body; the
caller decides whether that field represents isolated steering, combined
bump-and-steer, or motion along an authored sweep.  The mathematics cannot make
that distinction after the rates have been supplied.

For positions ``p_i(lambda)``, point rates ``r_i = dp_i/dlambda`` and a
reference point ``p_ref``, the least-squares fit recovers the parameterised
rigid-body twist satisfying

``r_i = r_ref + rho x (p_i - p_ref)``,

where ``lambda`` is the caller's driving parameter and
``rho = dtheta/dlambda``.  For virtual steering, ``lambda`` is rack
displacement, so point rates are measured per unit rack displacement rather
than per unit time.

Classical screw theory commonly calls these quantities a spatial velocity or
velocity twist because time is its usual parameter.  This module borrows that
twist and screw-axis algebra but deliberately uses *rate* nomenclature:
``lambda`` need not be time, and in the current steering analysis it is rack
displacement.

When ``rho`` is nonzero, :func:`extract_screw_axis` converts the twist to the
line about which the body instantaneously screws.  The result stores a closest
point on that line, its unoriented direction, pitch (translation along the axis
per radian), and angular rate with respect to ``lambda``.  Changing the scale
of the supplied rate field changes angular rate but not the line or pitch.

The implementation performs no geometry perturbation and never differentiates
neighbouring solved states or fitted axis values.  Invalid point geometry,
rank-deficient fits, non-finite values, near-pure translation and excessive fit
residuals are returned as explicit statuses.  Geometry, angular-rate and fit
tolerances scale with the supplied body and rate magnitudes so a tiny
rotational residual of an almost pure translation does not create an arbitrary
distant axis.

Two extracted axes are compared as infinite, unoriented lines: direction
agreement uses the absolute dot product, while position agreement uses the
shortest line-to-line distance or the distance from known pivot points to the
fitted line.  The signs of their direction vectors are not geometrically
meaningful.

An instantaneous screw axis becomes a *steering* axis only when its input
rate field was established by a steering-specific boundary condition.  In
particular, this module must not infer or correct suspension-travel motion; that
responsibility belongs to the steering-response orchestration layer.
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
_RIGID_BODY_TWIST_DOF = 6

RateField = TangentField | Mapping[PointKey, np.ndarray]


class ScrewAxisStatus(StrEnum):
    """Validity status for a fitted upright screw axis."""

    VALID = "valid"
    NO_STEERING_ACTUATOR = "no_steering_actuator"
    NO_STEERING_RESPONSE_DEFINITION = "no_steering_response_definition"
    TANGENT_UNAVAILABLE = "tangent_unavailable"
    INCONSISTENT_TANGENT = "inconsistent_tangent"
    DEGENERATE_UPRIGHT = "degenerate_upright"
    RANK_DEFICIENT = "rank_deficient"
    NEAR_PURE_TRANSLATION = "near_pure_translation"
    EXCESSIVE_FIT_ERROR = "excessive_fit_error"
    NON_FINITE = "non_finite"


@dataclass(frozen=True)
class RigidBodyTwist:
    """Least-squares rigid-body twist about a selected reference point.

    ``reference_point`` is the centroid of the points used in the fit.  The
    rate law is
    ``point_rate = reference_point_rate + rotation_rate_vector x offset``.
    Invalid fits leave the three kinematic vectors as ``None`` and carry a
    non-``VALID`` status rather than raising from a sweep.
    """

    reference_point: Point3 | None
    reference_point_rate: Vector3 | None
    rotation_rate_vector: Vector3 | None
    fit_rms: float
    fit_max: float
    fit_rank: int
    point_count: int
    rate_scale: float
    geometry_scale: float
    status: ScrewAxisStatus = ScrewAxisStatus.VALID
    message: str | None = None

    @property
    def valid(self) -> bool:
        """Whether a full-rank, finite twist was recovered."""
        return self.status is ScrewAxisStatus.VALID


@dataclass(frozen=True)
class InstantaneousScrewAxis:
    """Finite representation of an instantaneous screw axis line.

    ``angular_rate`` is per unit of the caller's driving parameter.  It is a
    time rate only when that parameter is time; for steering response it is
    rotation per unit rack displacement.
    """

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
    rates: np.ndarray
    status: ScrewAxisStatus
    message: str | None


def fit_rigid_body_twist(
    positions: Mapping[PointKey, Point3],
    tangent: RateField,
    point_keys: Sequence[PointKey],
) -> RigidBodyTwist:
    """Fit a rigid-body twist to analytical rates at selected points.

    ``tangent`` may be a :class:`TangentField` or a plain point-rate mapping,
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
    point_rates = selected.rates
    reference = np.mean(point_positions, axis=0)
    offsets = point_positions - reference
    geometry_scale = float(np.sqrt(np.mean(np.sum(offsets**2, axis=1))))
    rate_scale = float(np.sqrt(np.mean(np.sum(point_rates**2, axis=1))))

    design = np.empty(
        (3 * len(selected.keys), _RIGID_BODY_TWIST_DOF),
        dtype=np.float64,
    )
    for index, offset in enumerate(offsets):
        rows = slice(3 * index, 3 * index + 3)
        design[rows, :3] = np.eye(3)
        design[rows, 3:] = -_skew_symmetric(offset)

    solution, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        point_rates.reshape(-1),
        rcond=None,
    )
    fitted_rates = (design @ solution).reshape(-1, 3)
    residual_norms = np.linalg.norm(fitted_rates - point_rates, axis=1)
    fit_rms = float(np.sqrt(np.mean(residual_norms**2)))
    fit_max = float(np.max(residual_norms))

    if not (
        np.isfinite(solution).all() and np.isfinite(fit_rms) and np.isfinite(fit_max)
    ):
        return _invalid_twist(
            ScrewAxisStatus.NON_FINITE,
            "Rigid-body least-squares fit produced non-finite values.",
            point_count=len(selected.keys),
            rate_scale=rate_scale,
            geometry_scale=geometry_scale,
            fit_rank=int(rank),
        )
    if rank < _RIGID_BODY_TWIST_DOF:
        return _invalid_twist(
            ScrewAxisStatus.RANK_DEFICIENT,
            f"Rigid-body twist matrix rank {rank} is below the required rank "
            f"{_RIGID_BODY_TWIST_DOF}.",
            point_count=len(selected.keys),
            rate_scale=rate_scale,
            geometry_scale=geometry_scale,
            fit_rank=int(rank),
            fit_rms=fit_rms,
            fit_max=fit_max,
        )

    return RigidBodyTwist(
        reference_point=Point3(reference),
        reference_point_rate=Vector3(solution[:3]),
        rotation_rate_vector=Vector3(solution[3:]),
        fit_rms=fit_rms,
        fit_max=fit_max,
        fit_rank=int(rank),
        point_count=len(selected.keys),
        rate_scale=rate_scale,
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
    the effective thresholds also adapt to the rate and geometry scale of
    the fit.  This prevents a very small angular residual of a translation from
    becoming an arbitrary, distant plotted axis.
    """
    if not twist.valid:
        return None, twist.status, twist.message
    if (
        twist.reference_point is None
        or twist.reference_point_rate is None
        or twist.rotation_rate_vector is None
    ):
        return None, ScrewAxisStatus.NON_FINITE, "Twist vectors are unavailable."

    rotation_rate_vector = twist.rotation_rate_vector.data
    reference_point_rate = twist.reference_point_rate.data
    if not (
        np.isfinite(rotation_rate_vector).all()
        and np.isfinite(reference_point_rate).all()
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
        _FIT_RELATIVE_TOLERANCE * max(twist.rate_scale, 0.0),
    )
    if twist.fit_max > effective_fit_tolerance:
        return (
            None,
            ScrewAxisStatus.EXCESSIVE_FIT_ERROR,
            "Rigid-body fit maximum residual "
            f"{twist.fit_max:.6g} exceeds scale-aware tolerance "
            f"{effective_fit_tolerance:.6g}.",
        )

    angular_rate = float(np.linalg.norm(rotation_rate_vector))
    scale_rate = twist.rate_scale / max(twist.geometry_scale, 1e-15)
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
    direction = rotation_rate_vector / angular_rate
    closest_point = twist.reference_point.data + (
        np.cross(rotation_rate_vector, reference_point_rate) / angular_rate_squared
    )
    pitch = float(
        np.dot(rotation_rate_vector, reference_point_rate) / angular_rate_squared
    )
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
    tangent: RateField | None,
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
        return unavailable_upright_screw_axis(
            upright_label,
            stable_keys,
            ScrewAxisStatus.TANGENT_UNAVAILABLE,
            "No steering tangent field is available for this state.",
        )
    if tangent_rank_deficient:
        return unavailable_upright_screw_axis(
            upright_label,
            stable_keys,
            ScrewAxisStatus.RANK_DEFICIENT,
            "The analytical tangent solve is rank-deficient.",
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


def unavailable_upright_screw_axis(
    upright_label: str,
    point_keys: Sequence[PointKey],
    status: ScrewAxisStatus,
    message: str,
) -> UprightScrewAxisResult:
    """Return one explicit unavailable result without inventing a twist.

    Callers use this when the rate field itself cannot be established, for
    example because its boundary conditions are incomplete or inconsistent.
    Rigid-body fitting failures continue to carry their diagnostic
    :class:`RigidBodyTwist` through :func:`compute_upright_screw_axis`.
    """
    if status is ScrewAxisStatus.VALID:
        raise ValueError("An unavailable screw-axis result cannot have valid status")
    return UprightScrewAxisResult(
        upright_label=upright_label,
        point_keys=tuple(dict.fromkeys(point_keys)),
        axis=None,
        status=status,
        message=message,
    )


def _select_points(
    positions: Mapping[PointKey, Point3],
    tangent: RateField,
    point_keys: Sequence[PointKey],
) -> _SelectedPoints:
    """Collect finite position/rate pairs and validate their geometry."""
    stable_keys = tuple(dict.fromkeys(point_keys))
    usable_keys: list[PointKey] = []
    point_positions: list[np.ndarray] = []
    point_rates: list[np.ndarray] = []
    missing_tangent: list[PointKey] = []
    rate_mapping = _rate_mapping(tangent)

    for key in stable_keys:
        position = positions.get(key)
        if position is None:
            continue
        rate = rate_mapping.get(key)
        if rate is None:
            missing_tangent.append(key)
            continue
        position_array = np.asarray(position.data, dtype=np.float64)
        rate_array = np.asarray(rate, dtype=np.float64)
        if position_array.shape != (3,) or rate_array.shape != (3,):
            return _SelectedPoints(
                keys=tuple(usable_keys),
                positions=np.empty((0, 3)),
                rates=np.empty((0, 3)),
                status=ScrewAxisStatus.NON_FINITE,
                message=f"Point '{key}' has a position or rate outside shape (3,).",
            )
        if not (np.isfinite(position_array).all() and np.isfinite(rate_array).all()):
            return _SelectedPoints(
                keys=tuple(usable_keys),
                positions=np.empty((0, 3)),
                rates=np.empty((0, 3)),
                status=ScrewAxisStatus.NON_FINITE,
                message=f"Point '{key}' has non-finite position or tangent rate.",
            )
        usable_keys.append(key)
        point_positions.append(position_array)
        point_rates.append(rate_array)

    if len(usable_keys) < 3:
        status = (
            ScrewAxisStatus.TANGENT_UNAVAILABLE
            if missing_tangent
            else ScrewAxisStatus.DEGENERATE_UPRIGHT
        )
        return _SelectedPoints(
            keys=tuple(usable_keys),
            positions=np.empty((0, 3)),
            rates=np.empty((0, 3)),
            status=status,
            message=(
                f"Only {len(usable_keys)} usable rigid points were available; "
                "at least three non-collinear points are required."
            ),
        )

    selected_positions = np.vstack(point_positions)
    selected_rates = np.vstack(point_rates)
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
            rates=selected_rates,
            status=ScrewAxisStatus.DEGENERATE_UPRIGHT,
            message="Upright points are coincident or collinear.",
        )

    return _SelectedPoints(
        keys=tuple(usable_keys),
        positions=selected_positions,
        rates=selected_rates,
        status=ScrewAxisStatus.VALID,
        message=None,
    )


def _invalid_twist(
    status: ScrewAxisStatus,
    message: str | None,
    *,
    point_count: int,
    rate_scale: float = 0.0,
    geometry_scale: float = 0.0,
    fit_rank: int = 0,
    fit_rms: float = float("nan"),
    fit_max: float = float("nan"),
) -> RigidBodyTwist:
    """Build a consistent diagnostic twist without invented kinematics."""
    return RigidBodyTwist(
        reference_point=None,
        reference_point_rate=None,
        rotation_rate_vector=None,
        fit_rms=fit_rms,
        fit_max=fit_max,
        fit_rank=fit_rank,
        point_count=point_count,
        rate_scale=rate_scale,
        geometry_scale=geometry_scale,
        status=status,
        message=message,
    )


def _rate_mapping(tangent: RateField) -> Mapping[PointKey, np.ndarray]:
    """Return a raw point-rate mapping from either supported field shape."""
    return tangent.rates if isinstance(tangent, TangentField) else tangent


def _skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return ``[vector]x``, satisfying ``[vector]x @ other = vector x other``."""
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
