"""Wheel-plane tangency to a shared axle road plane.

The road plane is extruded along chassis X, so its normal is parameterized as
``(0, sin(theta), cos(theta))``.  The scalar solve chooses the locally
continuous root nearest the independent flat-road tangent estimate.
"""

from __future__ import annotations

from math import atan2, isfinite, pi
from typing import Any, Mapping, TypeVar

import numpy as np

from kinematics.core.enums import Axis, PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.dual import DualScalar, DualVec3, cos, dot, sin
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.generic import normalize_vector

_K = TypeVar("_K", bound=PointKey)
# A real axle road plane must stay well inside the vertical-normal singularity:
# at 80 degrees the wheel-plane normal projection remains safely resolvable.
_ROAD_ANGLE_LIMIT = 80.0 * pi / 180.0
_ROAD_SOLVE_LIMIT = _ROAD_ANGLE_LIMIT - 1e-6
_ROOT_TOLERANCE_MM = 1e-9
_NEWTON_ITERATIONS = 10
_FALLBACK_BRACKET_SEGMENTS = 64


def _as_road_vector(position: Any) -> np.ndarray | DualVec3:
    """Return a raw or dual vector from a geometry position value."""
    if isinstance(position, DualVec3):
        if not np.isfinite(position.val).all() or not np.isfinite(position.deriv).all():
            raise ValueError("Wheel-plane road tangent requires finite point inputs")
        return position
    vector = position.data
    if not np.isfinite(vector).all():
        raise ValueError("Wheel-plane road tangent requires finite point inputs")
    return vector


def _primal_vector(vector: np.ndarray | DualVec3) -> np.ndarray:
    """Strip any dual derivatives for deterministic scalar root selection."""
    return vector.val if isinstance(vector, DualVec3) else vector


def _ensure_finite_road_vector(
    vector: np.ndarray | DualVec3, description: str
) -> np.ndarray | DualVec3:
    """Reject non-finite raw or dual vectors before they enter root solving."""
    finite = (
        np.isfinite(vector.val).all() and np.isfinite(vector.deriv).all()
        if isinstance(vector, DualVec3)
        else np.isfinite(vector).all()
    )
    if not finite:
        raise ValueError(f"Wheel-plane road tangent produced non-finite {description}")
    return vector


def _make_position(vector: np.ndarray | DualVec3) -> Point3 | DualVec3:
    """Wrap a calculated vector as the matching derived-point position type."""
    return vector if isinstance(vector, DualVec3) else Point3(vector)


def _road_normal(angle: float | DualScalar) -> np.ndarray | DualVec3:
    """Return the zero-grade road-plane normal ``(0, sin(angle), cos(angle))``."""
    if isinstance(angle, DualScalar):
        sine = sin(angle)
        cosine = cos(angle)
        assert isinstance(sine, DualScalar)
        assert isinstance(cosine, DualScalar)
        return DualVec3(
            np.array((0.0, sine.val, cosine.val)),
            np.array((0.0, sine.deriv, cosine.deriv)),
        )
    return np.array((0.0, np.sin(angle), np.cos(angle)))


def _wheel_spin_axis(
    positions: Mapping[_K, Any], inboard: _K, outboard: _K
) -> np.ndarray | DualVec3:
    """Return the normalized inboard-to-outboard wheel spin axis."""
    axis = _as_road_vector(positions[outboard]) - _as_road_vector(positions[inboard])
    return _ensure_finite_road_vector(
        normalize_vector(axis),  # type: ignore[arg-type]
        "spin axis",
    )


def _wheel_plane_support_point(
    center: np.ndarray | DualVec3,
    spin_axis: np.ndarray | DualVec3,
    radius: float,
    road_normal: np.ndarray | DualVec3,
) -> np.ndarray | DualVec3:
    """Return the wheel-plane point tangent to a road plane with ``road_normal``."""
    if not isfinite(radius) or radius <= EPS_GEOMETRIC:
        raise ValueError("Wheel-plane road tangent requires a finite positive radius")
    # The dot overload is dual-aware, but ty cannot correlate its scalar result
    # with the dual/raw vector union carried by this generic helper.
    projected_normal = road_normal - dot(road_normal, spin_axis) * spin_axis  # ty: ignore[unsupported-operator]
    support_direction = normalize_vector(projected_normal)  # type: ignore[arg-type]
    return _ensure_finite_road_vector(
        center - support_direction * radius,
        "support point",
    )


def _shared_plane_residual(
    left_center: np.ndarray | DualVec3,
    left_axis: np.ndarray | DualVec3,
    left_radius: float,
    right_center: np.ndarray | DualVec3,
    right_axis: np.ndarray | DualVec3,
    right_radius: float,
    angle: float | DualScalar,
) -> float | DualScalar:
    """Return ``n·P_left - n·P_right`` for the candidate road-plane angle."""
    road_normal = _road_normal(angle)
    left_support = _wheel_plane_support_point(
        left_center, left_axis, left_radius, road_normal
    )
    right_support = _wheel_plane_support_point(
        right_center, right_axis, right_radius, road_normal
    )
    left_height = dot(road_normal, left_support)
    right_height = dot(road_normal, right_support)
    if isinstance(left_height, DualScalar):
        assert isinstance(right_height, DualScalar)
        return left_height - right_height
    assert not isinstance(right_height, DualScalar)
    return float(left_height - right_height)


def _flat_road_angle_estimate(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
) -> float:
    """Estimate roll from independent standalone flat-road tangent points."""
    flat_normal = np.array((0.0, 0.0, 1.0))
    left_support = _wheel_plane_support_point(
        left_center, left_axis, left_radius, flat_normal
    )
    right_support = _wheel_plane_support_point(
        right_center, right_axis, right_radius, flat_normal
    )
    lateral_separation = left_support[Axis.Y] - right_support[Axis.Y]
    if abs(lateral_separation) <= EPS_GEOMETRIC:
        raise ValueError("Cannot determine axle road tangent with collapsed track")
    estimate = -atan2(left_support[Axis.Z] - right_support[Axis.Z], lateral_separation)
    return float(np.clip(estimate, -_ROAD_ANGLE_LIMIT, _ROAD_ANGLE_LIMIT))


def _bisect_road_root(
    residual: Any, lower: float, upper: float, lower_value: float
) -> float:
    """Return the sign-changing root bracketed by ``lower`` and ``upper``."""
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        midpoint_value = float(residual(midpoint))
        if abs(midpoint_value) <= _ROOT_TOLERANCE_MM:
            return midpoint
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            lower_value = midpoint_value
    return (lower + upper) / 2.0


def _root_derivative(residual: Any, angle: float) -> float:
    """Return a centered finite-difference derivative within the valid angle domain."""
    step = min(1e-6, (_ROAD_SOLVE_LIMIT - abs(angle)) / 2.0)
    if step <= 0.0:
        raise ValueError("Shared wheel-plane road tangent reached invalid angle limit")
    derivative = (float(residual(angle + step)) - float(residual(angle - step))) / (
        2.0 * step
    )
    if not np.isfinite(derivative):
        raise ValueError("Shared wheel-plane road tangent has non-finite angle slope")
    return derivative


def _root_derivative_threshold(
    left_center: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_radius: float,
) -> float:
    """Scale the uniqueness guard to the physical axle and tyre dimensions."""
    geometry_scale = max(
        1.0,
        left_radius,
        right_radius,
        float(np.linalg.norm(left_center - right_center)),
    )
    return EPS_GEOMETRIC * geometry_scale


def _validate_road_root(
    residual: Any, angle: float, derivative_threshold: float
) -> float:
    """Require a finite, satisfied, locally unique shared-plane root."""
    value = float(residual(angle))
    if not np.isfinite(value) or abs(value) > _ROOT_TOLERANCE_MM:
        raise ValueError("Shared wheel-plane road tangent does not satisfy its plane")
    if abs(_root_derivative(residual, angle)) <= derivative_threshold:
        raise ValueError("Shared wheel-plane road tangent is locally non-unique")
    return angle


def _solve_road_angle(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
) -> float:
    """Solve the shared-plane condition near the current flat-road estimate."""

    def residual(angle: float) -> float:
        value = float(
            _shared_plane_residual(
                left_center,
                left_axis,
                left_radius,
                right_center,
                right_axis,
                right_radius,
                angle,
            )
        )
        if not np.isfinite(value):
            raise ValueError("Shared wheel-plane road tangent has non-finite residual")
        return value

    estimate = _flat_road_angle_estimate(
        left_center,
        left_axis,
        left_radius,
        right_center,
        right_axis,
        right_radius,
    )
    derivative_threshold = _root_derivative_threshold(
        left_center, left_radius, right_center, right_radius
    )
    angle = estimate
    for _ in range(_NEWTON_ITERATIONS):
        value = residual(angle)
        if abs(value) <= _ROOT_TOLERANCE_MM:
            return _validate_road_root(residual, angle, derivative_threshold)
        derivative = _root_derivative(residual, angle)
        if abs(derivative) <= derivative_threshold:
            break
        candidate = float(
            np.clip(angle - value / derivative, -_ROAD_SOLVE_LIMIT, _ROAD_SOLVE_LIMIT)
        )
        if abs(residual(candidate)) >= abs(value):
            break
        angle = candidate

    sample_angles = np.linspace(
        -_ROAD_SOLVE_LIMIT, _ROAD_SOLVE_LIMIT, _FALLBACK_BRACKET_SEGMENTS + 1
    )
    sample_values = [residual(float(sample)) for sample in sample_angles]
    roots = [
        float(sample)
        for sample, value in zip(sample_angles, sample_values, strict=True)
        if abs(value) <= _ROOT_TOLERANCE_MM
    ]
    for lower, upper, lower_value, upper_value in zip(
        sample_angles[:-1],
        sample_angles[1:],
        sample_values[:-1],
        sample_values[1:],
        strict=True,
    ):
        if lower_value * upper_value < 0.0:
            roots.append(
                _bisect_road_root(residual, float(lower), float(upper), lower_value)
            )
    if not roots:
        raise ValueError("Unable to find a shared wheel-plane road tangent")
    return _validate_road_root(
        residual,
        min(roots, key=lambda root: abs(root - estimate)),
        derivative_threshold,
    )


def _shared_road_angle(
    left_center: np.ndarray | DualVec3,
    left_axis: np.ndarray | DualVec3,
    left_radius: float,
    right_center: np.ndarray | DualVec3,
    right_axis: np.ndarray | DualVec3,
    right_radius: float,
) -> float | DualScalar:
    """Solve the scalar road angle and propagate it by implicit differentiation."""
    primal_left_center = _primal_vector(left_center)
    primal_left_axis = _primal_vector(left_axis)
    primal_right_center = _primal_vector(right_center)
    primal_right_axis = _primal_vector(right_axis)
    angle = _solve_road_angle(
        primal_left_center,
        primal_left_axis,
        left_radius,
        primal_right_center,
        primal_right_axis,
        right_radius,
    )
    if not any(
        isinstance(value, DualVec3)
        for value in (left_center, left_axis, right_center, right_axis)
    ):
        return angle

    input_residual = _shared_plane_residual(
        left_center,
        left_axis,
        left_radius,
        right_center,
        right_axis,
        right_radius,
        DualScalar(angle),
    )
    angle_residual = _shared_plane_residual(
        primal_left_center,
        primal_left_axis,
        left_radius,
        primal_right_center,
        primal_right_axis,
        right_radius,
        DualScalar(angle, 1.0),
    )
    assert isinstance(input_residual, DualScalar)
    assert isinstance(angle_residual, DualScalar)
    derivative_threshold = _root_derivative_threshold(
        primal_left_center, left_radius, primal_right_center, right_radius
    )
    if abs(angle_residual.deriv) <= derivative_threshold:
        raise ValueError("Shared wheel-plane road tangent is locally singular")
    return DualScalar(angle, -input_residual.deriv / angle_residual.deriv)


def get_wheel_plane_road_tangent(
    positions: Mapping[PointKey, Any], tire_radius: float
) -> Point3 | DualVec3:
    """Return a standalone wheel tangent against the flat +Z road normal."""
    wheel_center = _as_road_vector(positions[PointID.WHEEL_CENTER])
    spin_axis = _wheel_spin_axis(positions, PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)
    return _make_position(
        _wheel_plane_support_point(
            wheel_center, spin_axis, tire_radius, np.array((0.0, 0.0, 1.0))
        )
    )


def get_axle_wheel_plane_road_tangent(
    positions: Mapping[_K, Any],
    *,
    left_center: _K,
    left_axis_inboard: _K,
    left_axis_outboard: _K,
    left_radius: float,
    right_center: _K,
    right_axis_inboard: _K,
    right_axis_outboard: _K,
    right_radius: float,
    side: str,
) -> Point3 | DualVec3:
    """Compute one side's tangent to the axle's common zero-grade road plane."""
    left_center_vector = _as_road_vector(positions[left_center])
    right_center_vector = _as_road_vector(positions[right_center])
    left_axis = _wheel_spin_axis(positions, left_axis_inboard, left_axis_outboard)
    right_axis = _wheel_spin_axis(positions, right_axis_inboard, right_axis_outboard)
    road_normal = _road_normal(
        _shared_road_angle(
            left_center_vector,
            left_axis,
            left_radius,
            right_center_vector,
            right_axis,
            right_radius,
        )
    )
    if side == "left":
        support = _wheel_plane_support_point(
            left_center_vector, left_axis, left_radius, road_normal
        )
    elif side == "right":
        support = _wheel_plane_support_point(
            right_center_vector, right_axis, right_radius, road_normal
        )
    else:
        raise ValueError(f"Unsupported axle tangent side: {side!r}")
    return _make_position(support)
