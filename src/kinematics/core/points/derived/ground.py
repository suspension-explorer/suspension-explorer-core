"""Wheel-plane tangency to a shared axle ground plane.

The ground plane is extruded along chassis X, so its normal is parameterised as
``(0, sin(theta), cos(theta))``. The first scalar solve starts from the
independent flat-ground tangent estimate; subsequent accepted states continue
from the preceding root so a multi-root geometry stays on one local branch.

Sign convention.  The scalar solved here is the *ground-normal* angle: it rotates
the plane's normal.  The public axle datum
:class:`kinematics.core.metrics.ground.GroundDatum` instead reports
``angle_deg = atan2(tangent_z, tangent_y)``, which measures the ground *line*.
Rotating a normal by ``+theta`` tilts the line it belongs to by ``-theta``, so
the two quantities are exact negatives of one another::

    GroundDatum.angle_deg == -degrees(ground_normal_angle)

Every internal name here says ``normal_angle`` for that reason.  Nothing in
this module is the public roll angle, and the two must not be conflated.
"""

from __future__ import annotations

from math import atan2, isfinite, pi
from typing import Any, Callable, Literal, Mapping, TypeVar

import numpy as np

from kinematics.core.enums import Axis, PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.dual import DualScalar, DualVec3, cos, dot, sin
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.generic import normalize_vector

_K = TypeVar("_K", bound=PointKey)
# Product validity policy: reject near-vertical ground planes before wheel support
# becomes numerically fragile. Eighty degrees is a supported-domain limit, not
# an intrinsic singularity of the shared-plane equation.
_GROUND_NORMAL_ANGLE_LIMIT = 80.0 * pi / 180.0
# Every angle the solver produces -- seed, Newton candidate and bracket
# endpoint alike -- is clamped here, strictly inside the limit above, so a
# saturated value can never sit on the singular boundary itself.
_GROUND_SOLVE_LIMIT = _GROUND_NORMAL_ANGLE_LIMIT - 1e-6
# The residual is a height difference in millimetres, so its convergence
# tolerance is taken relative to the axle's own dimensions rather than as an
# absolute picometre floor that float64 cannot reach at automotive scale.
_ROOT_RELATIVE_TOLERANCE = 1e-12
_NEWTON_ITERATIONS = 10
_FALLBACK_BRACKET_SEGMENTS = 64
_BISECTION_ITERATIONS = 60

_GroundAngleKey = tuple[bytes, bytes, float, bytes, bytes, float]

_ResidualFn = Callable[[float], float]
_ResidualSlopeFn = Callable[[float], tuple[float, float]]


def _as_ground_vector(position: Any) -> np.ndarray | DualVec3:
    """Return a raw or dual vector from a geometry position value."""
    if isinstance(position, DualVec3):
        if not np.isfinite(position.val).all() or not np.isfinite(position.deriv).all():
            raise ValueError("Wheel-ground tangent requires finite point inputs")
        return position
    vector = position.data
    if not np.isfinite(vector).all():
        raise ValueError("Wheel-ground tangent requires finite point inputs")
    return vector


def _primal_vector(vector: np.ndarray | DualVec3) -> np.ndarray:
    """Strip any dual derivatives for deterministic scalar root selection."""
    return vector.val if isinstance(vector, DualVec3) else vector


def _ensure_finite_ground_vector(
    vector: np.ndarray | DualVec3, description: str
) -> np.ndarray | DualVec3:
    """Reject non-finite raw or dual vectors before they enter root solving."""
    finite = (
        np.isfinite(vector.val).all() and np.isfinite(vector.deriv).all()
        if isinstance(vector, DualVec3)
        else np.isfinite(vector).all()
    )
    if not finite:
        raise ValueError(
            f"Wheel-ground tangent produced non-finite {description}"
        )
    return vector


def _make_position(vector: np.ndarray | DualVec3) -> Point3 | DualVec3:
    """Wrap a calculated vector as the matching derived-point position type."""
    return vector if isinstance(vector, DualVec3) else Point3(vector)


def _clamp_ground_normal_angle(normal_angle: float) -> float:
    """Clamp a ground-normal angle strictly inside the resolvable domain."""
    return float(np.clip(normal_angle, -_GROUND_SOLVE_LIMIT, _GROUND_SOLVE_LIMIT))


def _ground_normal(normal_angle: float | DualScalar) -> np.ndarray | DualVec3:
    """Return the zero-grade ground-plane normal ``(0, sin(a), cos(a))``."""
    if isinstance(normal_angle, DualScalar):
        sine = sin(normal_angle)
        cosine = cos(normal_angle)
        assert isinstance(sine, DualScalar)
        assert isinstance(cosine, DualScalar)
        return DualVec3(
            np.array((0.0, sine.val, cosine.val)),
            np.array((0.0, sine.deriv, cosine.deriv)),
        )
    return np.array((0.0, np.sin(normal_angle), np.cos(normal_angle)))


def _wheel_spin_axis(
    positions: Mapping[_K, Any], inboard: _K, outboard: _K
) -> np.ndarray | DualVec3:
    """Return the normalized inboard-to-outboard wheel spin axis."""
    axis = _as_ground_vector(positions[outboard]) - _as_ground_vector(
        positions[inboard]
    )
    return _ensure_finite_ground_vector(
        normalize_vector(axis),  # type: ignore[arg-type]
        "spin axis",
    )


def _wheel_plane_support_point(
    center: np.ndarray | DualVec3,
    spin_axis: np.ndarray | DualVec3,
    radius: float,
    ground_normal: np.ndarray | DualVec3,
) -> np.ndarray | DualVec3:
    """Return the wheel-plane point tangent to a ground plane with ``ground_normal``."""
    if not isfinite(radius) or radius <= EPS_GEOMETRIC:
        raise ValueError("Wheel-ground tangent requires a finite positive radius")
    # The dot overload is dual-aware, but ty cannot correlate its scalar result
    # with the dual/raw vector union carried by this generic helper.
    projected_normal = ground_normal - dot(ground_normal, spin_axis) * spin_axis  # ty: ignore[unsupported-operator]
    support_direction = normalize_vector(projected_normal)  # type: ignore[arg-type]
    return _ensure_finite_ground_vector(
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
    normal_angle: float | DualScalar,
) -> float | DualScalar:
    """Return ``n·P_left - n·P_right`` for the candidate ground-normal angle."""
    ground_normal = _ground_normal(normal_angle)
    left_support = _wheel_plane_support_point(
        left_center, left_axis, left_radius, ground_normal
    )
    right_support = _wheel_plane_support_point(
        right_center, right_axis, right_radius, ground_normal
    )
    left_height = dot(ground_normal, left_support)
    right_height = dot(ground_normal, right_support)
    if isinstance(left_height, DualScalar):
        assert isinstance(right_height, DualScalar)
        return left_height - right_height
    assert not isinstance(right_height, DualScalar)
    return float(left_height - right_height)


def _flat_ground_normal_angle_estimate(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
) -> float:
    """Estimate the ground-normal angle from standalone flat-ground tangent points."""
    flat_normal = np.array((0.0, 0.0, 1.0))
    left_support = _wheel_plane_support_point(
        left_center, left_axis, left_radius, flat_normal
    )
    right_support = _wheel_plane_support_point(
        right_center, right_axis, right_radius, flat_normal
    )
    lateral_separation = float(left_support[Axis.Y] - right_support[Axis.Y])
    vertical_separation = float(left_support[Axis.Z] - right_support[Axis.Z])
    if abs(lateral_separation) <= EPS_GEOMETRIC:
        raise ValueError("Cannot determine axle ground tangent with collapsed track")
    if lateral_separation < 0.0:
        # Mirror GroundDatum.from_wheel_ground_tangents and orient the
        # separation from vehicle right to left.  Without this, laterally
        # crossed supports give an estimate near +/-pi, which the clamp then
        # turns into a seed unrelated to the geometry.
        lateral_separation = -lateral_separation
        vertical_separation = -vertical_separation
    return _clamp_ground_normal_angle(-atan2(vertical_separation, lateral_separation))


def _geometry_scale(
    left_center: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_radius: float,
) -> float:
    """Return the characteristic length of the axle and its tyres, in millimetres."""
    return max(
        1.0,
        left_radius,
        right_radius,
        float(np.linalg.norm(left_center - right_center)),
    )


def _root_derivative_threshold(
    left_center: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_radius: float,
) -> float:
    """Scale the uniqueness guard to the physical axle and tyre dimensions."""
    return EPS_GEOMETRIC * _geometry_scale(
        left_center, left_radius, right_center, right_radius
    )


def _root_tolerance_mm(
    left_center: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_radius: float,
) -> float:
    """Scale the residual tolerance to the physical axle and tyre dimensions."""
    return _ROOT_RELATIVE_TOLERANCE * _geometry_scale(
        left_center, left_radius, right_center, right_radius
    )


def _bisect_ground_root(
    residual: _ResidualFn,
    lower: float,
    upper: float,
    lower_value: float,
    tolerance: float,
) -> float:
    """Return the sign-changing root bracketed by ``lower`` and ``upper``."""
    for _ in range(_BISECTION_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        if midpoint <= lower or midpoint >= upper:
            # The bracket has collapsed onto adjacent floats, so further
            # halving cannot refine it.
            break
        midpoint_value = residual(midpoint)
        if abs(midpoint_value) <= tolerance:
            return midpoint
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            lower_value = midpoint_value
    return (lower + upper) / 2.0


def _ground_residual_and_slope(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
    normal_angle: float,
) -> tuple[float, float]:
    """Return the residual and its exact angle derivative at ``normal_angle``.

    Seeding the angle with a unit dual part yields ``d(residual)/d(angle)``
    analytically from a single evaluation.  Because no finite-difference step is
    taken, the slope is defined wherever the residual is -- including on the
    clamped domain boundary, which a centred difference could not straddle.
    """
    dual = _shared_plane_residual(
        left_center,
        left_axis,
        left_radius,
        right_center,
        right_axis,
        right_radius,
        DualScalar(normal_angle, 1.0),
    )
    assert isinstance(dual, DualScalar)
    if not isfinite(dual.val):
        raise ValueError("Shared wheel-ground tangent has non-finite residual")
    if not isfinite(dual.deriv):
        raise ValueError("Shared wheel-ground tangent has non-finite angle slope")
    return dual.val, dual.deriv


def _validate_ground_root(
    residual_and_slope: _ResidualSlopeFn,
    normal_angle: float,
    derivative_threshold: float,
    tolerance: float,
) -> float:
    """Require a finite, satisfied, locally unique shared-plane root."""
    value, slope = residual_and_slope(normal_angle)
    if abs(value) > tolerance:
        raise ValueError("Shared wheel-ground tangent does not satisfy its plane")
    if abs(slope) <= derivative_threshold:
        raise ValueError("Shared wheel-ground tangent is locally non-unique")
    return normal_angle


def _search_ground_normal_angle(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
    *,
    seed: float | None = None,
) -> float:
    """Solve the shared-plane condition on the branch nearest ``seed``.

    The flat-ground construction provides the first seed. Subsequent axle states
    pass the previously selected root so a multi-root geometry follows one
    continuous branch instead of independently choosing a root at every state.
    """

    def residual(normal_angle: float) -> float:
        value = _shared_plane_residual(
            left_center,
            left_axis,
            left_radius,
            right_center,
            right_axis,
            right_radius,
            normal_angle,
        )
        assert not isinstance(value, DualScalar)
        if not isfinite(value):
            raise ValueError(
                "Shared wheel-ground tangent has non-finite residual"
            )
        return value

    def residual_and_slope(normal_angle: float) -> tuple[float, float]:
        return _ground_residual_and_slope(
            left_center,
            left_axis,
            left_radius,
            right_center,
            right_axis,
            right_radius,
            normal_angle,
        )

    flat_estimate = _flat_ground_normal_angle_estimate(
        left_center,
        left_axis,
        left_radius,
        right_center,
        right_axis,
        right_radius,
    )
    selection_seed = (
        flat_estimate if seed is None else _clamp_ground_normal_angle(seed)
    )
    derivative_threshold = _root_derivative_threshold(
        left_center, left_radius, right_center, right_radius
    )
    tolerance = _root_tolerance_mm(left_center, left_radius, right_center, right_radius)
    normal_angle = selection_seed
    for _ in range(_NEWTON_ITERATIONS):
        value, slope = residual_and_slope(normal_angle)
        if abs(value) <= tolerance:
            return _validate_ground_root(
                residual_and_slope, normal_angle, derivative_threshold, tolerance
            )
        if abs(slope) <= derivative_threshold:
            break
        candidate = _clamp_ground_normal_angle(normal_angle - value / slope)
        if abs(residual(candidate)) >= abs(value):
            break
        normal_angle = candidate

    sample_angles = np.linspace(
        -_GROUND_SOLVE_LIMIT, _GROUND_SOLVE_LIMIT, _FALLBACK_BRACKET_SEGMENTS + 1
    )
    sample_values = [residual(float(sample)) for sample in sample_angles]
    roots = [
        float(sample)
        for sample, value in zip(sample_angles, sample_values, strict=True)
        if abs(value) <= tolerance
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
                _bisect_ground_root(
                    residual, float(lower), float(upper), lower_value, tolerance
                )
            )
    if not roots:
        raise ValueError("Unable to find a shared wheel-ground tangent")
    return _validate_ground_root(
        residual_and_slope,
        min(roots, key=lambda root: abs(root - selection_seed)),
        derivative_threshold,
        tolerance,
    )


def _ground_angle_cache_key(
    left_center: np.ndarray,
    left_axis: np.ndarray,
    left_radius: float,
    right_center: np.ndarray,
    right_axis: np.ndarray,
    right_radius: float,
) -> _GroundAngleKey:
    """Key the primal solve on the exact bit pattern of its scalar inputs."""
    return (
        np.ascontiguousarray(left_center, dtype=np.float64).tobytes(),
        np.ascontiguousarray(left_axis, dtype=np.float64).tobytes(),
        float(left_radius),
        np.ascontiguousarray(right_center, dtype=np.float64).tobytes(),
        np.ascontiguousarray(right_axis, dtype=np.float64).tobytes(),
        float(right_radius),
    )


class _GroundNormalContinuation:
    """Sweep-local root history shared by one axle's tangent-point functions.

    Exact accepted geometries retain their selected root so later tangent
    propagation cannot choose another branch. For a new geometry, the most
    recently visited root remains the branch-selection seed.
    """

    def __init__(self) -> None:
        self._last_angle: float | None = None
        self._angles_by_key: dict[_GroundAngleKey, float] = {}

    def reset(self) -> None:
        """Start an independent root history for a new sweep."""
        self._last_angle = None
        self._angles_by_key.clear()

    def solve(
        self,
        left_center: np.ndarray,
        left_axis: np.ndarray,
        left_radius: float,
        right_center: np.ndarray,
        right_axis: np.ndarray,
        right_radius: float,
    ) -> float:
        """Return the continuous root for one axle geometry state."""
        key = _ground_angle_cache_key(
            left_center, left_axis, left_radius, right_center, right_axis, right_radius
        )
        cached_angle = self._angles_by_key.get(key)
        if cached_angle is not None:
            self._last_angle = cached_angle
            return cached_angle

        normal_angle = _search_ground_normal_angle(
            left_center,
            left_axis,
            left_radius,
            right_center,
            right_axis,
            right_radius,
            seed=self._last_angle,
        )
        self._last_angle = normal_angle
        self._angles_by_key[key] = normal_angle
        return normal_angle


def _shared_ground_normal_angle(
    left_center: np.ndarray | DualVec3,
    left_axis: np.ndarray | DualVec3,
    left_radius: float,
    right_center: np.ndarray | DualVec3,
    right_axis: np.ndarray | DualVec3,
    right_radius: float,
    continuation: _GroundNormalContinuation | None = None,
) -> float | DualScalar:
    """Solve the ground-normal angle and propagate it by implicit differentiation."""
    primal_left_center = _primal_vector(left_center)
    primal_left_axis = _primal_vector(left_axis)
    primal_right_center = _primal_vector(right_center)
    primal_right_axis = _primal_vector(right_axis)
    solve = (
        continuation.solve
        if continuation is not None
        else _search_ground_normal_angle
    )
    normal_angle = solve(
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
        return normal_angle

    input_residual = _shared_plane_residual(
        left_center,
        left_axis,
        left_radius,
        right_center,
        right_axis,
        right_radius,
        DualScalar(normal_angle),
    )
    assert isinstance(input_residual, DualScalar)
    _, angle_slope = _ground_residual_and_slope(
        primal_left_center,
        primal_left_axis,
        left_radius,
        primal_right_center,
        primal_right_axis,
        right_radius,
        normal_angle,
    )
    derivative_threshold = _root_derivative_threshold(
        primal_left_center, left_radius, primal_right_center, right_radius
    )
    if abs(angle_slope) <= derivative_threshold:
        raise ValueError("Shared wheel-ground tangent is locally singular")
    return DualScalar(normal_angle, -input_residual.deriv / angle_slope)


def get_wheel_ground_tangent(
    positions: Mapping[PointKey, Any], tire_radius: float
) -> Point3 | DualVec3:
    """Return a standalone wheel tangent against the flat +Z ground normal."""
    wheel_center = _as_ground_vector(positions[PointID.WHEEL_CENTER])
    spin_axis = _wheel_spin_axis(positions, PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)
    return _make_position(
        _wheel_plane_support_point(
            wheel_center, spin_axis, tire_radius, np.array((0.0, 0.0, 1.0))
        )
    )


def get_axle_wheel_ground_tangent(
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
    side: Literal["left", "right"],
    continuation: _GroundNormalContinuation | None = None,
) -> Point3 | DualVec3:
    """Compute one side's tangent to the axle's common zero-grade ground plane."""
    left_center_vector = _as_ground_vector(positions[left_center])
    right_center_vector = _as_ground_vector(positions[right_center])
    left_axis = _wheel_spin_axis(positions, left_axis_inboard, left_axis_outboard)
    right_axis = _wheel_spin_axis(positions, right_axis_inboard, right_axis_outboard)
    ground_normal = _ground_normal(
        _shared_ground_normal_angle(
            left_center_vector,
            left_axis,
            left_radius,
            right_center_vector,
            right_axis,
            right_radius,
            continuation,
        )
    )
    if side == "left":
        support = _wheel_plane_support_point(
            left_center_vector, left_axis, left_radius, ground_normal
        )
    else:
        support = _wheel_plane_support_point(
            right_center_vector, right_axis, right_radius, ground_normal
        )
    return _make_position(support)
