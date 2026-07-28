"""Wheel-plane road tangency: flat-ground corner tangents and the coupled axle solve.

Why this module exists
======================

Every authored and solved coordinate in this package is chassis-fixed: ``+X``
forward, ``+Y`` left, ``+Z`` up, hardpoints stationary. There is no road in
that frame — the ground is wherever the tires are. The only geometric evidence
the model has about the road is that each rigid wheel disc must touch it, so
the road must be *inferred* from wheel tangency rather than authored.

For a standalone corner there is not enough information to orient a road: the
corner assumes a flat, horizontal road (``+Z`` normal) and its road contact is
simply the lowest point of the wheel disc, :func:`get_wheel_ground_tangent`.

For a two-corner axle the assumption of one horizontal plane per corner breaks
down: on an asymmetric state (roll, one-wheel bump), each corner's "flat road"
sits at a different height, and any consumer comparing the two sides — roll
centre construction, ride height, CG height — silently mixes two different
ground references. The fix is one shared ground plane that is *simultaneously
tangent to both wheel discs*. That is what :func:`solve_axle_wheel_ground_tangents`
computes.

The mathematics
===============

The shared plane is parameterised by a single scalar, the ground-normal angle
``theta``, giving the plane normal ``n = (0, sin(theta), cos(theta))``.

.. note::
    The zero ``X`` component is a deliberate modelling assumption: with only
    one axle modelled, longitudinal road grade is unknowable, so the plane is
    the axle's YZ ground line extruded along chassis ``±X`` — **zero
    longitudinal gradient is assumed throughout**. Pitch is reintroduced
    post hoc, as an interpretation, by :mod:`kinematics.core.pose`; it is
    never part of this tangency solve.

For one wheel with centre ``C``, unit spin axis ``a``, and nominal radius
``R``, the point of the wheel plane that touches a plane with unit normal
``n`` is the disc's support point in the ``-n`` direction, restricted to the
wheel plane::

    q = n - (n . a) a          # road normal projected into the wheel plane
    P = C - R q / ||q||        # wheel-plane road-tangent point

Both discs touch one plane when their support points have equal height along
the normal, giving the scalar residual whose root is solved here::

    f(theta) = n(theta) . P_left(theta) - n(theta) . P_right(theta) = 0

Because ``P`` itself depends on ``theta`` (camber and toe rotate the support
point around the disc), ``f`` is transcendental; there is no closed form once
the spin axes are non-trivial. The solve is a Newton iteration whose exact
slope comes from a dual-number evaluation of the residual, with a
bracketed-bisection fallback scanning the admissible angle domain. The domain
is clamped to ±80 degrees of normal tilt — a product policy bounding the
supported geometry, not an intrinsic singularity.

Root selection and the seed
===========================

``f`` can have several roots for exotic geometries (strongly cambered, narrow
axles). To keep a sweep on one physical branch, callers thread the previously
accepted root through the ``seed`` parameter; the solver converges to, or
selects, the root nearest that seed. With no seed, the estimate from the two
independent flat-ground tangents is used. This threading is *explicit and
stateless*: the function has no memory, and identical inputs plus an identical
seed always reproduce the same root. (An earlier design kept a hidden
continuation cache inside the derived-point graph; the post-solve closure
made that state unnecessary.)

How the results are used
========================

:meth:`AxleSuspension.apply_ground_closure` calls
:func:`solve_axle_wheel_ground_tangents` once per accepted solver state — a
post-solve closure, not a solver constraint — and writes the two tangent
points into the state under ``WHEEL_GROUND_TANGENT``. Those stored contacts
are then the geometric inputs to the post-solve WorldSpace placement. Metrics
consume the resulting complete 3D World ground plane; they do not publish a
second scalar ground-line representation.

The scalar solved here rotates the intermediate plane's *normal*, hence the
consistent internal name ``normal_angle``.

Dual-number inputs
==================

All entry points accept either raw ``numpy`` vectors or :class:`DualVec3`
positions. With dual inputs, the primal root is solved from the stripped
values and the root's sensitivity is attached by implicit differentiation of
the residual — this is what propagates solution-manifold tangents through the
tangency solve for derivative metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi
from typing import Any, Callable, Mapping, TypeVar

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
        raise ValueError(f"Wheel-ground tangent produced non-finite {description}")
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
        # Orient the separation from vehicle right to left. Without this,
        # laterally crossed supports give an estimate near +/-pi, which the
        # clamp then turns into a seed unrelated to the geometry.
        lateral_separation = -lateral_separation
        vertical_separation = -vertical_separation
    return _clamp_ground_normal_angle(-atan2(vertical_separation, lateral_separation))


def seed_from_tangent_points(
    left_tangent: np.ndarray | DualVec3 | Point3,
    right_tangent: np.ndarray | DualVec3 | Point3,
) -> float | None:
    """Recover the ground-normal angle implied by two stored tangent points.

    Accepted states carry the previously solved tangent points, so the angle
    of the line through them is an exact-root seed for the next solve — this
    is how branch continuity survives without any hidden solver state.
    Returns ``None`` when the stored points cannot orient a line.
    """
    left = _primal_vector(
        left_tangent.data if isinstance(left_tangent, Point3) else left_tangent
    )
    right = _primal_vector(
        right_tangent.data if isinstance(right_tangent, Point3) else right_tangent
    )
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    lateral_separation = float(left[Axis.Y] - right[Axis.Y])
    vertical_separation = float(left[Axis.Z] - right[Axis.Z])
    if abs(lateral_separation) <= EPS_GEOMETRIC:
        return None
    if lateral_separation < 0.0:
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

    With no seed, the flat-ground construction provides the starting estimate.
    Callers sweeping through states pass the previously accepted root so a
    multi-root geometry follows one continuous branch instead of independently
    choosing a root at every state.
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
            raise ValueError("Shared wheel-ground tangent has non-finite residual")
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

    if seed is None:
        selection_seed = _flat_ground_normal_angle_estimate(
            left_center,
            left_axis,
            left_radius,
            right_center,
            right_axis,
            right_radius,
        )
    else:
        selection_seed = _clamp_ground_normal_angle(seed)
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


def _shared_ground_normal_angle(
    left_center: np.ndarray | DualVec3,
    left_axis: np.ndarray | DualVec3,
    left_radius: float,
    right_center: np.ndarray | DualVec3,
    right_axis: np.ndarray | DualVec3,
    right_radius: float,
    seed: float | None = None,
) -> float | DualScalar:
    """Solve the ground-normal angle and propagate it by implicit differentiation."""
    primal_left_center = _primal_vector(left_center)
    primal_left_axis = _primal_vector(left_axis)
    primal_right_center = _primal_vector(right_center)
    primal_right_axis = _primal_vector(right_axis)
    normal_angle = _search_ground_normal_angle(
        primal_left_center,
        primal_left_axis,
        left_radius,
        primal_right_center,
        primal_right_axis,
        right_radius,
        seed=seed,
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


@dataclass(frozen=True)
class AxleGroundTangency:
    """One coupled tangency solution: both tangent points and the solved angle.

    ``normal_angle`` is the primal ground-normal angle in radians; it is the
    seed to pass to the next state's solve for branch continuity.
    """

    left: Point3 | DualVec3
    right: Point3 | DualVec3
    normal_angle: float


def solve_axle_wheel_ground_tangents(
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
    seed: float | None = None,
) -> AxleGroundTangency:
    """Solve both sides' tangents to the axle's common zero-grade ground plane.

    Stateless: the only cross-state coupling is the explicit ``seed``. The
    single scalar solve serves both sides, so the two returned points lie on
    one plane by construction.
    """
    left_center_vector = _as_ground_vector(positions[left_center])
    right_center_vector = _as_ground_vector(positions[right_center])
    left_axis = _wheel_spin_axis(positions, left_axis_inboard, left_axis_outboard)
    right_axis = _wheel_spin_axis(positions, right_axis_inboard, right_axis_outboard)
    normal_angle = _shared_ground_normal_angle(
        left_center_vector,
        left_axis,
        left_radius,
        right_center_vector,
        right_axis,
        right_radius,
        seed=seed,
    )
    ground_normal = _ground_normal(normal_angle)
    left_support = _wheel_plane_support_point(
        left_center_vector, left_axis, left_radius, ground_normal
    )
    right_support = _wheel_plane_support_point(
        right_center_vector, right_axis, right_radius, ground_normal
    )
    primal_angle = (
        normal_angle.val if isinstance(normal_angle, DualScalar) else normal_angle
    )
    return AxleGroundTangency(
        left=_make_position(left_support),
        right=_make_position(right_support),
        normal_angle=float(primal_angle),
    )
