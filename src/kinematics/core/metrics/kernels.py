"""
Dual-safe metric kernels.

Each kernel is a pure function from a positions mapping to a scalar, written
so it evaluates identically under two substrates:

- plain float positions (ndarray / Point3): returns a float, matching the
  corresponding catalog metric exactly;
- dual-number positions (DualVec3, e.g. seeded with a solution-manifold
  tangent from kinematics.core.sensitivity): returns a DualScalar whose .deriv is
  the metric's exact rate along the tangent.

That second mode is how derivative metrics (camber gain, bump steer, motion
ratios) are computed without finite differences: one dual evaluation per
kernel per tangent.

To stay substrate-generic the kernels avoid the float-only geometry wrappers
(Point3 / Vector3 / Direction3) and route all math through the dual-aware
helpers in kinematics.core.primitives.dual. Angle kernels feed unnormalized direction
vectors straight into atan2: atan2(k*y, k*x) is identical to atan2(y, x) for
any positive scale k, as a function of the positions, so values AND
derivatives are unaffected and the normalisation step can be skipped.

Sign conventions mirror the catalog metrics in ``kinematics.core.metrics.angles``.
All supplied positions and axes are expressed in chassis coordinates using the
ISO 8855 vehicle-axis orientation (X forward, Y left, Z up; ``side_sign`` +1
left / -1 right). These kernels do not reference road space or world space.
"""

from __future__ import annotations

from typing import Mapping, Union

import numpy as np

from kinematics.core.enums import Axis
from kinematics.core.primitives.dual import (
    DualScalar,
    DualVec3,
    atan2,
    cross,
    degrees,
    dot,
)
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey

# A kernel input position: raw array, geometry wrapper, or dual vector.
PositionLike = Union[np.ndarray, DualVec3, object]
# A kernel result: float on the plain substrate, DualScalar on the dual one.
Scalar = Union[float, DualScalar]

# Constant chassis axes as raw arrays (constants have zero derivative).
_X_AXIS = np.array([1.0, 0.0, 0.0])


def _vec(positions: Mapping[PointKey, PositionLike], point_id: PointKey):
    """
    Fetch a position as a dual-safe vector: DualVec3 passes through,
    anything else is unwrapped to its raw ndarray.
    """
    value = positions[point_id]
    if isinstance(value, DualVec3):
        return value
    return extract_array(value)


def rotation_about_fixed_axis_deg(
    positions: Mapping[PointKey, PositionLike],
    point: PointKey,
    design_position: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
) -> Scalar:
    """Return signed rotation about a fixed chassis-space axis.

    The point, design position and fixed axis are all expressed in chassis
    coordinates. The angular result is invariant under a rigid world transform
    and has no road-plane reference.
    """
    design_radius = np.asarray(design_position) - axis_point
    current_radius = _vec(positions, point) - axis_point
    design_perpendicular = (
        design_radius - np.dot(design_radius, axis_direction) * axis_direction
    )
    current_perpendicular = (
        current_radius - dot(current_radius, axis_direction) * axis_direction
    )
    sine = dot(axis_direction, cross(design_radius, current_radius))
    cosine = dot(design_perpendicular, current_perpendicular)
    return degrees(atan2(sine, cosine))


def _component(vector, axis: Axis) -> Scalar:
    """
    Single component on either substrate (DualScalar or float).
    """
    value = vector[int(axis)]
    if isinstance(value, DualScalar):
        return value
    return float(value)


def coordinate(
    positions: Mapping[PointKey, PositionLike],
    point_id: PointKey,
    axis: Axis,
) -> Scalar:
    """
    A single chassis-space coordinate of a point (mm).
    """
    return _component(_vec(positions, point_id), axis)


def camber_deg(
    positions: Mapping[PointKey, PositionLike],
    side_sign: float,
    axle_inboard: PointKey,
    axle_outboard: PointKey,
) -> Scalar:
    """
    Camber angle in degrees; mirrors metrics.angles.calculate_camber.

    The wheel-axis endpoints are supplied by the caller, normally from
    ``CornerSuspension.wheel_axis_points()``. The result resolves the wheel
    axis in the chassis YZ plane against chassis +Z; it does not use road or
    world space. Negative camber tilts the top of the wheel towards the vehicle
    centerline.
    """
    axle = _vec(positions, axle_outboard) - _vec(positions, axle_inboard)

    # Wheel "up" vector: perpendicular to both the axle and vehicle X.
    # Multiplying by -side keeps it pointing roughly +Z on both sides.
    wheel_up = cross(axle, _X_AXIS) * (-side_sign)

    # Front-view (YZ) angle from vertical.
    angle = atan2(_component(wheel_up, Axis.Y), _component(wheel_up, Axis.Z))

    # Right side: inward tilt is +Y which flips the sign; invert to match
    # the shared convention.
    camber = angle if side_sign > 0 else -angle
    return degrees(camber)


def toe_deg(
    positions: Mapping[PointKey, PositionLike],
    side_sign: float,
    axle_inboard: PointKey,
    axle_outboard: PointKey,
) -> Scalar:
    """
    Toe angle in degrees; mirrors metrics.angles.calculate_toe.

    The wheel-axis endpoints are supplied by the caller, normally from
    ``CornerSuspension.wheel_axis_points()``. The result resolves wheel heading
    in the chassis XY plane relative to chassis +X; it does not use road or
    world space. Positive is toe-in: the front of the wheel points towards the
    vehicle centerline.
    """
    axle = _vec(positions, axle_outboard) - _vec(positions, axle_inboard)
    proj_x = _component(axle, Axis.X)
    proj_y = _component(axle, Axis.Y)

    # Toe-in points the axle vector slightly forward (+X) on the left,
    # measured against +Y; on the right, against -Y.
    if side_sign > 0:
        toe = atan2(proj_x, proj_y)
    else:
        toe = atan2(proj_x, -proj_y)
    return degrees(toe)


def steer_deg(
    positions: Mapping[PointKey, PositionLike],
    side_sign: float,
    axle_inboard: PointKey,
    axle_outboard: PointKey,
) -> Scalar:
    """ISO 8855 steer angle in degrees; mirrors ``calculate_steer``.

    The wheel-forward vector is constructed from the inboard-to-outboard spin
    axis, then its vehicle-fixed right-hand-rule heading is taken about +Z.
    A left turn is positive for both corners.
    """
    axle = _vec(positions, axle_outboard) - _vec(positions, axle_inboard)
    forward_x = side_sign * _component(axle, Axis.Y)
    forward_y = -side_sign * _component(axle, Axis.X)
    return degrees(atan2(forward_y, forward_x))


def caster_deg(
    positions: Mapping[PointKey, PositionLike],
    lower_pivot: PointKey,
    upper_pivot: PointKey,
) -> Scalar:
    """
    Caster angle in degrees; mirrors metrics.angles.calculate_caster.

    The steering-axis pivots are architecture-specific and must be supplied by
    the caller, normally from ``CornerSuspension.steering_axis_points()``. The
    result resolves the axis in the chassis XZ plane against chassis +Z; it
    does not use road or world space. Positive caster tilts the top rearward.
    """
    steering = _vec(positions, upper_pivot) - _vec(positions, lower_pivot)
    # Side-view (XZ) angle from vertical; negate X so rearward tilt of the
    # top is positive.
    caster = atan2(-_component(steering, Axis.X), _component(steering, Axis.Z))
    return degrees(caster)


def kpi_deg(
    positions: Mapping[PointKey, PositionLike],
    side_sign: float,
    lower_pivot: PointKey,
    upper_pivot: PointKey,
) -> Scalar:
    """
    Kingpin inclination in degrees; mirrors metrics.angles.calculate_kpi.

    The steering-axis pivots are architecture-specific and must be supplied by
    the caller, normally from ``CornerSuspension.steering_axis_points()``. The
    result resolves the axis in the chassis YZ plane against chassis +Z; it
    does not use road or world space. Positive KPI tilts the top towards the
    vehicle centerline.
    """
    steering = _vec(positions, upper_pivot) - _vec(positions, lower_pivot)
    # Front-view (YZ) angle from vertical; -side folds left/right into the
    # shared inward-positive convention.
    kpi = atan2(
        -side_sign * _component(steering, Axis.Y),
        _component(steering, Axis.Z),
    )
    return degrees(kpi)
