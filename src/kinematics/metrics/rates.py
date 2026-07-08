"""
Derivative metrics: motion ratios and kinematic rates.

These metrics are first derivatives of geometry with respect to the sweep
inputs -- camber gain, bump steer, motion ratios -- computed exactly from the
solution-manifold tangents (kinematics.sensitivity), never by finite
differencing adjacent sweep steps.

The pipeline per solved state is:

1. classify each sweep target as a driver (a bump driver commands a wheel
   center along Z; a rack driver commands a trackrod inboard point along Y);
2. seed the state's positions with the driver's tangent velocity field
   (a directional dual-number seed);
3. evaluate the dual-safe metric kernels (kinematics.metrics.kernels) once
   per driver: the .deriv of each result is the exact rate.

Modal rates (response to roll and heave of an axle) are linear combinations
of the per-side bump tangents: commanding left wheel +dz and right wheel -dz
is a roll input, so d(metric)/d(roll) falls out of the same tangents with no
extra solves.

Conventions:

- Bump rates are per millimetre of upward (+Z) wheel center travel of the
  named corner, regardless of the sweep target's authored direction sign.
- Rack rates are per millimetre of +Y (leftward) rack travel.
- Roll is the rotation of the wheel pair relative to the chassis,
  right-hand-rule about +X: positive roll puts the LEFT wheel into bump and
  the right wheel into droop. Roll rates are per degree of that rotation.
- Heave rates are per millimetre of simultaneous equal bump on both sides.
- The damper motion ratio follows the installation-ratio convention
  MR = d(damper compression) / d(wheel bump travel), so a damper that
  shortens in bump has a positive ratio (typically below 1 for outboard
  coilovers). Wheel rate = spring rate * MR^2.
"""

from __future__ import annotations

from collections import OrderedDict
from math import pi
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from kinematics.core.dual import DualScalar, seed_positions_with_tangent
from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import extract_array
from kinematics.core.point_ref import PointKey, PointRef, Side
from kinematics.metrics import kernels, registry
from kinematics.sensitivity import TangentField, combine_tangents
from kinematics.targets import resolve_target

if TYPE_CHECKING:
    from kinematics.state import SuspensionState
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
    from kinematics.suspensions.base import Suspension

RateRow = OrderedDict[str, "float | None"]

# A target direction must be essentially axis-aligned to qualify as a driver;
# this is the cosine of the acceptance cone (about 2.5 degrees).
_AXIS_ALIGNMENT_TOLERANCE = 0.999


def _axis_alignment(target, axis: Axis) -> float | None:
    """
    Signed alignment of a target's direction with a world axis.

    Returns the dot product (effectively +1.0 or -1.0) when the target
    direction is aligned or anti-aligned with the axis, else None. The sign
    converts d/d(target value) into d/d(physical displacement along +axis):
    if the target measures -Z, a +1 mm bump changes the target value by -1.
    """
    direction = resolve_target(target.direction).data
    unit = np.zeros(3)
    unit[int(axis)] = 1.0
    alignment = float(np.dot(direction, unit))
    if abs(alignment) < _AXIS_ALIGNMENT_TOLERANCE:
        return None
    return alignment


def _find_bump_driver(
    tangents: Sequence[TangentField],
    wheel_center_key: PointKey,
) -> tuple[TangentField, float] | None:
    """
    Find the tangent whose driving target commands the given wheel center
    along Z. Returns (field, alignment) or None.
    """
    for field in tangents:
        if field.target.point_id != wheel_center_key:
            continue
        alignment = _axis_alignment(field.target, Axis.Z)
        if alignment is not None:
            return field, alignment
    return None


def _find_rack_driver(
    tangents: Sequence[TangentField],
) -> tuple[TangentField, float] | None:
    """
    Find the tangent whose driving target commands a trackrod inboard point
    along Y (the rack input; either side qualifies, the rigid rack couples
    them). Returns (field, alignment) or None.
    """
    for field in tangents:
        point_id = field.target.point_id
        base_point = point_id.point if isinstance(point_id, PointRef) else point_id
        if base_point != PointID.TRACKROD_INBOARD:
            continue
        alignment = _axis_alignment(field.target, Axis.Y)
        if alignment is not None:
            return field, alignment
    return None


def _rate(kernel_result) -> float:
    """
    Extract the derivative from a dual kernel evaluation.
    """
    assert isinstance(kernel_result, DualScalar)
    return float(kernel_result.deriv)


def _scaled(velocities: Mapping[PointKey, np.ndarray], scale: float):
    """
    Scale a velocity field by a constant (used to fold the driver's
    direction sign into the tangent).
    """
    return {point_id: scale * velocity for point_id, velocity in velocities.items()}


def _corner_velocities(
    velocities: Mapping[PointKey, np.ndarray],
    side: Side,
) -> dict[PointKey, np.ndarray]:
    """
    Strip the side tag from an axle velocity field, keeping one corner.

    Mirrors DoubleWishboneAxleSuspension.corner_state: the returned mapping
    is keyed on plain PointID so corner-level kernels and design states can
    consume it. CENTER-side points are dropped (they are chassis-fixed with
    zero velocity).
    """
    corner: dict[PointKey, np.ndarray] = {}
    for point_id, velocity in velocities.items():
        if isinstance(point_id, PointRef) and point_id.side == side:
            corner[point_id.point] = velocity
    return corner


def _corner_bump_rates(
    positions: Mapping[PointKey, object],
    bump_velocities: Mapping[PointKey, np.ndarray],
    side_sign: float,
    suspension: "Suspension",
) -> RateRow:
    """
    All bump-driven rate metrics for one corner.

    Args:
        positions: The corner's solved positions (PointID-keyed).
        bump_velocities: d(position)/d(wheel bump), already sign-folded so
            the derivative is per +1 mm of upward wheel center travel.
        side_sign: +1 left / -1 right.
        suspension: The corner suspension (for has_rocker / has_strut and
            the design state used by rotation metrics).
    """
    duals = seed_positions_with_tangent(positions, bump_velocities)
    row: RateRow = OrderedDict()

    row[registry.CAMBER_GAIN.key] = _rate(kernels.camber_deg(duals, side_sign))
    row[registry.ROADWHEEL_ANGLE_VS_BUMP.key] = _rate(
        kernels.roadwheel_angle_deg(duals, side_sign)
    )
    row[registry.CASTER_GAIN.key] = _rate(kernels.caster_deg(duals))
    row[registry.KPI_GAIN.key] = _rate(kernels.kpi_deg(duals, side_sign))

    # Half-track rate: outward-positive lateral speed of the contact patch.
    # The lateral coordinate is |y|, so the outward direction is +Y on the
    # left (side_sign +1) and -Y on the right.
    contact_patch_lateral = _rate(
        kernels.coordinate(duals, PointID.CONTACT_PATCH_CENTER, Axis.Y)
    )
    row[registry.HALF_TRACK_RATE.key] = side_sign * contact_patch_lateral

    # Recession: rearward-positive longitudinal speed of the contact patch.
    contact_patch_longitudinal = _rate(
        kernels.coordinate(duals, PointID.CONTACT_PATCH_CENTER, Axis.X)
    )
    row[registry.WHEEL_RECESSION_RATE.key] = -contact_patch_longitudinal

    if suspension.has_strut:
        # Installation ratio: damper compression per mm of bump. The length
        # derivative is negative for a damper that shortens in bump, so the
        # ratio is its negation.
        row[registry.DAMPER_MOTION_RATIO.key] = -_rate(kernels.strut_length_mm(duals))

    if suspension.has_rocker:
        design = suspension.initial_state()
        axis_front = extract_array(design.positions[PointID.ROCKER_AXIS_FRONT])
        axis_rear = extract_array(design.positions[PointID.ROCKER_AXIS_REAR])
        axis_direction = axis_rear - axis_front
        axis_direction = axis_direction / np.linalg.norm(axis_direction)

        # Same side normalisation as the rocker angle metric: symmetric
        # heave reports equal ratios on both sides.
        rocker_rate = side_sign * _rate(
            kernels.rotation_about_fixed_axis_deg(
                duals,
                PointID.PUSHROD_INBOARD,
                extract_array(design.positions[PointID.PUSHROD_INBOARD]),
                axis_front,
                axis_direction,
            )
        )
        row[registry.ROCKER_MOTION_RATIO.key] = rocker_rate
        # Declared alias: the torsion bar is coaxial with the rocker pivot.
        row[registry.TORSION_BAR_MOTION_RATIO.key] = rocker_rate

    return row


def _corner_rack_rates(
    positions: Mapping[PointKey, object],
    rack_velocities: Mapping[PointKey, np.ndarray],
    side_sign: float,
) -> RateRow:
    """
    Rack-driven (steer) rate metrics for one corner, per +Y mm of rack
    travel.
    """
    duals = seed_positions_with_tangent(positions, rack_velocities)
    row: RateRow = OrderedDict()
    row[registry.ROADWHEEL_ANGLE_VS_RACK.key] = _rate(
        kernels.roadwheel_angle_deg(duals, side_sign)
    )
    row[registry.CAMBER_VS_RACK.key] = _rate(kernels.camber_deg(duals, side_sign))
    return row


def compute_corner_rate_metrics(
    state: "SuspensionState",
    suspension: "Suspension",
    tangents: Sequence[TangentField],
) -> RateRow:
    """
    Rate metrics for a single-corner model.

    Emits bump rates when the sweep drives this corner's wheel center in Z,
    and rack rates when it drives the trackrod inboard point in Y. Columns
    for absent drivers are omitted entirely (their presence is constant
    across a sweep, keeping export columns consistent).

    Args:
        state: The solved corner state (PointID-keyed).
        suspension: The corner suspension.
        tangents: Tangent fields for this state, from
            kinematics.sensitivity.compute_state_tangents.

    Returns:
        Ordered mapping of rate column names to values.
    """
    side_sign = (
        -1.0 if float(state.positions[PointID.AXLE_OUTBOARD][Axis.Y]) < 0 else 1.0
    )

    row: RateRow = OrderedDict()

    bump = _find_bump_driver(tangents, PointID.WHEEL_CENTER)
    if bump is not None:
        field, alignment = bump
        row.update(
            _corner_bump_rates(
                state.positions,
                _scaled(field.velocities, alignment),
                side_sign,
                suspension,
            )
        )

    rack = _find_rack_driver(tangents)
    if rack is not None:
        field, alignment = rack
        row.update(
            _corner_rack_rates(
                state.positions,
                _scaled(field.velocities, alignment),
                side_sign,
            )
        )

    return row


def compute_axle_rate_metrics(
    state: "SuspensionState",
    axle: "DoubleWishboneAxleSuspension",
    tangents: Sequence[TangentField],
) -> RateRow:
    """
    Rate metrics for an axle model: per-side corner rates plus modal (roll
    and heave) rates.

    Per-side bump and rack rates reuse the corner machinery on side-stripped
    positions and velocities, prefixed left_/right_. When both wheel centers
    are driven in Z, modal roll/heave tangents are formed as linear
    combinations of the two bump tangents:

        heave: both wheels +1 mm            -> v_heave = v_left + v_right
        roll:  left +track/2 * d(phi),
               right -track/2 * d(phi)      -> v_roll per degree of phi

    with phi the wheel-pair rotation about +X (positive = left wheel up).

    Args:
        state: The solved axle state (PointRef-keyed).
        axle: The axle suspension.
        tangents: Tangent fields for this state (PointRef-keyed velocities).

    Returns:
        Ordered mapping of rate column names to values.
    """
    row: RateRow = OrderedDict()

    side_signs = {Side.LEFT: 1.0, Side.RIGHT: -1.0}
    bump_fields: dict[Side, tuple[TangentField, float]] = {}

    for side in (Side.LEFT, Side.RIGHT):
        bump = _find_bump_driver(tangents, PointRef(side, PointID.WHEEL_CENTER))
        if bump is not None:
            bump_fields[side] = bump

    rack = _find_rack_driver(tangents)

    # Per-side corner rates on side-stripped positions and velocities.
    for side in (Side.LEFT, Side.RIGHT):
        prefix = f"{side.name.lower()}_"
        corner_state = axle.corner_state(state, side)
        corner = axle.corners[side]

        if side in bump_fields:
            field, alignment = bump_fields[side]
            velocities = _scaled(_corner_velocities(field.velocities, side), alignment)
            for column, value in _corner_bump_rates(
                corner_state.positions, velocities, side_signs[side], corner
            ).items():
                row[prefix + column] = value

            # ARB motion ratio: arm rotation about the shared chassis-fixed
            # axis per mm of this corner's bump. Evaluated on the full axle
            # state (the droplink ARB end and the axis are axle-level points),
            # side-sign normalised like the rocker ratio so mirrored corners
            # report equal ratios in symmetric heave.
            if axle.has_arb:
                full_velocities = _scaled(field.velocities, alignment)
                row[prefix + registry.ARB_MOTION_RATIO.key] = side_signs[
                    side
                ] * _arb_arm_rate(state, axle, full_velocities, side)

        if rack is not None:
            field, alignment = rack
            velocities = _scaled(_corner_velocities(field.velocities, side), alignment)
            for column, value in _corner_rack_rates(
                corner_state.positions, velocities, side_signs[side]
            ).items():
                row[prefix + column] = value

    # Modal rates need both bump drivers.
    if len(bump_fields) == 2:
        left_field, left_alignment = bump_fields[Side.LEFT]
        right_field, right_alignment = bump_fields[Side.RIGHT]

        # Current track from the contact patches sets the roll lever arm.
        left_cp = extract_array(
            state.positions[PointRef(Side.LEFT, PointID.CONTACT_PATCH_CENTER)]
        )
        right_cp = extract_array(
            state.positions[PointRef(Side.RIGHT, PointID.CONTACT_PATCH_CENTER)]
        )
        track = abs(float(left_cp[Axis.Y]) - float(right_cp[Axis.Y]))

        # One degree of roll (right-hand rule about +X) lifts the left wheel
        # by (track/2) * pi/180 mm and drops the right wheel by the same.
        half_track_per_degree = (track / 2.0) * (pi / 180.0)
        roll_velocities = combine_tangents(
            [left_field, right_field],
            [
                left_alignment * half_track_per_degree,
                -right_alignment * half_track_per_degree,
            ],
        )
        heave_velocities = combine_tangents(
            [left_field, right_field],
            [left_alignment, right_alignment],
        )

        for side in (Side.LEFT, Side.RIGHT):
            prefix = f"{side.name.lower()}_"
            corner_state = axle.corner_state(state, side)
            sign = side_signs[side]

            roll_duals = seed_positions_with_tangent(
                corner_state.positions, _corner_velocities(roll_velocities, side)
            )
            row[prefix + registry.ROADWHEEL_ANGLE_VS_ROLL.key] = _rate(
                kernels.roadwheel_angle_deg(roll_duals, sign)
            )
            row[prefix + registry.CAMBER_VS_ROLL.key] = _rate(
                kernels.camber_deg(roll_duals, sign)
            )

            heave_duals = seed_positions_with_tangent(
                corner_state.positions, _corner_velocities(heave_velocities, side)
            )
            row[prefix + registry.ROADWHEEL_ANGLE_VS_HEAVE.key] = _rate(
                kernels.roadwheel_angle_deg(heave_duals, sign)
            )
            row[prefix + registry.CAMBER_VS_HEAVE.key] = _rate(
                kernels.camber_deg(heave_duals, sign)
            )

        if axle.has_arb:
            row[registry.ARB_TWIST_VS_ROLL.key] = _arb_arm_rate(
                state, axle, roll_velocities, Side.LEFT
            ) - _arb_arm_rate(state, axle, roll_velocities, Side.RIGHT)

    return row


def _arb_arm_rate(
    state: "SuspensionState",
    axle: "DoubleWishboneAxleSuspension",
    velocities: Mapping[PointKey, np.ndarray],
    side: Side,
) -> float:
    """
    Rate of one ARB arm's rotation along a tangent field (degrees per unit
    of the driving input).

    The arm angle is the RAW signed rotation of the side's ``DROPLINK_ARB``
    about the shared authored A -> B axis (no side normalisation), mirroring
    the ARB angle metrics; the ARB twist rate is the left rate minus the
    right rate along the same tangent.
    """
    design = axle.initial_state()
    axis_a = extract_array(design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_A)])
    axis_b = extract_array(design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_B)])
    axis_direction = axis_b - axis_a
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    duals = seed_positions_with_tangent(state.positions, velocities)

    key = PointRef(side, PointID.DROPLINK_ARB)
    return _rate(
        kernels.rotation_about_fixed_axis_deg(
            duals,
            key,
            extract_array(design.positions[key]),
            axis_a,
            axis_direction,
        )
    )
