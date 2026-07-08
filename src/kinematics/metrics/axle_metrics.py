"""
Axle-level per-state metrics.

Computes chassis-relative modal quantities for a solved axle state: heave,
suspension roll, ride-height change, and steering Ackermann percentage. Each
is measured against the design (as-authored) condition of the two corners.

Sign conventions follow ISO 8855 (X forward, Y left, Z up). Wheel-centre and
contact-patch displacements are chassis-fixed: in that frame the ground follows
the tires, so wheels rising in the chassis means the chassis sits lower.
"""

from __future__ import annotations

from math import atan, atan2, degrees, radians, tan
from typing import TYPE_CHECKING

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.metrics import registry

if TYPE_CHECKING:
    from kinematics.metrics.main import MetricRow
    from kinematics.schema.config import SuspensionConfig
    from kinematics.state import SuspensionState
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension


# Below this steer magnitude (degrees) Ackermann is dominated by parallel-steer
# noise (the two wheels are effectively parallel) and the percentage is
# ill-conditioned, so we report None.
_MIN_ACKERMANN_STEER_DEG = 0.5


def append_axle_state_metrics(
    row: "MetricRow",
    state: "SuspensionState",
    axle: "DoubleWishboneAxleSuspension",
    config: "SuspensionConfig",
    side_rows: "dict[Side, MetricRow]",
) -> None:
    """
    Append heave, roll, ride-height-change, and Ackermann to the axle row.

    Args:
        row: The axle metric row being assembled (mutated in place).
        state: The solved axle state (PointRef-keyed).
        axle: The axle suspension.
        config: Shared vehicle configuration.
        side_rows: The already-computed per-side corner metric rows, keyed by
            side (provides the current ``roadwheel_angle_deg`` per side).
    """
    # Per-side wheel-centre and contact-patch Z displacements from design.
    dz_wheel: dict[Side, float] = {}
    dz_contact: dict[Side, float] = {}
    contact_y: dict[Side, float] = {}
    for side in (Side.LEFT, Side.RIGHT):
        design = axle.corners[side].initial_state()

        wc_now = float(state.positions[PointRef(side, PointID.WHEEL_CENTER)][Axis.Z])
        wc_design = float(design.get(PointID.WHEEL_CENTER)[Axis.Z])
        dz_wheel[side] = wc_now - wc_design

        cp_ref = PointRef(side, PointID.CONTACT_PATCH_CENTER)
        cp_now = float(state.positions[cp_ref][Axis.Z])
        cp_design = float(design.get(PointID.CONTACT_PATCH_CENTER)[Axis.Z])
        dz_contact[side] = cp_now - cp_design

        contact_y[side] = float(state.positions[cp_ref][Axis.Y])

    dz_left = dz_wheel[Side.LEFT]
    dz_right = dz_wheel[Side.RIGHT]

    # Heave: mean vertical wheel-centre displacement. Positive = both wheels up
    # in the chassis frame.
    row[registry.HEAVE.key] = 0.5 * (dz_left + dz_right)

    # Suspension roll of the wheel pair relative to the chassis. The lever is
    # the current contact-patch track; the numerator is the left-minus-right
    # wheel-centre rise. atan2 keeps it well-defined at large angles. Positive
    # = LEFT wheel in bump relative to right (right-hand rule about +X).
    track = abs(contact_y[Side.LEFT] - contact_y[Side.RIGHT])
    row[registry.ROLL.key] = degrees(atan2(dz_left - dz_right, track))

    # Ride-height change: the ground follows the contact patches, so the mean
    # upward contact-patch displacement is negated to give the chassis's change
    # in ride height (wheels up -> chassis lower -> negative).
    row[registry.RIDE_HEIGHT_CHANGE.key] = -0.5 * (
        dz_contact[Side.LEFT] + dz_contact[Side.RIGHT]
    )

    row[registry.ACKERMANN.key] = _ackermann_pct(axle, config, side_rows, track)


def _design_roadwheel_angle(
    axle: "DoubleWishboneAxleSuspension",
    config: "SuspensionConfig",
    side: Side,
) -> float | None:
    """
    Roadwheel (toe) angle in degrees for a side at the design condition.
    """
    # Deferred imports: angles/context import metrics types.
    from kinematics.metrics.angles import calculate_roadwheel_angle
    from kinematics.metrics.context import MetricContext

    corner = axle.corners[side]
    ctx = MetricContext(state=corner.initial_state(), suspension=corner, config=config)
    return calculate_roadwheel_angle(ctx)


def _ackermann_pct(
    axle: "DoubleWishboneAxleSuspension",
    config: "SuspensionConfig",
    side_rows: "dict[Side, MetricRow]",
    track: float,
) -> float | None:
    """
    Steering Ackermann percentage.

    Per side, the yaw-steer angle is the change in roadwheel (toe) angle from
    design. Toe-in is positive toward the centreline on each side, so to get a
    consistent yaw sign (positive = steering to the left) we fold the signs:

        delta_left  = -(roadwheel_angle_left  - roadwheel_angle_left_design)
        delta_right = +(roadwheel_angle_right - roadwheel_angle_right_design)

    mean_steer = (delta_left + delta_right) / 2. If the mean steer magnitude is
    below the parallel-steer threshold the percentage is ill-conditioned -> None.

    Otherwise the inner wheel is the one toward the turn (left when steering
    left). With delta_o the outer steer and delta_i the inner steer (both
    magnitudes, radians), the ideal Ackermann inner angle satisfies

        cot(delta_i_ideal) = cot(delta_o) - track / L

    i.e. delta_i_ideal = atan(1 / (1/tan(delta_o) - track/L)). Then

        ackermann_pct = 100 * (delta_i - delta_o) / (delta_i_ideal - delta_o)

    where 100 = perfect Ackermann, 0 = parallel steer, negative = anti-Ackermann.
    Returns None if any roadwheel angle value is undefined or a denominator is
    degenerate.
    """
    roadwheel_angle_left = side_rows[Side.LEFT].get("roadwheel_angle_deg")
    roadwheel_angle_right = side_rows[Side.RIGHT].get("roadwheel_angle_deg")
    roadwheel_angle_left_design = _design_roadwheel_angle(axle, config, Side.LEFT)
    roadwheel_angle_right_design = _design_roadwheel_angle(axle, config, Side.RIGHT)
    if (
        roadwheel_angle_left is None
        or roadwheel_angle_right is None
        or roadwheel_angle_left_design is None
        or roadwheel_angle_right_design is None
    ):
        return None

    # Yaw-steer per side, positive = steering to the left (see docstring).
    delta_left = -(roadwheel_angle_left - roadwheel_angle_left_design)
    delta_right = roadwheel_angle_right - roadwheel_angle_right_design
    mean_steer = 0.5 * (delta_left + delta_right)
    if abs(mean_steer) < _MIN_ACKERMANN_STEER_DEG:
        # Parallel-steer noise: geometry too close to straight-ahead.
        return None

    # Inner wheel is toward the turn: left when steering left (mean_steer > 0).
    if mean_steer > 0.0:
        delta_inner, delta_outer = delta_left, delta_right
    else:
        delta_inner, delta_outer = delta_right, delta_left

    delta_i = abs(radians(delta_inner))
    delta_o = abs(radians(delta_outer))

    wheelbase = config.wheelbase
    tan_outer = tan(delta_o)
    if abs(tan_outer) < EPS_GEOMETRIC or wheelbase < EPS_GEOMETRIC:
        # Outer wheel essentially straight ahead: ideal inner angle undefined.
        return None

    # cot(delta_i_ideal) = cot(delta_o) - track / L.
    cot_ideal = 1.0 / tan_outer - track / wheelbase
    if abs(cot_ideal) < EPS_GEOMETRIC:
        return None
    delta_i_ideal = atan(1.0 / cot_ideal)

    denom = delta_i_ideal - delta_o
    if abs(denom) < EPS_GEOMETRIC:
        return None
    return 100.0 * (delta_i - delta_o) / denom
