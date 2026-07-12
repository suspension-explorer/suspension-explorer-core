"""Topology-independent modal metrics for a solved two-corner axle."""

from __future__ import annotations

from math import atan2, degrees
from typing import TYPE_CHECKING

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side

if TYPE_CHECKING:
    from kinematics.metrics.main import MetricRow
    from kinematics.state import SuspensionState
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension


def append_axle_state_metrics(
    row: MetricRow,
    state: SuspensionState,
    axle: DoubleWishboneAxleSuspension,
) -> None:
    """Append all metrics defined at axle rather than corner scope."""
    wheel_delta_z: dict[Side, float] = {}
    contact_delta_z: dict[Side, float] = {}
    contact_y: dict[Side, float] = {}
    for side in (Side.LEFT, Side.RIGHT):
        design = axle.corners[side].initial_state()
        wheel_ref = PointRef(side, PointID.WHEEL_CENTER)
        contact_ref = PointRef(side, PointID.CONTACT_PATCH_CENTER)
        wheel_delta_z[side] = float(state.get(wheel_ref)[Axis.Z]) - float(
            design.get(PointID.WHEEL_CENTER)[Axis.Z]
        )
        contact_delta_z[side] = float(state.get(contact_ref)[Axis.Z]) - float(
            design.get(PointID.CONTACT_PATCH_CENTER)[Axis.Z]
        )
        contact_y[side] = float(state.get(contact_ref)[Axis.Y])

    left_wheel_z = wheel_delta_z[Side.LEFT]
    right_wheel_z = wheel_delta_z[Side.RIGHT]
    track = abs(contact_y[Side.LEFT] - contact_y[Side.RIGHT])
    row["heave"] = 0.5 * (left_wheel_z + right_wheel_z)
    row["roll"] = degrees(atan2(left_wheel_z - right_wheel_z, track))
    row["ride_height_change"] = -0.5 * (
        contact_delta_z[Side.LEFT] + contact_delta_z[Side.RIGHT]
    )
    row["track"] = track

    roll_center_y, roll_center_z = _roll_center(state, axle)
    row["roll_center_y"] = roll_center_y
    row["roll_center_z"] = roll_center_z

    design_trackrod_y = float(
        axle.corners[Side.LEFT].initial_state().get(PointID.TRACKROD_INBOARD)[Axis.Y]
    )
    current_trackrod_y = float(
        state.get(PointRef(Side.LEFT, PointID.TRACKROD_INBOARD))[Axis.Y]
    )
    row["trackrod_inboard_displacement"] = current_trackrod_y - design_trackrod_y


def _roll_center(
    state: SuspensionState,
    axle: DoubleWishboneAxleSuspension,
) -> tuple[float | None, float | None]:
    """Intersect the two contact-patch-to-FVIC lines in the YZ plane."""
    lines: list[tuple[float, float, float, float]] = []
    for side in (Side.LEFT, Side.RIGHT):
        corner_state = axle.corner_state(state, side)
        fvic = axle.corners[side].compute_front_view_instant_center(corner_state)
        if fvic is None:
            return None, None
        contact = corner_state.get(PointID.CONTACT_PATCH_CENTER)
        contact_y = float(contact[Axis.Y])
        contact_z = float(contact[Axis.Z])
        lines.append(
            (
                contact_y,
                contact_z,
                float(fvic[Axis.Y]) - contact_y,
                float(fvic[Axis.Z]) - contact_z,
            )
        )

    left, right = lines
    denominator = left[2] * right[3] - left[3] * right[2]
    if abs(denominator) < EPS_GEOMETRIC:
        return None, None
    parameter = (
        (right[0] - left[0]) * right[3] - (right[1] - left[1]) * right[2]
    ) / denominator
    return left[0] + parameter * left[2], left[1] + parameter * left[3]
