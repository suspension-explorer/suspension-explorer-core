"""Topology-independent modal metrics for a solved two-corner axle."""

from __future__ import annotations

from math import atan2, degrees
from typing import TYPE_CHECKING

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
    """Append heave, roll, ride-height, track, and steering metrics."""
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
    row["heave_mm"] = 0.5 * (left_wheel_z + right_wheel_z)
    row["roll_deg"] = degrees(atan2(left_wheel_z - right_wheel_z, track))
    row["ride_height_change_mm"] = -0.5 * (
        contact_delta_z[Side.LEFT] + contact_delta_z[Side.RIGHT]
    )
    row["track_mm"] = track


__all__ = ["append_axle_state_metrics"]
