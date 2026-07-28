"""Topology-independent modal metrics for a solved two-corner axle."""

from __future__ import annotations

from math import atan2, degrees
from typing import TYPE_CHECKING

from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension


def append_axle_state_metrics(
    row: MetricRow,
    state: SuspensionState,
    axle: AxleSuspension,
    ground: GroundDatum,
    design_ground: GroundDatum | None,
) -> None:
    """Append axle-scoped metrics from the current and design ground data."""
    wheel_delta_z: dict[Side, float] = {}
    tangents: dict[Side, Point3] = {}
    for side in (Side.LEFT, Side.RIGHT):
        design = axle.corners[side].initial_state()
        wheel_ref = PointRef(side, PointID.WHEEL_CENTER)
        tangent_ref = PointRef(side, PointID.WHEEL_GROUND_TANGENT)
        wheel_delta_z[side] = float(state.get(wheel_ref)[Axis.Z]) - float(
            design.get(PointID.WHEEL_CENTER)[Axis.Z]
        )
        tangents[side] = state.get(tangent_ref)

    left_wheel_z = wheel_delta_z[Side.LEFT]
    right_wheel_z = wheel_delta_z[Side.RIGHT]
    track = abs(
        float(ground.lateral.dot(tangents[Side.LEFT] - tangents[Side.RIGHT]))
    )
    row["heave"] = 0.5 * (left_wheel_z + right_wheel_z)
    row["roll"] = degrees(atan2(left_wheel_z - right_wheel_z, track))
    chassis_origin = Point3((0.0, 0.0, 0.0))
    row["ride_height_change"] = (
        ground.signed_distance(chassis_origin)
        - design_ground.signed_distance(chassis_origin)
        if design_ground is not None
        else None
    )
    row["track"] = track

    roll_center_y, roll_center_z = _roll_center(state, axle)
    row["roll_center_y"] = roll_center_y
    row["roll_center_z"] = roll_center_z

    # Rack displacement is measured on the left corner rack attachment; the rack
    # coupling makes the right corner move identically.
    left_corner = axle.corners[Side.LEFT]
    rack_attachment = left_corner.rack_attachment_point()
    if rack_attachment is None:
        row["rack_displacement"] = None
    else:
        design_rack_y = float(left_corner.initial_state().get(rack_attachment)[Axis.Y])
        current_rack_y = float(state.get(PointRef(Side.LEFT, rack_attachment))[Axis.Y])
        row["rack_displacement"] = current_rack_y - design_rack_y


def _roll_center(
    state: SuspensionState,
    axle: AxleSuspension,
) -> tuple[float | None, float | None]:
    """Intersect the two wheel-plane-ground-tangent-to-FVIC lines in the YZ plane."""
    lines: list[tuple[float, float, float, float]] = []
    for side in (Side.LEFT, Side.RIGHT):
        corner_state = axle.corner_state(state, side)
        fvic = axle.corners[side].compute_front_view_instant_center(corner_state)
        if fvic is None:
            return None, None
        ground_tangent = corner_state.get(PointID.WHEEL_GROUND_TANGENT)
        tangent_y = float(ground_tangent[Axis.Y])
        tangent_z = float(ground_tangent[Axis.Z])
        lines.append(
            (
                tangent_y,
                tangent_z,
                float(fvic[Axis.Y]) - tangent_y,
                float(fvic[Axis.Z]) - tangent_z,
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
