"""Topology-independent modal metrics for a solved two-corner axle."""

from __future__ import annotations

from math import atan2, degrees, hypot, radians, tan
from typing import TYPE_CHECKING

from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics.angles import calculate_steer
from kinematics.core.metrics.context import MetricContext
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.road import RoadPlane

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.axle import AxleSuspension


def append_axle_state_metrics(
    row: MetricRow,
    state: SuspensionState,
    axle: AxleSuspension,
    road: RoadPlane,
    design_road: RoadPlane | None,
) -> None:
    """Append axle metrics using their declared chassis or road references.

    ``heave`` is mean wheel-center chassis Z travel. ``roll`` is ISO 8855
    suspension roll angle (§5.2.5): the inclination of the current line from
    the right wheel center to the left wheel center, relative to chassis XY.
    Positive roll means the left wheel center is higher than the right. It is a
    kinematic axle state, not a solved body attitude. ``ride_height_change`` is
    the change in perpendicular distance from the chassis origin to the
    axle-local road plane. ``track`` is ISO 8855 track (§4.4): the design
    wheel-contact-center separation measured parallel to chassis Y with the
    vehicle at rest on a horizontal surface. ``track_change`` is the generic
    current-minus-design road-lateral difference. ISO 8855 defines the narrower
    ``ride track change`` (§8.1.1) only for symmetric wheel displacement, so
    that name is deliberately not used for arbitrary solved axle states.

    Roll-center coordinates and rack displacement are reported in chassis
    space. The road datum follows the ISO 8855 local or equivalent road-plane
    concept and is expressed in chassis coordinates. None of these metrics uses
    world space or inferred longitudinal pitch.
    """
    wheel_delta_z: dict[Side, float] = {}
    wheel_centers: dict[Side, Point3] = {}
    contact_centers: dict[Side, Point3] = {}
    design_contact_centers: dict[Side, Point3] = {}
    for side in (Side.LEFT, Side.RIGHT):
        design = axle.corners[side].initial_state()
        wheel_ref = PointRef(side, PointID.WHEEL_CENTER)
        contact_center_ref = PointRef(side, PointID.WHEEL_CONTACT_CENTER)
        wheel_delta_z[side] = float(state.get(wheel_ref)[Axis.Z]) - float(
            design.get(PointID.WHEEL_CENTER)[Axis.Z]
        )
        wheel_centers[side] = state.get(wheel_ref)
        contact_centers[side] = state.get(contact_center_ref)
        design_contact_centers[side] = design.get(PointID.WHEEL_CONTACT_CENTER)

    left_wheel_z = wheel_delta_z[Side.LEFT]
    right_wheel_z = wheel_delta_z[Side.RIGHT]
    wheel_center_line = wheel_centers[Side.LEFT] - wheel_centers[Side.RIGHT]
    horizontal_wheel_center_distance = hypot(
        float(wheel_center_line[Axis.X]),
        float(wheel_center_line[Axis.Y]),
    )
    current_contact_separation = abs(
        float(
            road.lateral.dot(contact_centers[Side.LEFT] - contact_centers[Side.RIGHT])
        )
    )
    design_contact_separation = abs(
        float(
            design_road.lateral.dot(
                design_contact_centers[Side.LEFT] - design_contact_centers[Side.RIGHT]
            )
        )
        if design_road is not None
        else float(
            design_contact_centers[Side.LEFT][Axis.Y]
            - design_contact_centers[Side.RIGHT][Axis.Y]
        )
    )
    row["heave"] = 0.5 * (left_wheel_z + right_wheel_z)
    row["roll"] = degrees(
        atan2(float(wheel_center_line[Axis.Z]), horizontal_wheel_center_distance)
    )
    chassis_origin = Point3((0.0, 0.0, 0.0))
    row["ride_height_change"] = (
        road.signed_distance(chassis_origin)
        - design_road.signed_distance(chassis_origin)
        if design_road is not None
        else None
    )
    row["track"] = design_contact_separation
    row["track_change"] = current_contact_separation - design_contact_separation
    row["ackermann_percentage"] = _ackermann_percentage(
        state,
        axle,
        road,
        design_contact_separation,
    )

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


def _ackermann_percentage(
    state: SuspensionState,
    axle: AxleSuspension,
    road: RoadPlane,
    track_mm: float,
) -> float | None:
    """Return geometric Ackermann percentage for the current steered state.

    Perfect Ackermann satisfies `cot(outer) - cot(inner) = track / wheelbase`.
    Parallel steer is zero percent. The result is undefined at or sufficiently
    near straight ahead, or for an axle without a rack actuator.
    """
    if any(corner.rack_attachment_point() is None for corner in axle.corners.values()):
        return None

    steer_deg: dict[Side, float] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner = axle.corners[side]
        if corner.config is None:
            return None
        context = MetricContext(
            axle.corner_state(state, side),
            corner,
            corner.config,
            road=road,
        )
        steer_deg[side] = calculate_steer(context)

    left_config = axle.corners[Side.LEFT].config
    if left_config is None:
        return None
    wheelbase_mm = left_config.wheelbase
    return _ackermann_from_angles(
        steer_deg[Side.LEFT],
        steer_deg[Side.RIGHT],
        track_mm,
        wheelbase_mm,
    )


def _ackermann_from_angles(
    left_steer_deg: float,
    right_steer_deg: float,
    track_mm: float,
    wheelbase_mm: float,
) -> float | None:
    """Return Ackermann percentage from signed left and right steer angles."""
    steer_deg = {Side.LEFT: left_steer_deg, Side.RIGHT: right_steer_deg}
    mean_steer = 0.5 * (left_steer_deg + right_steer_deg)
    if abs(mean_steer) < EPS_GEOMETRIC:
        return None
    inner_side = Side.LEFT if mean_steer > 0.0 else Side.RIGHT
    outer_side = Side.RIGHT if inner_side is Side.LEFT else Side.LEFT
    inner_tangent = tan(radians(abs(steer_deg[inner_side])))
    outer_tangent = tan(radians(abs(steer_deg[outer_side])))
    if abs(inner_tangent) < EPS_GEOMETRIC or abs(outer_tangent) < EPS_GEOMETRIC:
        return None

    ideal_cotangent_difference = track_mm / wheelbase_mm
    if abs(ideal_cotangent_difference) < EPS_GEOMETRIC:
        return None
    actual_cotangent_difference = 1.0 / outer_tangent - 1.0 / inner_tangent
    return 100.0 * actual_cotangent_difference / ideal_cotangent_difference


def _roll_center(
    state: SuspensionState,
    axle: AxleSuspension,
) -> tuple[float | None, float | None]:
    """Return the left/right contact-to-FVIC intersection in chassis YZ.

    This is the classic chassis front-view construction. Its returned Y and Z
    coordinates use the ISO 8855 vehicle-axis orientation but are not resolved
    into road space or world space.
    """
    lines: list[tuple[float, float, float, float]] = []
    for side in (Side.LEFT, Side.RIGHT):
        corner_state = axle.corner_state(state, side)
        fvic = axle.corners[side].compute_front_view_instant_center(corner_state)
        if fvic is None:
            return None, None
        contact_center = corner_state.get(PointID.WHEEL_CONTACT_CENTER)
        contact_y = float(contact_center[Axis.Y])
        contact_z = float(contact_center[Axis.Z])
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
