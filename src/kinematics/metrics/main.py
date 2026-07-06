"""
Metrics public API.

Provides the top-level entry points for computing post-solve kinematic
metrics. Returns ordered mappings ready for direct export integration.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.metrics.catalog import get_default_corner_metrics
from kinematics.metrics.context import MetricContext
from kinematics.state import SuspensionState
from kinematics.suspensions.config.settings import SuspensionConfig

if TYPE_CHECKING:
    from kinematics.core.geometry import Point3
    from kinematics.sensitivity import TangentField
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
    from kinematics.suspensions.base import Suspension


MetricRow = OrderedDict[str, float | None]


def compute_metrics_for_state(
    state: SuspensionState,
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
) -> MetricRow:
    """
    Compute all corner-level metrics for a single solved state.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.
        tangents: Optional solution-manifold tangents for this state (from
            kinematics.sensitivity). When given, derivative metrics (camber
            gain, bump steer, motion ratios, ...) are appended for every
            recognized sweep driver.

    Returns:
        An ordered mapping of metric column names to values. Values are
        None when the underlying geometry is undefined (e.g. parallel
        links producing an IC at infinity).
    """
    ctx = MetricContext(
        state=state,
        suspension=suspension,
        config=config,
    )

    catalog = get_default_corner_metrics()
    row: MetricRow = OrderedDict()
    for metric in catalog:
        row[metric.column_name] = metric.compute(ctx)

    if suspension.has_rocker:
        _append_rocker_metrics(row, ctx)

    if tangents:
        from kinematics.metrics.rates import compute_corner_rate_metrics

        row.update(compute_corner_rate_metrics(state, suspension, tangents))

    return row


def _append_rocker_metrics(row: MetricRow, ctx: MetricContext) -> None:
    """
    Append rocker rotation and torsion-bar twist columns for a corner.

    The rocker angle is the signed rotation of ``PUSHROD_INBOARD`` about the
    rocker axis (measured about the authored ROCKER_AXIS_FRONT -> ROCKER_AXIS_REAR
    direction), from the design condition to the current state, then multiplied by
    the corner's ``side_sign`` (+1 left, -1 right).

    The ``side_sign`` normalisation is deliberate. Mirroring left <-> right is a
    reflection (``y -> -y``); a reflection negates a signed angle measured about a
    mirrored axis. Without normalisation a symmetric heave motion would report
    equal-and-opposite rocker angles on the two sides. Multiplying by ``side_sign``
    makes symmetric heave report EQUAL rocker angles on both sides, and antisymmetric
    roll report equal-and-opposite ones, which is the physically intuitive reading.

    ``torsion_bar_twist_deg`` is identical to ``rocker_angle_deg``: the torsion bar
    is coaxial with the rocker pivot and grounded at its far end, so its twist is
    exactly the rocker rotation. It is kept as its own column for clarity.
    """
    from math import degrees

    from kinematics.core.vector_utils.geometric import signed_angle_about_axis

    design = ctx.suspension.initial_state()
    axis_front = design.positions[PointID.ROCKER_AXIS_FRONT]
    axis_rear = design.positions[PointID.ROCKER_AXIS_REAR]
    axis_direction = (axis_rear - axis_front).normalize()

    raw = signed_angle_about_axis(
        design.positions[PointID.PUSHROD_INBOARD],
        ctx.state.positions[PointID.PUSHROD_INBOARD],
        axis_front,
        axis_direction,
    )
    rocker_angle_deg = degrees(raw) * ctx.side_sign
    row["rocker_angle_deg"] = rocker_angle_deg
    row["torsion_bar_twist_deg"] = rocker_angle_deg


def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
) -> list[MetricRow]:
    """
    Compute all corner-level metrics for a sweep of solved states.

    Args:
        states: List of solved SuspensionStates from a parametric sweep.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.

    Returns:
        A list of ordered metric rows, one per state.
    """
    return [compute_metrics_for_state(state, suspension, config) for state in states]


def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> MetricRow:
    """
    Compute metrics using parameters from the suspension configuration.

    Convenience wrapper that extracts config from the suspension instance.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension containing configuration.

    Returns:
        An ordered mapping of metric column names to values.

    Raises:
        ValueError: If the suspension has no configuration.
    """
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    return compute_metrics_for_state(
        state=state,
        suspension=suspension,
        config=suspension.config,
    )


def _intersect_lines_yz(
    p0_y: float,
    p0_z: float,
    d0_y: float,
    d0_z: float,
    p1_y: float,
    p1_z: float,
    d1_y: float,
    d1_z: float,
) -> tuple[float, float] | None:
    """
    Intersect two lines in the front-view (YZ) plane.

    Each line is given by a point ``(p, y/z)`` and a direction ``(d, y/z)``.
    Returns the ``(y, z)`` intersection, or ``None`` if the directions are
    parallel (cross product below ``EPS_GEOMETRIC``).
    """
    denom = d0_y * d1_z - d0_z * d1_y
    if abs(denom) < EPS_GEOMETRIC:
        return None
    # Solve p0 + t*d0 = p1 + s*d1 for t.
    t = ((p1_y - p0_y) * d1_z - (p1_z - p0_z) * d1_y) / denom
    return (p0_y + t * d0_y, p0_z + t * d0_z)


def _axle_roll_center(
    state: SuspensionState,
    axle: "DoubleWishboneAxleSuspension",
) -> tuple[float | None, float | None]:
    """
    Front-view roll center from the two contact-patch -> FVIC lines.

    For each side, the line runs from the contact patch center to the front-view
    instant center (FVIC), both projected into the YZ plane. Their intersection
    is the roll center. Returns ``(None, None)`` if either FVIC is undefined or
    the two lines are parallel.
    """
    side_lines: list[tuple[float, float, float, float]] = []
    for side in (Side.LEFT, Side.RIGHT):
        corner = axle.corners[side]
        corner_state = axle.corner_state(state, side)
        fvic = corner.compute_front_view_instant_center(corner_state)
        if fvic is None:
            return (None, None)
        cp: "Point3" = corner_state.get(PointID.CONTACT_PATCH_CENTER)
        cp_y = float(cp[Axis.Y])
        cp_z = float(cp[Axis.Z])
        d_y = float(fvic[Axis.Y]) - cp_y
        d_z = float(fvic[Axis.Z]) - cp_z
        side_lines.append((cp_y, cp_z, d_y, d_z))

    left, right = side_lines
    result = _intersect_lines_yz(*left, *right)
    if result is None:
        return (None, None)
    return result


def compute_metrics_for_axle_state(
    state: SuspensionState,
    axle: "DoubleWishboneAxleSuspension",
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
) -> MetricRow:
    """
    Compute per-side and axle-level metrics for a solved axle state.

    Per side, the solved axle state is stripped to a plain corner state
    (``axle.corner_state``) and the standard corner metrics are computed against
    that side's corner suspension, with every column prefixed ``left_`` /
    ``right_``. Axle-level metrics are appended afterwards.

    Axle metrics:

    - ``roll_center_y_mm`` / ``roll_center_z_mm``: front-view intersection of the
      two contact-patch -> FVIC lines (``None`` if undefined or parallel).
    - ``total_toe_deg``: sum of the two per-side toe-in angles. Each corner's
      ``roadwheel_angle_deg`` already encodes toe-in as positive via the corner
      ``side_sign`` convention (front of the wheel toward the vehicle
      centreline), so their sum is the total toe-in of the axle. Positive means
      both wheels toe in.
    - ``track_mm``: absolute lateral (Y) distance between the two contact patch
      centers.
    - ``rack_displacement_mm``: signed Y displacement of the LEFT inboard
      trackrod from its design position (positive = +Y = leftward).

    Args:
        state: The solved axle state (``PointRef``-keyed).
        axle: The axle suspension.
        config: Shared vehicle configuration.
        tangents: Optional solution-manifold tangents for this state (from
            kinematics.sensitivity, ``PointRef``-keyed). When given, per-side
            and modal (roll/heave) derivative metrics are appended.

    Returns:
        An ordered metric row.
    """
    row: MetricRow = OrderedDict()

    # Per-side corner metrics, prefixed.
    side_rows: dict[Side, MetricRow] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner_state = axle.corner_state(state, side)
        side_row = compute_metrics_for_state(corner_state, axle.corners[side], config)
        side_rows[side] = side_row
        prefix = f"{side.name.lower()}_"
        for column, value in side_row.items():
            row[prefix + column] = value

    # Axle-level metrics.
    roll_center_y, roll_center_z = _axle_roll_center(state, axle)
    row["roll_center_y_mm"] = roll_center_y
    row["roll_center_z_mm"] = roll_center_z

    left_toe = side_rows[Side.LEFT]["roadwheel_angle_deg"]
    right_toe = side_rows[Side.RIGHT]["roadwheel_angle_deg"]
    if left_toe is None or right_toe is None:
        row["total_toe_deg"] = None
    else:
        row["total_toe_deg"] = left_toe + right_toe

    left_cp_y = float(
        state.positions[PointRef(Side.LEFT, PointID.CONTACT_PATCH_CENTER)][Axis.Y]
    )
    right_cp_y = float(
        state.positions[PointRef(Side.RIGHT, PointID.CONTACT_PATCH_CENTER)][Axis.Y]
    )
    row["track_mm"] = abs(left_cp_y - right_cp_y)

    design_rack_y = float(
        axle.corners[Side.LEFT]
        .initial_state()
        .positions[PointID.TRACKROD_INBOARD][Axis.Y]
    )
    current_rack_y = float(
        state.positions[PointRef(Side.LEFT, PointID.TRACKROD_INBOARD)][Axis.Y]
    )
    row["rack_displacement_mm"] = current_rack_y - design_rack_y

    if axle.has_arb:
        _append_arb_metrics(row, state, axle)

    # Axle-level per-state (non-derivative) modal metrics.
    from kinematics.metrics.axle_metrics import append_axle_state_metrics

    append_axle_state_metrics(row, state, axle, config, side_rows)

    if tangents:
        from kinematics.metrics.rates import compute_axle_rate_metrics

        row.update(compute_axle_rate_metrics(state, axle, tangents))

    return row


def _append_arb_metrics(
    row: MetricRow,
    state: SuspensionState,
    axle: "DoubleWishboneAxleSuspension",
) -> None:
    """
    Append per-side ARB arm angles and the axle-level ARB twist.

    The anti-roll bar shares a single chassis-fixed axis (``ARB_AXIS_A`` ->
    ``ARB_AXIS_B``). Unlike the rocker angle, the arm angles use RAW signed
    angles about that single authored direction with NO side normalisation,
    because both arms rotate about the same physical axis. Each arm angle is the
    signed rotation of that side's ``ARB_DROPLINK`` about the axis, from design to
    current.

    ``arb_twist_deg = left - right`` is the physical relative twist of the
    torsion element between the two arm stations (right-hand rule about A -> B).
    For a conventional mirrored transverse-axis ARB, symmetric heave rotates both
    arms equally (twist ~ 0) while roll rotates them oppositely (twist != 0).
    """
    from math import degrees

    from kinematics.core.vector_utils.geometric import signed_angle_about_axis

    design = axle.initial_state()
    axis_a = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_A)]
    axis_b = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_B)]
    axis_direction = (axis_b - axis_a).normalize()

    angles: dict[Side, float] = {}
    for side in (Side.LEFT, Side.RIGHT):
        key = PointRef(side, PointID.ARB_DROPLINK)
        raw = signed_angle_about_axis(
            design.positions[key],
            state.positions[key],
            axis_a,
            axis_direction,
        )
        angles[side] = degrees(raw)

    row["left_arb_arm_angle_deg"] = angles[Side.LEFT]
    row["right_arb_arm_angle_deg"] = angles[Side.RIGHT]
    row["arb_twist_deg"] = angles[Side.LEFT] - angles[Side.RIGHT]
