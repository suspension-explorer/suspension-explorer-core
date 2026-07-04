"""
Metrics public API.

Provides the top-level entry points for computing post-solve kinematic
metrics. Returns ordered mappings ready for direct export integration.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.metrics.catalog import get_default_corner_metrics
from kinematics.metrics.context import MetricContext
from kinematics.state import SuspensionState
from kinematics.suspensions.config.settings import SuspensionConfig

if TYPE_CHECKING:
    from kinematics.core.geometry import Point3
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
    from kinematics.suspensions.base import Suspension


MetricRow = OrderedDict[str, float | None]


def compute_metrics_for_state(
    state: SuspensionState,
    suspension: "Suspension",
    config: SuspensionConfig,
) -> MetricRow:
    """
    Compute all corner-level metrics for a single solved state.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.

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
    return row


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
    Front-view roll centre from the two contact-patch -> FVIC lines.

    For each side, the line runs from the contact patch centre to the front-view
    instant centre (FVIC), both projected into the YZ plane. Their intersection
    is the roll centre. Returns ``(None, None)`` if either FVIC is undefined or
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
      centres.
    - ``rack_displacement_mm``: signed Y displacement of the LEFT inboard
      trackrod from its design position (positive = +Y = leftward).

    Args:
        state: The solved axle state (``PointRef``-keyed).
        axle: The axle suspension.
        config: Shared vehicle configuration.

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

    return row
