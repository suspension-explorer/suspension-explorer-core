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
from kinematics.metrics.catalog import (
    get_default_corner_derivative_metrics,
    get_default_corner_metrics,
)
from kinematics.metrics.context import MetricContext
from kinematics.schema.config import SuspensionConfig
from kinematics.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.sensitivity import TangentField
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
    from kinematics.suspensions.base import Suspension


MetricRow = OrderedDict[str, float | None]


def compute_metrics_for_axle_state(
    state: SuspensionState,
    axle: DoubleWishboneAxleSuspension,
    config: SuspensionConfig,
) -> MetricRow:
    """Compute suffixed per-corner rows followed by axle-level metrics."""
    row: MetricRow = OrderedDict()
    for side in (Side.LEFT, Side.RIGHT):
        corner_state = axle.corner_state(state, side)
        side_row = compute_metrics_for_state(
            corner_state,
            axle.corners[side],
            config,
        )
        suffix = f"_{side.name.lower()}"
        for column, value in side_row.items():
            row[f"{column}{suffix}"] = value

    roll_center_y, roll_center_z = _axle_roll_center(state, axle)
    row["roll_center_y_mm"] = roll_center_y
    row["roll_center_z_mm"] = roll_center_z

    design_trackrod_y = float(
        axle.corners[Side.LEFT].initial_state().get(PointID.TRACKROD_INBOARD)[Axis.Y]
    )
    current_trackrod_y = float(
        state.get(PointRef(Side.LEFT, PointID.TRACKROD_INBOARD))[Axis.Y]
    )
    row["trackrod_inboard_displacement_mm"] = current_trackrod_y - design_trackrod_y

    from kinematics.metrics.axle_metrics import append_axle_state_metrics

    append_axle_state_metrics(row, state, axle)
    return row


def _axle_roll_center(
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
        tangents: Optional solution-manifold tangents. Derivative columns are
            appended only when these are supplied.

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
    if tangents:
        from kinematics.metrics.derivatives import evaluate_derivative_metrics

        definitions = (
            *get_default_corner_derivative_metrics(suspension),
            *suspension.derivative_metric_definitions(),
        )
        row.update(evaluate_derivative_metrics(definitions, state, tangents))
    return row


def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[MetricRow]:
    """
    Compute all corner-level metrics for a sweep of solved states.

    Args:
        states: List of solved SuspensionStates from a parametric sweep.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.
        tangents_per_state: Optional tangents aligned one-to-one with
            ``states``. Callers that already have only solved states retain
            the historical non-derivative result; high-level sweep consumers
            should use :func:`kinematics.main.compute_sweep_metrics`.

    Returns:
        A list of ordered metric rows, one per state.
    """
    if tangents_per_state is None:
        return [
            compute_metrics_for_state(state, suspension, config) for state in states
        ]
    if len(states) != len(tangents_per_state):
        raise ValueError("State/tangent row count mismatch")
    return [
        compute_metrics_for_state(state, suspension, config, tangents)
        for state, tangents in zip(states, tangents_per_state)
    ]


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
