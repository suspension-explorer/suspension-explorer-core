"""
Metrics public API.

Provides the top-level entry points for computing post-solve kinematic
metrics. Returns ordered mappings ready for direct export integration.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, overload

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


@dataclass(frozen=True)
class AxleMetricRows:
    """Location-independent axle metrics plus one row per corner."""

    axle: MetricRow
    corners: dict[str, MetricRow]

    def flat_row(self) -> MetricRow:
        """Render the structured rows for a flat export boundary."""
        return flatten_metric_rows(self.axle, self.corners)


def flatten_metric_rows(
    metrics: MetricRow,
    corner_metrics: Mapping[str, MetricRow],
) -> MetricRow:
    """Flatten structural metric locations using side suffixes."""
    from kinematics.metrics.registry import flat_key

    flat: MetricRow = OrderedDict()
    for location, row in corner_metrics.items():
        for key, value in row.items():
            flat[flat_key(key, location)] = value
    flat.update(metrics)
    return flat


def compute_metrics_for_axle_state(
    state: SuspensionState,
    axle: DoubleWishboneAxleSuspension,
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
) -> AxleMetricRows:
    """Compute structural per-corner rows followed by axle-level metrics."""
    axle_row: MetricRow = OrderedDict()
    corner_rows: dict[str, MetricRow] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner = axle.corners[side]
        corner_state = axle.corner_state(state, side)
        corner_config = corner.config if corner.config is not None else config
        side_row = compute_metrics_for_state(
            corner_state,
            corner,
            corner_config,
            _corner_tangents(tangents, side) if tangents else None,
        )
        corner_rows[side.name.lower()] = side_row

    from kinematics.metrics.axle_metrics import append_axle_state_metrics

    append_axle_state_metrics(axle_row, state, axle)
    for key, value in axle.topology_metric_values(state).items():
        base_key, location = _split_topology_location(key)
        if location is None:
            axle_row[base_key] = value
        else:
            corner_rows[location][base_key] = value
    if tangents:
        from kinematics.metrics.derivatives import evaluate_derivative_metrics

        axle_row.update(
            evaluate_derivative_metrics(
                axle.derivative_metric_definitions(),
                state,
                tangents,
            )
        )
    return AxleMetricRows(axle=axle_row, corners=corner_rows)


def _split_topology_location(key: str) -> tuple[str, str | None]:
    """Split topology-owned flat side keys while their hook is migrated."""
    from kinematics.metrics.registry import split_flat_key

    return split_flat_key(key)


def _corner_tangents(
    tangents: "Sequence[TangentField]",
    side: Side,
) -> list["TangentField"]:
    """Strip one side's PointRef qualifiers from axle tangent fields."""
    from kinematics.sensitivity import TangentField

    result: list[TangentField] = []
    for tangent in tangents:
        target_key = tangent.target.point_id
        if not isinstance(target_key, PointRef) or target_key.side is not side:
            continue
        result.append(
            TangentField(
                target_index=tangent.target_index,
                target=tangent.target._replace(point_id=target_key.point),
                velocities={
                    key.point: velocity
                    for key, velocity in tangent.velocities.items()
                    if isinstance(key, PointRef) and key.side is side
                },
            )
        )
    return result


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
    row.update(suspension.topology_metric_values(state))
    if tangents:
        from kinematics.metrics.derivatives import evaluate_derivative_metrics

        definitions = (
            *get_default_corner_derivative_metrics(suspension),
            *suspension.derivative_metric_definitions(),
        )
        row.update(evaluate_derivative_metrics(definitions, state, tangents))
    return row


@overload
def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "DoubleWishboneAxleSuspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[AxleMetricRows]: ...


@overload
def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[MetricRow]: ...


def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[MetricRow | AxleMetricRows]:
    """
    Compute metrics for a sweep of solved corner or axle states.

    Args:
        states: List of solved SuspensionStates from a parametric sweep.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.
        tangents_per_state: Optional tangents aligned one-to-one with
            ``states``. Callers that already have only solved states retain
            the historical non-derivative result; high-level sweep consumers
            should use :func:`kinematics.main.compute_sweep_metrics`.

    Returns:
        One metric result per state. Corner suspensions return ordered rows;
        axle suspensions return structural axle and per-corner rows.
    """
    if tangents_per_state is None:
        return [
            _compute_metrics_for_suspension_state(state, suspension, config)
            for state in states
        ]
    if len(states) != len(tangents_per_state):
        raise ValueError("State/tangent row count mismatch")
    return [
        _compute_metrics_for_suspension_state(
            state,
            suspension,
            config,
            tangents,
        )
        for state, tangents in zip(states, tangents_per_state)
    ]


def _compute_metrics_for_suspension_state(
    state: SuspensionState,
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
) -> MetricRow | AxleMetricRows:
    """Dispatch metric calculation without applying corner metrics to an axle."""
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

    if isinstance(suspension, DoubleWishboneAxleSuspension):
        return compute_metrics_for_axle_state(state, suspension, config, tangents)
    return compute_metrics_for_state(state, suspension, config, tangents)


@overload
def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "DoubleWishboneAxleSuspension",
) -> AxleMetricRows: ...


@overload
def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> MetricRow: ...


def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> MetricRow | AxleMetricRows:
    """
    Compute metrics using parameters from the suspension configuration.

    Convenience wrapper that extracts config from the suspension instance.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension containing configuration.

    Returns:
        An ordered row for a corner suspension, or structural axle and
        per-corner rows for an axle suspension.

    Raises:
        ValueError: If the suspension has no configuration.
    """
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    return _compute_metrics_for_suspension_state(
        state,
        suspension,
        suspension.config,
    )
