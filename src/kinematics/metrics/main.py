"""
Metrics public API.

Provides the top-level entry points for computing post-solve kinematic
metrics. Returns ordered mappings ready for direct export integration.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence

from kinematics.metrics.catalog import get_default_corner_metrics
from kinematics.metrics.context import MetricContext
from kinematics.schema.config import SuspensionConfig
from kinematics.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.sensitivity import TangentField
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
        from kinematics.metrics.rates import compute_corner_rate_metrics

        row.update(compute_corner_rate_metrics(state, suspension, tangents))
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
