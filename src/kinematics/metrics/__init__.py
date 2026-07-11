"""
Post-solve suspension metrics.

This package computes kinematic metrics from solved suspension states.
Metrics are computed after solving, never inside the solve loop.
"""

from kinematics.metrics.context import MetricContext
from kinematics.metrics.main import (
    AxleMetricRows,
    MetricRow,
    compute_metrics_for_state,
    compute_metrics_for_state_from_suspension,
    compute_metrics_for_sweep,
)

__all__ = [
    "AxleMetricRows",
    "MetricContext",
    "MetricRow",
    "compute_metrics_for_state",
    "compute_metrics_for_state_from_suspension",
    "compute_metrics_for_sweep",
]
