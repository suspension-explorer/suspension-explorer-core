"""
Post-solve suspension metrics.

This package computes kinematic metrics from solved suspension states.
Metrics are computed after solving, never inside the solve loop.

Solved geometry is represented in chassis coordinates. Individual metrics
declare whether they use the chassis axes directly or resolve values into
ISO 8855 tyre axes and the local road plane. Metric calculation does not use
the optional world-space presentation transform.
"""

from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.main import (
    AxleMetricRows,
    MetricRow,
    compute_metrics_for_state,
    compute_metrics_for_state_from_suspension,
    compute_metrics_for_sweep,
)
from kinematics.core.metrics.steering_axis_geometry import SteeringAxis
