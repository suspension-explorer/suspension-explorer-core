"""Public API for suspension loading, solving, and structured analysis."""

from kinematics.analysis import (
    AnalyzedFrame,
    ReferenceCondition,
    StaticPose,
    SuspensionInfo,
    SweepAnalysis,
    SweepParameter,
    analyze_sweep,
    initial_pose,
)
from kinematics.io import load_geometry, load_sweep
from kinematics.main import solve_sweep
from kinematics.metrics.metadata import MetricDisplay
from kinematics.metrics.registry import MetricSpec
