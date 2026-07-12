"""
Public, transport-independent suspension solver API.

This package owns validated schemas, suspension construction, numerical solving,
diagnostics, metrics, and structured analysis. It does not read or write files,
render plots, format terminal output, or control process exit behavior.
"""

from kinematics.core.analysis import (
    AnalyzedFrame,
    ReferenceCondition,
    StaticPose,
    SuspensionInfo,
    SweepAnalysis,
    SweepParameter,
    analyze_solved_sweep,
    analyze_sweep,
    initial_pose,
)
from kinematics.core.diagnostics import DiagnosticIssue, SweepDiagnostics
from kinematics.core.metrics.metadata import MetricDisplay
from kinematics.core.metrics.registry import MetricSpec
from kinematics.core.schema import (
    GeometrySpec,
    SweepSpec,
    build_sweep_config,
    parse_geometry_spec,
)
from kinematics.core.solver import SolverInfo
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.build import build_suspension
from kinematics.core.suspensions.registry import list_supported_types
from kinematics.core.sweep import solve_sweep
