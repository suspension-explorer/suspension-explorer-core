"""
open-kinematics: a geometric constraint solver for suspension kinematics.

This top-level module is the package's public API. Consumers (the CLI, API
servers, notebooks) import from here; deeper module paths are internal and
free to change.

Typical flow::

    from kinematics import (
        analyze_sweep,
        build_suspension,
        build_sweep_config,
        parse_geometry_spec,
        SweepSpec,
    )

    suspension = build_suspension(parse_geometry_spec(geometry_data))
    sweep = build_sweep_config(SweepSpec.model_validate(sweep_data), suspension)
    analysis = analyze_sweep(suspension, sweep)

YAML file loading lives in ``kinematics.io``; plotting in
``kinematics.visualization`` (requires the ``viz`` extra). Importing
``kinematics`` never pulls in matplotlib.
"""

from importlib.metadata import PackageNotFoundError, version

from kinematics.analysis import (
    AnalyzedFrame,
    StaticPose,
    SweepAnalysis,
    analyze_sweep,
    initial_pose,
)
from kinematics.main import compute_sweep_metrics, compute_sweep_tangents, solve_sweep
from kinematics.metrics.main import AxleMetricRows, flatten_metric_rows
from kinematics.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.metrics.registry import (
    MetricSpec,
    all_metric_specs,
    flat_key,
    motion_ratio_specs,
)
from kinematics.schema import (
    GeometrySpec,
    SweepSpec,
    build_sweep_config,
    parse_geometry_spec,
)
from kinematics.solver import SolverInfo
from kinematics.suspensions.build import build_suspension
from kinematics.suspensions.registry import get_suspension_class, list_supported_types

try:
    __version__ = version("kinematics")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = [
    "AnalyzedFrame",
    "AxleMetricRows",
    "GeometrySpec",
    "MetricDisplay",
    "MetricSpec",
    "SolverInfo",
    "StaticPose",
    "SweepAnalysis",
    "SweepSpec",
    "__version__",
    "all_metric_specs",
    "analyze_sweep",
    "build_suspension",
    "build_sweep_config",
    "compute_sweep_metrics",
    "compute_sweep_tangents",
    "flat_key",
    "flatten_metric_rows",
    "get_suspension_class",
    "initial_pose",
    "list_supported_types",
    "metric_display_for_keys",
    "motion_ratio_specs",
    "parse_geometry_spec",
    "solve_sweep",
]
