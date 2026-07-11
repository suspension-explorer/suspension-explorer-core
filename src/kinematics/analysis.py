"""
High-level, structured suspension sweep analysis for front-end consumers.

The API in this module keeps corner locations structural. Side suffixes are an
export concern and are not embedded in analysis metric keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from kinematics.core.enums import TargetPositionMode
from kinematics.core.point_ref import PointKey, PointRef, Side, point_key_name
from kinematics.core.types import PointTarget, SweepConfig
from kinematics.diagnostics import DiagnosticIssue, diagnose_sweep
from kinematics.main import SweepMetricsResult, compute_sweep_metrics, solve_sweep
from kinematics.metrics.main import AxleMetricRows, MetricRow
from kinematics.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.metrics.registry import MetricSpec, metric_specs_for_suspension
from kinematics.solver import SolverInfo
from kinematics.suspensions.base import Suspension
from kinematics.visualization.display import (
    DisplayLink,
    RockerDisplayGroup,
    WheelAnchorNames,
    WheelDisplayDimensions,
    display_links,
    display_point_keys,
    display_positions,
    rocker_display_groups,
    wheel_anchor_names,
    wheel_display_dimensions,
)

Positions = dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class SuspensionInfo:
    """Identifying metadata for an analyzed suspension."""

    name: str
    type_key: str
    units: str


@dataclass(frozen=True)
class SweepParameter:
    """One principal-axis sweep dimension usable as a chart axis."""

    point: str
    axis: str
    side: str | None


@dataclass(frozen=True)
class AnalyzedFrame:
    """One solved and analyzed sweep step."""

    index: int
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]
    solver: SolverInfo


@dataclass(frozen=True)
class ReferenceCondition:
    """A solved reference pose for comparison with the sweep."""

    label: str
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]


@dataclass(frozen=True)
class StaticPose:
    """The as-assembled initial pose of a suspension geometry."""

    suspension: SuspensionInfo
    point_keys: list[str]
    positions: Positions
    wheel: WheelDisplayDimensions | None
    links: list[DisplayLink]
    wheel_anchors: list[WheelAnchorNames]


@dataclass(frozen=True)
class SweepAnalysis:
    """Complete structured result of a suspension sweep."""

    suspension: SuspensionInfo
    point_keys: list[str]
    metric_keys: list[str]
    corner_metric_keys: list[str]
    locations: list[str]
    metric_display: list[MetricDisplay]
    sweep_parameters: list[SweepParameter]
    references: dict[str, ReferenceCondition]
    wheel: WheelDisplayDimensions | None
    links: list[DisplayLink]
    wheel_anchors: list[WheelAnchorNames]
    diagnostics: list[DiagnosticIssue]
    frames: list[AnalyzedFrame] = field(default_factory=list)

    @property
    def steps(self) -> int:
        """Return the number of solved frames."""
        return len(self.frames)


def _suspension_info(suspension: Suspension) -> SuspensionInfo:
    return SuspensionInfo(
        name=suspension.name,
        type_key=suspension.TYPE_KEY,
        units=suspension.units.symbol,
    )


def sweep_parameters(sweep_config: SweepConfig) -> list[SweepParameter]:
    """Describe every principal-axis dimension in a sweep."""
    parameters: list[SweepParameter] = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        axis = getattr(target.direction, "axis", None)
        if axis is None:
            continue
        key = target.point_id
        side = None
        if isinstance(key, PointRef) and key.side is not Side.CENTER:
            side = key.side.name.lower()
        parameters.append(
            SweepParameter(point=point_key_name(key), axis=axis.name.lower(), side=side)
        )
    return parameters


def _hold_sweep_config(sweep_config: SweepConfig) -> SweepConfig | None:
    hold_dimensions: list[list[PointTarget]] = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        hold_dimensions.append(
            [
                PointTarget(
                    point_id=target.point_id,
                    direction=target.direction,
                    value=0.0,
                    mode=TargetPositionMode.RELATIVE,
                )
            ]
        )
    return SweepConfig(hold_dimensions) if hold_dimensions else None


def _split_metric_rows(
    rows: MetricRow | AxleMetricRows,
) -> tuple[MetricRow, dict[str, MetricRow]]:
    if isinstance(rows, AxleMetricRows):
        return rows.axle, rows.corners
    return rows, {}


def _setup_reference(
    suspension: Suspension,
    sweep_config: SweepConfig,
    point_keys: tuple[PointKey, ...],
    rocker_groups: list[RockerDisplayGroup],
) -> ReferenceCondition | None:
    """Solve the nominal setup pose without making it a hard dependency."""
    hold_config = _hold_sweep_config(sweep_config)
    if hold_config is None:
        return None
    try:
        states, _ = solve_sweep(suspension, hold_config)
        if not states:
            return None
        row = compute_sweep_metrics(suspension, hold_config, states).rows[0]
    except Exception:  # noqa: BLE001 - the reference is optional
        return None
    metrics, corner_metrics = _split_metric_rows(row)
    return ReferenceCondition(
        label="Setup",
        positions=display_positions(states[0].positions, point_keys, rocker_groups),
        metrics=metrics,
        corner_metrics=corner_metrics,
    )


def _derivative_issues(result: SweepMetricsResult) -> list[DiagnosticIssue]:
    """Turn tangent-computation health into visible advisory diagnostics."""
    issues: list[DiagnosticIssue] = []
    if result.derivative_error is not None:
        issues.append(
            DiagnosticIssue(
                step=None,
                category="derivatives",
                severity="warning",
                message=(
                    "Derivative metrics unavailable: tangent computation failed "
                    f"({result.derivative_error}); derivative columns are omitted."
                ),
                value=None,
            )
        )
    infos = result.tangent_solve_infos or []
    deficient = [step for step, info in enumerate(infos) if info.rank_deficient]
    if deficient:
        first = deficient[0]
        min_sv = min(infos[step].smallest_singular_value for step in deficient)
        issues.append(
            DiagnosticIssue(
                step=first,
                category="derivatives",
                severity="warning",
                message=(
                    f"Tangent system rank-deficient at {len(deficient)} of "
                    f"{len(infos)} steps (first at step {first}, rank "
                    f"{infos[first].rank}/{infos[first].n_variables}, smallest "
                    f"singular value {min_sv:.3g}); derivative values may not "
                    "be unique."
                ),
                value=min_sv,
            )
        )
    return issues


def _metric_specs(suspension: Suspension) -> Mapping[str, MetricSpec]:
    """Collect canonical static and topology-specific derivative metadata."""
    return metric_specs_for_suspension(suspension)


def analyze_sweep(suspension: Suspension, sweep_config: SweepConfig) -> SweepAnalysis:
    """Solve a sweep and assemble a complete front-end-ready analysis."""
    states, solver_stats = solve_sweep(suspension, sweep_config)
    metrics_result = compute_sweep_metrics(suspension, sweep_config, states)
    point_keys = display_point_keys(suspension)
    rocker_groups = rocker_display_groups(suspension)

    frames: list[AnalyzedFrame] = []
    for index, (state, info, row) in enumerate(
        zip(states, solver_stats, metrics_result.rows)
    ):
        metrics, corner_metrics = _split_metric_rows(row)
        frames.append(
            AnalyzedFrame(
                index=index,
                positions=display_positions(state.positions, point_keys, rocker_groups),
                metrics=metrics,
                corner_metrics=corner_metrics,
                solver=info,
            )
        )

    metric_keys: list[str] = []
    corner_metric_keys: list[str] = []
    locations: list[str] = []
    for frame in frames:
        if not frame.metrics and not frame.corner_metrics:
            continue
        metric_keys = list(frame.metrics)
        locations = list(frame.corner_metrics)
        if frame.corner_metrics:
            corner_metric_keys = list(next(iter(frame.corner_metrics.values())))
        break

    display_keys = corner_metric_keys + [
        key for key in metric_keys if key not in corner_metric_keys
    ]
    references: dict[str, ReferenceCondition] = {}
    setup = _setup_reference(suspension, sweep_config, point_keys, rocker_groups)
    if setup is not None:
        references["setup"] = setup

    try:
        diagnostics = list(diagnose_sweep(suspension, states, solver_stats).issues)
    except Exception:  # noqa: BLE001 - diagnostics are advisory
        diagnostics = []
    diagnostics.extend(_derivative_issues(metrics_result))

    return SweepAnalysis(
        suspension=_suspension_info(suspension),
        point_keys=[point_key_name(key) for key in point_keys],
        metric_keys=metric_keys,
        corner_metric_keys=corner_metric_keys,
        locations=locations,
        metric_display=metric_display_for_keys(display_keys, _metric_specs(suspension)),
        sweep_parameters=sweep_parameters(sweep_config),
        references=references,
        wheel=wheel_display_dimensions(suspension.config),
        links=display_links(suspension),
        wheel_anchors=wheel_anchor_names(suspension),
        diagnostics=diagnostics,
        frames=frames,
    )


def initial_pose(suspension: Suspension) -> StaticPose:
    """Return the as-assembled pose without running a sweep."""
    state = suspension.initial_state()
    point_keys = display_point_keys(suspension)
    rocker_groups = rocker_display_groups(suspension)
    return StaticPose(
        suspension=_suspension_info(suspension),
        point_keys=[point_key_name(key) for key in point_keys],
        positions=display_positions(state.positions, point_keys, rocker_groups),
        wheel=wheel_display_dimensions(suspension.config),
        links=display_links(suspension),
        wheel_anchors=wheel_anchor_names(suspension),
    )
