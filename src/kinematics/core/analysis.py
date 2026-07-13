"""
High-level, structured suspension sweep analysis for front-end consumers.

The API in this module keeps corner locations structural. Side suffixes are an
export concern and are not embedded in analysis metric keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
)
from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.core.metrics.registry import metric_specs_for_suspension
from kinematics.core.presentation import (
    NamedElementPath,
    WheelDimensions,
    WheelReferences,
    named_element_paths,
    named_point_keys,
    resolve_positions,
    wheel_dimensions,
    wheel_references,
)
from kinematics.core.primitives.enums import TargetPositionMode
from kinematics.core.primitives.point_ref import (
    PointRef,
    Side,
    point_key_name,
)
from kinematics.core.solver import SolverInfo
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.sweep import (
    EvaluatedSweep,
    compute_sweep_metrics,
    evaluate_solved_sweep,
    solve_evaluated_sweep,
    solve_sweep,
)
from kinematics.core.targeting import PointTarget, SweepConfig

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
    wheel: WheelDimensions | None
    elements: list[NamedElementPath]
    wheel_references: list[WheelReferences]


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
    wheel: WheelDimensions | None
    elements: list[NamedElementPath]
    wheel_references: list[WheelReferences]
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
    assembly: SuspensionAssembly,
) -> tuple[ReferenceCondition | None, DiagnosticIssue | None]:
    """Solve the nominal setup pose without making it a hard dependency."""
    hold_config = _hold_sweep_config(sweep_config)
    if hold_config is None:
        return None, None
    try:
        states, _solver_stats = solve_sweep(suspension, hold_config)
        if not states:
            return None, None
        row = compute_sweep_metrics(suspension, hold_config, states).rows[0]
    except Exception as error:  # noqa: BLE001 - the reference is optional
        return None, DiagnosticIssue(
            step=None,
            category=DiagnosticCategory.REFERENCE,
            severity=DiagnosticSeverity.WARNING,
            message=(
                "Setup reference unavailable: reference solve failed "
                f"({type(error).__name__}: {error})."
            ),
            value=None,
        )
    metrics, corner_metrics = _split_metric_rows(row)
    return (
        ReferenceCondition(
            label="Setup",
            positions=resolve_positions(states[0].positions, assembly),
            metrics=metrics,
            corner_metrics=corner_metrics,
        ),
        None,
    )


def analyze_sweep(suspension: Suspension, sweep_config: SweepConfig) -> SweepAnalysis:
    """Solve a sweep and assemble a complete structured analysis."""
    return analyze_evaluated_sweep(
        suspension,
        sweep_config,
        solve_evaluated_sweep(suspension, sweep_config),
    )


def analyze_solved_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: list[SuspensionState],
    solver_stats: list[SolverInfo],
) -> SweepAnalysis:
    """Assemble structured analysis from an already solved suspension sweep."""
    return analyze_evaluated_sweep(
        suspension,
        sweep_config,
        evaluate_solved_sweep(
            suspension,
            sweep_config,
            states,
            solver_stats,
        ),
    )


def analyze_evaluated_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
    evaluated: EvaluatedSweep,
) -> SweepAnalysis:
    """Build the rich presentation model for an evaluated sweep."""
    assembly = suspension.assembly()

    frames: list[AnalyzedFrame] = []
    for index, (state, info, row) in enumerate(
        zip(evaluated.states, evaluated.solver_stats, evaluated.metrics.rows)
    ):
        metrics, corner_metrics = _split_metric_rows(row)
        frames.append(
            AnalyzedFrame(
                index=index,
                positions=resolve_positions(state.positions, assembly),
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
    setup, reference_issue = _setup_reference(
        suspension,
        sweep_config,
        assembly,
    )
    if setup is not None:
        references["setup"] = setup
    diagnostics = list(evaluated.diagnostics)
    if reference_issue is not None:
        diagnostics.append(reference_issue)

    return SweepAnalysis(
        suspension=_suspension_info(suspension),
        point_keys=named_point_keys(assembly),
        metric_keys=metric_keys,
        corner_metric_keys=corner_metric_keys,
        locations=locations,
        metric_display=metric_display_for_keys(
            display_keys,
            metric_specs_for_suspension(suspension),
        ),
        sweep_parameters=sweep_parameters(sweep_config),
        references=references,
        wheel=wheel_dimensions(suspension.config),
        elements=named_element_paths(assembly),
        wheel_references=wheel_references(assembly),
        diagnostics=diagnostics,
        frames=frames,
    )


def initial_pose(suspension: Suspension) -> StaticPose:
    """Return the as-assembled pose without running a sweep."""
    state = suspension.initial_state()
    assembly = suspension.assembly()
    return StaticPose(
        suspension=_suspension_info(suspension),
        point_keys=named_point_keys(assembly),
        positions=resolve_positions(state.positions, assembly),
        wheel=wheel_dimensions(suspension.config),
        elements=named_element_paths(assembly),
        wheel_references=wheel_references(assembly),
    )
