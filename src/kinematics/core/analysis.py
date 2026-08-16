"""
High-level, structured suspension sweep analysis for front-end consumers.

The API in this module keeps corner locations structural. Side suffixes are an
export concern and are not embedded in analysis metric keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.coordinates import ScalarCoordinate
from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
)
from kinematics.core.enums import TargetValueMode
from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.core.metrics.registry import metric_specs_for_suspension
from kinematics.core.pose import WorldSpace, world_space_for_axle_state
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
from kinematics.core.primitives.point_ref import point_key_name
from kinematics.core.screw_axis import ScrewAxisStatus, UprightScrewAxisResult
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
from kinematics.core.targeting import SweepConfig

if TYPE_CHECKING:
    from kinematics.core.steering_response import (
        SuspensionHoldCatalogue,
    )

if TYPE_CHECKING:
    from kinematics.core.suspensions.axle import AxleSuspension

Positions = dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class SuspensionInfo:
    """Identifying metadata for an analyzed suspension."""

    name: str
    type_key: str
    units: str


@dataclass(frozen=True)
class SweepParameter:
    """One measured scalar sweep dimension usable as a chart axis."""

    type: str
    coordinate_id: str
    label: str
    unit: str
    values: list[float]
    point: str | None = None
    axis: str | None = None
    actuator: str | None = None
    element: str | None = None
    side: str | None = None


@dataclass(frozen=True)
class CoordinateInfo:
    """Transport-safe metadata for one suspension coordinate."""

    id: str
    type: str
    label: str
    unit: str
    scope: str
    side: str | None
    point_keys: tuple[str, ...]


@dataclass(frozen=True)
class SteeringResponseDefinitionInfo:
    """Transport-safe provenance for the steering-response derivative."""

    owner: str
    definition_id: str
    provenance: str
    steering_coordinate_id: str
    held_coordinates: tuple[CoordinateInfo, ...]
    requested_option_id: str | None
    resolved_option_id: str
    selection_source: str
    label: str
    description: str
    warning: str | None


@dataclass(frozen=True)
class SuspensionHoldOptionInfo:
    """Transport-safe metadata for one topology-owned suspension hold."""

    id: str
    label: str
    description: str
    availability: str
    warning: str | None
    unavailable_reason: str | None
    held_coordinates: tuple[CoordinateInfo, ...]


@dataclass(frozen=True)
class SuspensionHoldCatalogueInfo:
    """Renderer-neutral suspension-hold catalogue."""

    default_option_id: str
    options: tuple[SuspensionHoldOptionInfo, ...]


@dataclass(frozen=True)
class AnalyzedSteeringResponseAxis:
    """Renderer-independent steering-axis result for one upright and frame."""

    upright_label: str
    point_keys: tuple[str, ...]
    status: ScrewAxisStatus
    point: tuple[float, float, float] | None
    direction: tuple[float, float, float] | None
    pitch: float | None
    angular_rate: float | None
    fit_rms: float | None
    fit_max: float | None
    fit_rank: int
    point_count: int
    message: str | None


@dataclass(frozen=True)
class AnalyzedFrame:
    """One solved and analyzed sweep step."""

    index: int
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]
    world_space: WorldSpace | None
    solver: SolverInfo
    steering_response_axes: tuple[AnalyzedSteeringResponseAxis, ...] = ()


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
    drive_coordinates: list[CoordinateInfo]
    suspension_hold_catalogue: SuspensionHoldCatalogueInfo | None


@dataclass(frozen=True)
class SweepAnalysis:
    """Complete structured result of a suspension sweep."""

    suspension: SuspensionInfo
    steering_response: SteeringResponseDefinitionInfo | None
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
        type_key=suspension.reported_type_key(),
        units=suspension.units.symbol,
    )


def _coordinate_info(
    coordinate: ScalarCoordinate,
) -> CoordinateInfo:
    """Convert one suspension coordinate to stable analysis metadata."""
    return CoordinateInfo(
        id=coordinate.id,
        type=coordinate.kind.value,
        label=coordinate.label,
        unit=coordinate.unit,
        scope=coordinate.scope.value,
        side=(coordinate.side.name.lower() if coordinate.side is not None else None),
        point_keys=tuple(point_key_name(point) for point in coordinate.point_keys),
    )


def _steering_response_info(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> SteeringResponseDefinitionInfo | None:
    """Describe the topology-owned suspension hold, when available."""
    definition = suspension.resolve_suspension_hold(sweep_config.suspension_hold_id)
    if definition is None:
        return None
    return SteeringResponseDefinitionInfo(
        owner=definition.owner,
        definition_id=definition.definition_id,
        provenance=definition.provenance,
        steering_coordinate_id=definition.steering_coordinate_id,
        held_coordinates=tuple(
            _coordinate_info(coordinate) for coordinate in definition.hold.coordinates
        ),
        requested_option_id=definition.requested_option_id,
        resolved_option_id=definition.definition_id,
        selection_source=definition.selection_source.value,
        label=definition.label,
        description=definition.description,
        warning=definition.warning,
    )


def _suspension_hold_catalogue_info(
    catalogue: "SuspensionHoldCatalogue | None",
) -> SuspensionHoldCatalogueInfo | None:
    """Convert topology capability metadata without adding UI policy."""
    if catalogue is None:
        return None
    return SuspensionHoldCatalogueInfo(
        default_option_id=catalogue.default_option_id,
        options=tuple(
            SuspensionHoldOptionInfo(
                id=option.id,
                label=option.label,
                description=option.description,
                availability=option.availability.value,
                warning=option.warning,
                unavailable_reason=option.unavailable_reason,
                held_coordinates=tuple(
                    _coordinate_info(coordinate)
                    for coordinate in option.hold.coordinates
                ),
            )
            for option in catalogue.options
        ),
    )


def sweep_parameters(
    sweep_config: SweepConfig,
    states: Sequence[SuspensionState] | None = None,
) -> list[SweepParameter]:
    """Describe every scalar dimension and its authored or measured values."""
    parameters: list[SweepParameter] = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        values = (
            [float(target.measure(state.positions)) for state in states]
            if states is not None
            else [float(item.value) for item in dimension]
        )
        structural_side = target.structural_side
        parameters.append(
            SweepParameter(
                type=target.kind.value,
                coordinate_id=target.coordinate_id,
                label=target.label,
                unit=target.unit,
                values=values,
                point=target.parameter_point,
                axis=target.parameter_axis,
                actuator=target.parameter_actuator,
                element=target.parameter_element,
                side=(
                    structural_side.name.lower()
                    if structural_side is not None
                    else None
                ),
            )
        )
    return parameters


def _hold_sweep_config(sweep_config: SweepConfig) -> SweepConfig | None:
    hold_dimensions = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        hold_dimensions.append([target.with_value(0.0, TargetValueMode.RELATIVE)])
    if not hold_dimensions:
        return None
    return SweepConfig(
        hold_dimensions,
        hold=sweep_config.hold,
        suspension_hold_id=sweep_config.suspension_hold_id,
    )


def _split_metric_rows(
    rows: MetricRow | AxleMetricRows,
) -> tuple[MetricRow, dict[str, MetricRow]]:
    """Preserve metric values and their already-declared reference systems."""
    if isinstance(rows, AxleMetricRows):
        return rows.axle, {side.name.lower(): row for side, row in rows.corners.items()}
    return rows, {}


def _finite_or_none(value: float) -> float | None:
    """Return a JSON-safe finite float, or ``None`` for unavailable diagnostics."""
    return float(value) if np.isfinite(value) else None


def _vector_tuple(values: np.ndarray) -> tuple[float, float, float]:
    """Convert a three-component internal array to a fixed primitive tuple."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _analyzed_steering_response_axis(
    result: UprightScrewAxisResult,
) -> AnalyzedSteeringResponseAxis:
    """Convert one core result to primitive structured analysis values."""
    axis = result.axis
    twist = result.twist
    fit_rms = axis.fit_rms if axis is not None else (twist.fit_rms if twist else None)
    fit_max = axis.fit_max if axis is not None else (twist.fit_max if twist else None)
    return AnalyzedSteeringResponseAxis(
        upright_label=result.upright_label,
        point_keys=tuple(point_key_name(key) for key in result.point_keys),
        status=result.status,
        point=_vector_tuple(axis.point.data) if axis else None,
        direction=_vector_tuple(axis.direction.data) if axis else None,
        pitch=float(axis.pitch) if axis else None,
        angular_rate=float(axis.angular_rate) if axis else None,
        fit_rms=_finite_or_none(fit_rms) if fit_rms is not None else None,
        fit_max=_finite_or_none(fit_max) if fit_max is not None else None,
        fit_rank=twist.fit_rank if twist is not None else 0,
        point_count=result.point_count,
        message=result.message,
    )


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
    for index, (state, info, row, steering_axes) in enumerate(
        zip(
            evaluated.states,
            evaluated.solver_stats,
            evaluated.metrics.rows,
            evaluated.steering_response_axes,
            strict=True,
        )
    ):
        metrics, corner_metrics = _split_metric_rows(row)
        frames.append(
            AnalyzedFrame(
                index=index,
                positions=resolve_positions(state.positions, assembly),
                metrics=metrics,
                corner_metrics=corner_metrics,
                world_space=(
                    world_space_for_axle_state(
                        cast("AxleSuspension", suspension),
                        state,
                    )
                    if suspension.is_axle
                    else None
                ),
                solver=info,
                steering_response_axes=tuple(
                    _analyzed_steering_response_axis(result) for result in steering_axes
                ),
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
        for row in frame.corner_metrics.values():
            for key in row:
                if key not in corner_metric_keys:
                    corner_metric_keys.append(key)
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
        steering_response=_steering_response_info(suspension, sweep_config),
        point_keys=named_point_keys(assembly),
        metric_keys=metric_keys,
        corner_metric_keys=corner_metric_keys,
        locations=locations,
        metric_display=metric_display_for_keys(
            display_keys,
            metric_specs_for_suspension(suspension),
        ),
        sweep_parameters=sweep_parameters(sweep_config, evaluated.states),
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
        drive_coordinates=[
            _coordinate_info(coordinate)
            for coordinate in suspension.drive_coordinates()
        ],
        suspension_hold_catalogue=_suspension_hold_catalogue_info(
            suspension.suspension_hold_catalogue()
        ),
    )
