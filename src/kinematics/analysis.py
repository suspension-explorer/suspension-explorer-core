"""
High-level sweep analysis for front-ends.

This module is the package's one-call answer to "sweep this geometry this
way and give me everything a user-facing tool needs": solved frames with
positions and full metric rows (including derivative metrics such as motion
ratios and camber gain), metric display metadata, the name-keyed display
topology for 3D rendering, swept-parameter descriptors for chart axes, the
"setup" reference condition, and advisory diagnostics.

Consumers (the CLI, an API server, a notebook) should not need to know about
tangents, automatic differentiation, or display geometry -- those live in
their own modules and are orchestrated here. Adapters only translate the
returned dataclasses into their transport format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kinematics.core.enums import TargetPositionMode
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import PointTarget, SweepConfig
from kinematics.diagnostics import DiagnosticIssue, diagnose_sweep
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.metrics.main import AxleMetricRows, MetricRow
from kinematics.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.solver import SolverInfo
from kinematics.suspensions.base import Suspension
from kinematics.visualization.display import (
    DisplayLink,
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
    """
    Identifying metadata of the analyzed suspension.
    """

    name: str
    type_key: str
    units: str


@dataclass(frozen=True)
class SweepParameter:
    """
    One axis-aligned swept dimension, usable as a chart X axis.

    Attributes:
        point: Position key of the swept point (side-prefixed for an axle,
            so it can be looked up in frame positions directly).
        axis: Principal axis swept along: "X", "Y", or "Z".
        side: "left" / "right" for side-qualified axle targets, else None.
    """

    point: str
    axis: str
    side: str | None


@dataclass(frozen=True)
class AnalyzedFrame:
    """
    One solved sweep step, ready for display.

    Attributes:
        index: Step index within the sweep.
        positions: Name-keyed positions, including synthetic display points
            (rocker lever-arm feet).
        metrics: The model-level metric row: a corner model's full row, or
            the axle-level row of an axle model (location-less keys).
        corner_metrics: Location ("left" / "right") -> that corner's metric
            row; empty for corner models. Keys within each row are
            location-less; canonical flat names are rendered only at export
            boundaries via kinematics.metrics.main.flatten_metric_rows.
        solver: Solver convergence info for the step.
    """

    index: int
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]
    solver: SolverInfo


@dataclass(frozen=True)
class ReferenceCondition:
    """
    A reference pose the sweep can be compared against.

    ``metrics`` / ``corner_metrics`` follow the same structured shape as
    :class:`AnalyzedFrame`.
    """

    label: str
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]


@dataclass(frozen=True)
class StaticPose:
    """
    The as-assembled initial pose of a geometry, for live preview.
    """

    suspension: SuspensionInfo
    point_keys: list[str]
    positions: Positions
    wheel: WheelDisplayDimensions | None
    links: list[DisplayLink]
    wheel_anchors: list[WheelAnchorNames]


@dataclass(frozen=True)
class SweepAnalysis:
    """
    The complete result of analyzing a sweep, ready for any front-end.

    Metric structure mirrors the frames: locations are structural, never
    key-mangled. ``metric_keys`` lists the model-level row's base keys (the
    axle-level row for an axle, the corner's own row for a corner model);
    ``corner_metric_keys`` lists the base keys of the per-corner rows and
    ``locations`` the corner locations present (both empty for corner
    models). ``metric_display`` carries display metadata for every base key
    (corner keys first, then model keys). Flat column names exist only in
    result files, rendered by kinematics.metrics.main.flatten_metric_rows.
    """

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
        """
        Number of solved steps.
        """
        return len(self.frames)


def _suspension_info(suspension: Suspension) -> SuspensionInfo:
    """
    Extract the identifying metadata of a suspension.
    """
    return SuspensionInfo(
        name=suspension.name,
        type_key=suspension.TYPE_KEY,
        units=suspension.units.symbol,
    )


def sweep_parameters(sweep_config: SweepConfig) -> list[SweepParameter]:
    """
    Describe each axis-aligned swept dimension of a sweep.

    Each dimension is a list of PointTargets (one per step) sharing a point
    and direction, read off the first step. Vector-direction targets have no
    single principal axis and are skipped.
    """
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
        parameters.append(SweepParameter(point=key.name, axis=axis.name, side=side))
    return parameters


def _hold_sweep_config(sweep_config: SweepConfig) -> SweepConfig | None:
    """
    A one-step sweep holding every swept target at relative zero.

    This reproduces the as-assembled nominal pose the sweep departs from
    (the "setup" condition): the same targets as the sweep, each commanding
    zero displacement from the design position.
    """
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
    if not hold_dimensions:
        return None
    return SweepConfig(hold_dimensions)


def _setup_reference(
    suspension: Suspension,
    sweep_config: SweepConfig,
    point_keys,
    rocker_groups,
) -> ReferenceCondition | None:
    """
    Solve the "setup" reference condition, best-effort.

    Because any configured setup shims are baked into initial_state(), the
    solved pose reflects them -- hence "setup" rather than "design". A
    shim-free "design" reference can be added alongside this later. Returns
    None when there is nothing to solve or the solve fails; a missing
    reference must never fail the analysis.
    """
    hold_config = _hold_sweep_config(sweep_config)
    if hold_config is None:
        return None

    try:
        states, _stats = solve_sweep(suspension, hold_config)
        if not states:
            return None
        rows = compute_sweep_metrics(suspension, hold_config, states)[0]
    except Exception:  # noqa: BLE001 - the reference is optional
        return None

    metrics, corner_metrics = _split_metric_rows(rows)
    return ReferenceCondition(
        label="Setup",
        positions=display_positions(states[0].positions, point_keys, rocker_groups),
        metrics=metrics,
        corner_metrics=corner_metrics,
    )


def _split_metric_rows(
    rows: "MetricRow | AxleMetricRows",
) -> tuple[MetricRow, dict[str, MetricRow]]:
    """
    Normalize a per-state metrics result into (model row, corner rows).

    Corner models produce a single row and no corner map; axle models carry
    an axle-level row plus one row per corner location.
    """
    if isinstance(rows, AxleMetricRows):
        return rows.axle, rows.corners
    return rows, {}


def analyze_sweep(suspension: Suspension, sweep_config: SweepConfig) -> SweepAnalysis:
    """
    Solve a sweep and assemble the complete front-end-ready analysis.

    Args:
        suspension: The suspension to analyze.
        sweep_config: The sweep to run.

    Returns:
        The full analysis: frames (positions, metrics, solver info), metric
        display metadata, display topology, sweep parameters, references,
        and diagnostics.

    Raises:
        RuntimeError: If the solver fails to converge (from solve_sweep);
            adapters translate this into their own error type.
    """
    states, solver_stats = solve_sweep(suspension, sweep_config)
    metric_rows = compute_sweep_metrics(suspension, sweep_config, states)

    point_keys = display_point_keys(suspension)
    rocker_groups = rocker_display_groups(suspension)

    frames: list[AnalyzedFrame] = []
    for index, (state, info) in enumerate(zip(states, solver_stats)):
        metrics, corner_metrics = _split_metric_rows(metric_rows[index])
        frames.append(
            AnalyzedFrame(
                index=index,
                positions=display_positions(state.positions, point_keys, rocker_groups),
                metrics=metrics,
                corner_metrics=corner_metrics,
                solver=info,
            )
        )

    # Base-key lists from the first non-empty frame: the model-level row's
    # keys, the per-corner rows' keys, and the corner locations present.
    # Display metadata covers every base key, corner keys first.
    metric_keys: list[str] = []
    corner_metric_keys: list[str] = []
    locations: list[str] = []
    for frame in frames:
        if not frame.metrics and not frame.corner_metrics:
            continue
        metric_keys = list(frame.metrics.keys())
        locations = list(frame.corner_metrics.keys())
        for row in frame.corner_metrics.values():
            corner_metric_keys = list(row.keys())
            break
        break

    display_keys = corner_metric_keys + [
        key for key in metric_keys if key not in corner_metric_keys
    ]

    references: dict[str, ReferenceCondition] = {}
    setup = _setup_reference(suspension, sweep_config, point_keys, rocker_groups)
    if setup is not None:
        references["setup"] = setup

    # Diagnostics are advisory: any internal failure yields an empty list
    # rather than failing an analysis that already produced frames.
    try:
        diagnostics = list(diagnose_sweep(suspension, states, solver_stats).issues)
    except Exception:  # noqa: BLE001 - diagnostics must never break a solve
        diagnostics = []

    return SweepAnalysis(
        suspension=_suspension_info(suspension),
        point_keys=[key.name for key in point_keys],
        metric_keys=metric_keys,
        corner_metric_keys=corner_metric_keys,
        locations=locations,
        metric_display=metric_display_for_keys(display_keys),
        sweep_parameters=sweep_parameters(sweep_config),
        references=references,
        wheel=wheel_display_dimensions(suspension.config),
        links=display_links(suspension),
        wheel_anchors=wheel_anchor_names(suspension),
        diagnostics=diagnostics,
        frames=frames,
    )


def initial_pose(suspension: Suspension) -> StaticPose:
    """
    Evaluate a geometry at its static initial pose, without running a sweep.

    Powers live previews while a user edits hardpoints/config: the
    as-assembled pose (hardpoints plus derived points, with any setup shims
    applied) and the display topology to draw it.
    """
    state = suspension.initial_state()
    point_keys = display_point_keys(suspension)
    rocker_groups = rocker_display_groups(suspension)
    return StaticPose(
        suspension=_suspension_info(suspension),
        point_keys=[key.name for key in point_keys],
        positions=display_positions(state.positions, point_keys, rocker_groups),
        wheel=wheel_display_dimensions(suspension.config),
        links=display_links(suspension),
        wheel_anchors=wheel_anchor_names(suspension),
    )
