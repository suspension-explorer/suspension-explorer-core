"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List

from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
    diagnose_sweep,
)
from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.sensitivity import (
    TangentField,
    TangentSolveInfo,
    compute_state_tangents,
)
from kinematics.core.solver import (
    SolverInfo,
    convert_targets_to_absolute,
    solve_suspension_sweep,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.targeting import SweepConfig


def solve_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> tuple[List[SuspensionState], List[SolverInfo]]:
    """
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by setting up derived point calculations and running the solver across target
    configurations.

    Args:
        suspension: The Suspension instance containing geometry and behavior.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        Tuple containing the list of solved suspension states and corresponding
        solver information for each step in the sweep.
    """
    derived_spec = suspension.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states, solver_stats = solve_suspension_sweep(
        initial_state=suspension.initial_state(),
        constraints=suspension.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states, solver_stats


@dataclass(frozen=True)
class SweepTangents:
    """
    Per-state tangent fields and their numerical solve health.
    """

    per_step: list[list[TangentField]]
    solve_infos: list[TangentSolveInfo]


@dataclass(frozen=True)
class SweepMetricsResult:
    """
    Metric rows plus visible derivative-computation status.
    """

    rows: list[MetricRow | AxleMetricRows]
    derivative_error: str | None = None
    tangent_solve_infos: list[TangentSolveInfo] | None = None


@dataclass(frozen=True)
class EvaluatedSweep:
    """
    Lean solved sweep result shared by transport and presentation adapters.
    """

    states: list[SuspensionState]
    solver_stats: list[SolverInfo]
    metrics: SweepMetricsResult
    diagnostics: list[DiagnosticIssue]

    def __post_init__(self) -> None:
        """
        Reject partial results that would truncate adapter output.
        """
        lengths = (len(self.states), len(self.solver_stats), len(self.metrics.rows))
        if len(set(lengths)) != 1:
            raise ValueError(
                "Evaluated sweep state, solver-stat, and metric counts must match: "
                f"{lengths[0]} states, {lengths[1]} solver stats, "
                f"{lengths[2]} metric rows."
            )


def compute_sweep_tangents(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepTangents:
    """
    Compute solution-manifold tangents for every solved sweep state.
    """
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    constraints = suspension.constraints()
    initial_state = suspension.initial_state()
    tangents_per_step: list[list[TangentField]] = []
    solve_infos: list[TangentSolveInfo] = []

    for step_index, state in enumerate(states):
        step_targets = convert_targets_to_absolute(
            [target_sweep[step_index] for target_sweep in sweep_config.target_sweeps],
            initial_state,
        )
        fields, solve_info = compute_state_tangents(
            state,
            constraints,
            derived_manager,
            step_targets,
        )
        tangents_per_step.append(fields)
        solve_infos.append(solve_info)

    return SweepTangents(per_step=tangents_per_step, solve_infos=solve_infos)


def compute_sweep_metrics(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepMetricsResult:
    """
    Compute all sweep metrics, reporting derivative failures explicitly.
    """
    if suspension.config is None:
        return SweepMetricsResult(rows=[OrderedDict() for _ in states])

    derivative_error: str | None = None
    try:
        tangents = compute_sweep_tangents(suspension, sweep_config, states)
    except Exception as error:  # noqa: BLE001 - metrics degrade without derivatives
        tangents = None
        derivative_error = f"{type(error).__name__}: {error}"

    rows = [
        suspension.compute_state_metrics(
            state,
            tangents.per_step[index] if tangents is not None else None,
        )
        for index, state in enumerate(states)
    ]
    return SweepMetricsResult(
        rows=rows,
        derivative_error=derivative_error,
        tangent_solve_infos=tangents.solve_infos if tangents is not None else None,
    )


def _derivative_issues(result: SweepMetricsResult) -> list[DiagnosticIssue]:
    """
    Turn tangent-computation health into visible advisory diagnostics.
    """
    issues: list[DiagnosticIssue] = []
    if result.derivative_error is not None:
        issues.append(
            DiagnosticIssue(
                step=None,
                category=DiagnosticCategory.DERIVATIVES,
                severity=DiagnosticSeverity.WARNING,
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
                category=DiagnosticCategory.DERIVATIVES,
                severity=DiagnosticSeverity.WARNING,
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


def evaluate_solved_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: list[SuspensionState],
    solver_stats: list[SolverInfo],
) -> EvaluatedSweep:
    """
    Compute metrics and diagnostics for an already solved sweep.
    """
    if len(states) != len(solver_stats):
        raise ValueError(
            "Solved state and solver-stat counts must match: "
            f"{len(states)} states, {len(solver_stats)} solver stats."
        )

    metrics = compute_sweep_metrics(suspension, sweep_config, states)
    try:
        diagnostics = list(diagnose_sweep(suspension, states, solver_stats).issues)
    except Exception as error:  # noqa: BLE001 - diagnostics are advisory
        diagnostics = [
            DiagnosticIssue(
                step=None,
                category=DiagnosticCategory.DIAGNOSTICS,
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "Sweep diagnostics unavailable: diagnostic evaluation failed "
                    f"({type(error).__name__}: {error})."
                ),
                value=None,
            )
        ]
    diagnostics.extend(_derivative_issues(metrics))
    return EvaluatedSweep(
        states=states,
        solver_stats=solver_stats,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def solve_evaluated_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> EvaluatedSweep:
    """
    Solve one sweep and compute its metrics and advisory diagnostics.
    """
    states, solver_stats = solve_sweep(suspension, sweep_config)
    return evaluate_solved_sweep(
        suspension,
        sweep_config,
        states,
        solver_stats,
    )
