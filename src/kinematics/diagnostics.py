"""
Post-sweep diagnostics for suspension solves.

Diagnostics are advisory and inspect a completed sweep without changing solver
behaviour. The checks in this module are topology-independent; topology-specific
checks belong with the topology that defines them.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from kinematics.core.constants import SOLVE_ACCEPT_RESIDUAL
from kinematics.core.point_ref import PointKey

if TYPE_CHECKING:
    from kinematics.solver import SolverInfo
    from kinematics.state import SuspensionState
    from kinematics.suspensions.base import Suspension


# A jump must exceed both a physical floor and the point's typical step size.
# The floor suppresses numerical jitter while the relative threshold adapts to
# sweeps whose ordinary steps are larger.
CONTINUITY_ABS_FLOOR_MM: float = 5.0
CONTINUITY_MEDIAN_FACTOR: float = 4.0


@dataclass(frozen=True)
class DiagnosticIssue:
    """
    A single diagnostic finding about a solved sweep.

    Attributes:
        step: Sweep step index, or ``None`` for a sweep-wide issue.
        category: Machine-readable issue category.
        severity: ``"error"`` or ``"warning"``.
        message: Human-readable description.
        value: Salient numeric value, when applicable.
    """

    step: int | None
    category: str
    severity: str
    message: str
    value: float | None


@dataclass
class SweepDiagnostics:
    """
    Collected diagnostics for one solved sweep.
    """

    issues: list[DiagnosticIssue]

    @property
    def ok(self) -> bool:
        """Return whether the report contains no errors."""
        return not self.errors

    @property
    def warnings(self) -> list[DiagnosticIssue]:
        """Return warning-severity issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[DiagnosticIssue]:
        """Return error-severity issues."""
        return [issue for issue in self.issues if issue.severity == "error"]


def diagnose_sweep(
    suspension: Suspension,
    states: list[SuspensionState],
    stats: list[SolverInfo],
) -> SweepDiagnostics:
    """
    Run topology-independent checks over a completed sweep.

    Args:
        suspension: Suspension that produced the states.
        states: Solved states in sweep order.
        stats: Solver information aligned with the states.

    Returns:
        Aggregated diagnostic findings.
    """
    issues = _check_convergence_and_residual(stats)
    issues.extend(_check_continuity(suspension, states))
    issues.extend(suspension.topology_diagnostics(states))
    return SweepDiagnostics(issues=issues)


def _check_convergence_and_residual(
    stats: list[SolverInfo],
) -> list[DiagnosticIssue]:
    """Report non-convergence and residuals above the acceptance threshold."""
    issues: list[DiagnosticIssue] = []
    for step, info in enumerate(stats):
        if not info.converged:
            issues.append(
                DiagnosticIssue(
                    step=step,
                    category="convergence",
                    severity="error",
                    message=f"Step {step} did not converge.",
                    value=None,
                )
            )
        if info.max_residual > SOLVE_ACCEPT_RESIDUAL:
            issues.append(
                DiagnosticIssue(
                    step=step,
                    category="residual",
                    severity="error",
                    message=(
                        f"Step {step} residual {info.max_residual:.6g} exceeds the "
                        f"acceptance tolerance {SOLVE_ACCEPT_RESIDUAL:.6g}."
                    ),
                    value=info.max_residual,
                )
            )
    return issues


def _check_continuity(
    suspension: Suspension,
    states: list[SuspensionState],
) -> list[DiagnosticIssue]:
    """Report free-point jumps that are large relative to ordinary sweep steps."""
    if len(states) < 2:
        return []

    issues: list[DiagnosticIssue] = []
    for key in suspension.free_points():
        displacements = _point_step_displacements(states, key)
        nonzero = [displacement for displacement in displacements if displacement > 0]
        typical_step = median(nonzero) if nonzero else 0.0
        threshold = max(
            CONTINUITY_ABS_FLOOR_MM,
            CONTINUITY_MEDIAN_FACTOR * typical_step,
        )
        for previous_step, displacement in enumerate(displacements):
            if displacement <= threshold:
                continue
            step = previous_step + 1
            name = getattr(key, "name", str(key))
            issues.append(
                DiagnosticIssue(
                    step=step,
                    category="jump",
                    severity="warning",
                    message=(
                        f"Point '{name}' jumped {displacement:.3g} mm from step "
                        f"{previous_step} to step {step} (threshold "
                        f"{threshold:.3g} mm); possible branch snap."
                    ),
                    value=displacement,
                )
            )
    return issues


def _point_step_displacements(
    states: list[SuspensionState],
    key: PointKey,
) -> list[float]:
    """Return Euclidean displacement of one point for each sweep transition."""
    displacements: list[float] = []
    previous = states[0].positions.get(key)
    for state in states[1:]:
        current = state.positions.get(key)
        if previous is None or current is None:
            displacements.append(0.0)
        else:
            displacements.append(float(np.linalg.norm(current.data - previous.data)))
        previous = current
    return displacements
