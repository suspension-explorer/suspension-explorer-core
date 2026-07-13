"""Tests for topology-independent sweep diagnostics."""

from types import SimpleNamespace

import pytest

from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
    SweepDiagnostics,
    diagnose_sweep,
)
from kinematics.core.primitives.constants import SOLVE_ACCEPT_RESIDUAL
from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.geometry import Point3
from kinematics.core.solver import SolverInfo
from kinematics.core.state import SuspensionState


def _state(x: float) -> SuspensionState:
    return SuspensionState(
        positions={PointID.WHEEL_CENTER: Point3([x, 0.0, 0.0])},
        free_points={PointID.WHEEL_CENTER},
    )


def _suspension():
    return SimpleNamespace(
        free_points=lambda: (PointID.WHEEL_CENTER,),
        topology_diagnostics=lambda _states: [],
    )


def test_sweep_diagnostics_groups_issues_by_severity() -> None:
    warning = DiagnosticIssue(
        1,
        DiagnosticCategory.JUMP,
        DiagnosticSeverity.WARNING,
        "jump",
        10.0,
    )
    error = DiagnosticIssue(
        2,
        DiagnosticCategory.RESIDUAL,
        DiagnosticSeverity.ERROR,
        "residual",
        1.0,
    )

    diagnostics = SweepDiagnostics([warning, error])

    assert diagnostics.warnings == [warning]
    assert diagnostics.errors == [error]
    assert not diagnostics.ok


def test_diagnose_sweep_reports_convergence_and_residual_errors() -> None:
    stats = [
        SolverInfo(converged=True, nfev=1, max_residual=0.0),
        SolverInfo(
            converged=False,
            nfev=10,
            max_residual=SOLVE_ACCEPT_RESIDUAL * 2.0,
        ),
    ]

    diagnostics = diagnose_sweep(_suspension(), [_state(0.0), _state(1.0)], stats)

    assert [issue.category for issue in diagnostics.errors] == [
        "convergence",
        "residual",
    ]
    assert all(issue.step == 1 for issue in diagnostics.errors)


def test_diagnose_sweep_accepts_smooth_motion() -> None:
    states = [_state(float(x)) for x in range(6)]
    stats = [SolverInfo(True, 1, 0.0) for _ in states]

    diagnostics = diagnose_sweep(_suspension(), states, stats)

    assert diagnostics.ok
    assert diagnostics.issues == []


def test_diagnose_sweep_reports_discontinuous_free_point_motion() -> None:
    states = [_state(x) for x in (0.0, 1.0, 2.0, 22.0, 23.0, 24.0)]
    stats = [SolverInfo(True, 1, 0.0) for _ in states]

    diagnostics = diagnose_sweep(_suspension(), states, stats)

    assert diagnostics.ok
    assert len(diagnostics.warnings) == 1
    issue = diagnostics.warnings[0]
    assert issue.category == "jump"
    assert issue.step == 3
    assert issue.value == pytest.approx(20.0)


def test_continuity_ignores_points_missing_from_a_state() -> None:
    missing = SuspensionState(positions={}, free_points=set())
    states = [_state(0.0), missing, _state(100.0)]
    stats = [SolverInfo(True, 1, 0.0) for _ in states]

    diagnostics = diagnose_sweep(_suspension(), states, stats)

    assert diagnostics.issues == []
