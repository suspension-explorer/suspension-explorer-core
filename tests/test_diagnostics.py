"""
Tests for chirality, residual acceptance, and the diagnostics module.

Covers the three defects the rocker/ARB axle exhibited on out-of-range sweeps:

1. ``ScalarTripleProductConstraint`` pins rocker handedness (mirror branch).
2. The solver rejects a converged-but-infeasible least-squares compromise.
3. ``diagnose_sweep`` reports convergence/residual, branch-snap jumps, chirality
   inversion, and transmission (topping-out) margin.

Feasible-travel notes for ``axle_geometry_rocker.yaml`` (found by incremental
bisection with the chirality constraint active; a steering-pinned quasi-static
sweep from design). These bound the "in-range" sweeps used below:

- left corner bump (+Z wheel): feasible to ~+118 mm.
- left corner droop (-Z wheel): feasible to ~-72 mm.
- roll (left up / right down, opposite equal): feasible to ~+/-46 mm per side
  before the ARB-coupled droplink tops out.

The original reproduction sweep (left Z -80 -> +80, right Z +90 -> -80) starts at
an out-of-range roll (left -80, right +90), so step 0 is genuinely infeasible: the
solver now raises the informative RuntimeError at step 0 instead of folding the
left rocker onto its +540000 mirror branch and creeping out garbage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinematics.constraints import ScalarTripleProductConstraint
from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetPositionMode,
)
from kinematics.core.vector_utils.geometric import compute_scalar_triple_product
from kinematics.diagnostics import (
    TRANSMISSION_MARGIN_WARN,
    DiagnosticIssue,
    SweepDiagnostics,
    diagnose_sweep,
)
from kinematics.io.geometry_loader import load_geometry
from kinematics.jacobians import jac_coplanar
from kinematics.main import solve_sweep
from kinematics.state import SuspensionState

FD_STEP = 1e-6
FD_TOLERANCE = 1e-6


def _rel(point_key, axis: Axis, value: float) -> PointTarget:
    return PointTarget(
        point_id=point_key,
        direction=PointTargetAxis(axis),
        value=value,
        mode=TargetPositionMode.RELATIVE,
    )


def _axle_sweep(
    heave_left: list[float],
    heave_right: list[float],
    steer_left: list[float],
) -> SweepConfig:
    return SweepConfig(
        [
            [
                _rel(PointRef(Side.LEFT, PointID.WHEEL_CENTER), Axis.Z, v)
                for v in heave_left
            ],
            [
                _rel(PointRef(Side.RIGHT, PointID.WHEEL_CENTER), Axis.Z, v)
                for v in heave_right
            ],
            [
                _rel(PointRef(Side.LEFT, PointID.TRACKROD_INBOARD), Axis.Y, v)
                for v in steer_left
            ],
        ]
    )


@pytest.fixture
def axle_rocker_file(test_data_dir: Path) -> Path:
    return test_data_dir / "axle_geometry_rocker.yaml"


# ----------------------------------------------------------------------
# 1. ScalarTripleProductConstraint
# ----------------------------------------------------------------------


class TestScalarTripleProductConstraint:
    """The chirality constraint's residual, remap, and Jacobian."""

    # A non-degenerate, non-coplanar quadruple (design has a real signed volume).
    P1 = PointID.ROCKER_AXIS_FRONT
    P2 = PointID.ROCKER_AXIS_REAR
    P3 = PointID.PUSHROD_INBOARD
    P4 = PointID.ROCKER_DROPLINK

    def _design_positions(self) -> dict[PointID, Point3]:
        return {
            self.P1: Point3([100.0, 340.0, 450.0]),
            self.P2: Point3([-100.0, 340.0, 450.0]),
            self.P3: Point3([0.0, 430.0, 480.0]),
            self.P4: Point3([0.0, 250.0, 450.0]),
        }

    def _design_triple(self, pos: dict[PointID, Point3]) -> float:
        return compute_scalar_triple_product(
            pos[self.P2] - pos[self.P1],
            pos[self.P3] - pos[self.P1],
            pos[self.P4] - pos[self.P1],
        )

    def test_zero_residual_at_design(self) -> None:
        pos = self._design_positions()
        target = self._design_triple(pos)
        scale = max(abs(target), 1.0)
        c = ScalarTripleProductConstraint(
            self.P1, self.P2, self.P3, self.P4, target_volume=target, scale=scale
        )
        assert c.residual(pos) == pytest.approx(0.0, abs=1e-9)

    def test_mirrored_droplink_residual(self) -> None:
        # Reflect the droplink through the plane containing the axis and the
        # pushrod pickup: this satisfies every distance but flips the triple sign,
        # so the residual becomes ~ -2 * target / scale (magnitude 2|target|/scale).
        pos = self._design_positions()
        target = self._design_triple(pos)
        scale = max(abs(target), 1.0)

        p0 = pos[self.P1]
        normal = (pos[self.P2] - p0).cross(pos[self.P3] - p0).normalize()
        d = pos[self.P4] - p0
        reflected = pos[self.P4] - normal * (2.0 * d.dot(normal))
        mirrored = dict(pos)
        mirrored[self.P4] = reflected

        c = ScalarTripleProductConstraint(
            self.P1, self.P2, self.P3, self.P4, target_volume=target, scale=scale
        )
        residual = c.residual(mirrored)
        assert abs(residual) == pytest.approx(2.0 * abs(target) / scale, rel=1e-6)

    def test_remap_round_trip(self) -> None:
        target, scale = -540000.0, 540000.0
        c = ScalarTripleProductConstraint(
            self.P1, self.P2, self.P3, self.P4, target_volume=target, scale=scale
        )
        remapped = c.remap(lambda pid: PointRef(Side.LEFT, pid))
        assert isinstance(remapped, ScalarTripleProductConstraint)
        assert remapped.involved_points == {
            PointRef(Side.LEFT, self.P1),
            PointRef(Side.LEFT, self.P2),
            PointRef(Side.LEFT, self.P3),
            PointRef(Side.LEFT, self.P4),
        }
        assert remapped.target_volume == target
        assert remapped.scale == scale
        # Round-trip back to plain PointID keys.
        back = remapped.remap(lambda ref: ref.point)
        assert back.involved_points == {self.P1, self.P2, self.P3, self.P4}

    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="scale must be strictly positive"):
            ScalarTripleProductConstraint(
                self.P1, self.P2, self.P3, self.P4, target_volume=1.0, scale=0.0
            )

    def test_analytic_jacobian_matches_finite_difference(self) -> None:
        pos = self._design_positions()
        target = self._design_triple(pos)
        scale = max(abs(target), 1.0)
        c = ScalarTripleProductConstraint(
            self.P1, self.P2, self.P3, self.P4, target_volume=target, scale=scale
        )

        # Analytic row: coplanar Jacobian scaled by 1/scale (matches solver branch).
        analytic = (
            jac_coplanar(
                pos[self.P1].data,
                pos[self.P2].data,
                pos[self.P3].data,
                pos[self.P4].data,
            )
            / scale
        )

        order = [self.P1, self.P2, self.P3, self.P4]
        numerical = np.zeros(12)
        for k, pid in enumerate(order):
            base = pos[pid].data.copy()
            for axis in range(3):
                plus = dict(pos)
                minus = dict(pos)
                bp = base.copy()
                bp[axis] += FD_STEP
                bm = base.copy()
                bm[axis] -= FD_STEP
                plus[pid] = Point3(bp)
                minus[pid] = Point3(bm)
                numerical[3 * k + axis] = (c.residual(plus) - c.residual(minus)) / (
                    2.0 * FD_STEP
                )
        np.testing.assert_allclose(analytic, numerical, atol=FD_TOLERANCE)


# ----------------------------------------------------------------------
# 2. Solver residual acceptance
# ----------------------------------------------------------------------


class TestSolverAcceptance:
    """The sweep solver rejects a converged-but-infeasible compromise."""

    def test_infeasible_sweep_raises_with_step_and_constraint(
        self, axle_rocker_file: Path
    ) -> None:
        axle = load_geometry(axle_rocker_file)
        # Wheel Z target far beyond travel on both sides.
        cfg = _axle_sweep([300.0], [-300.0], [0.0])
        with pytest.raises(RuntimeError) as exc:
            solve_sweep(axle, cfg)
        message = str(exc.value)
        assert "step 0" in message
        # The worst residual row is named (a constraint or a target).
        assert "Worst residual row" in message
        assert "kinematic lock-out" in message

    def test_healthy_sweep_does_not_raise(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        states, stats = solve_sweep(axle, _axle_sweep([10.0], [10.0], [0.0]))
        assert all(s.converged for s in stats)
        assert all(s.max_residual < 1e-3 for s in stats)


# ----------------------------------------------------------------------
# 3. Chirality in anger (in-range reproduction sweep)
# ----------------------------------------------------------------------


class TestChiralityInRange:
    """An in-range roll sweep keeps the triple sign constant on both sides."""

    def _triple(self, state: SuspensionState, side: Side) -> float:
        pos = state.positions
        af = pos[PointRef(side, PointID.ROCKER_AXIS_FRONT)]
        ar = pos[PointRef(side, PointID.ROCKER_AXIS_REAR)]
        pin = pos[PointRef(side, PointID.PUSHROD_INBOARD)]
        dl = pos[PointRef(side, PointID.ROCKER_DROPLINK)]
        return compute_scalar_triple_product(ar - af, pin - af, dl - af)

    def test_in_range_reproduction_sign_constant(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        design = axle.initial_state()
        design_sign = {
            side: np.sign(self._triple(design, side))
            for side in (Side.LEFT, Side.RIGHT)
        }
        assert design_sign[Side.LEFT] < 0 and design_sign[Side.RIGHT] > 0

        # In-range version of the reproduction motion: left droop->bump,
        # right bump->droop, passing through design. +/-40 mm stays inside the
        # ~+/-46 mm roll limit.
        n = 21
        left_z = list(np.linspace(-40.0, 40.0, n))
        right_z = list(np.linspace(40.0, -40.0, n))
        steer = [0.0] * n
        states, stats = solve_sweep(axle, _axle_sweep(left_z, right_z, steer))

        assert all(s.converged for s in stats)
        assert all(s.max_residual < 1e-3 for s in stats)
        for st in states:
            for side in (Side.LEFT, Side.RIGHT):
                assert np.sign(self._triple(st, side)) == design_sign[side]

        # And the diagnostics agree: no chirality errors.
        diag = diagnose_sweep(axle, states, stats)
        assert not [i for i in diag.issues if i.category == "chirality"]


# ----------------------------------------------------------------------
# 4. Continuity check (synthetic states)
# ----------------------------------------------------------------------


class TestContinuityCheck:
    """The jump check flags an injected discontinuity and nothing else."""

    def test_injected_jump_flagged_at_right_step(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        # A smooth in-range heave sweep as the baseline.
        heave = list(np.linspace(0.0, 20.0, 11))
        states, stats = solve_sweep(axle, _axle_sweep(heave, heave, [0.0] * len(heave)))
        # Inject a large jump into one free point at a specific step.
        jump_step = 6
        jump_key = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
        # WHEEL_CENTER is derived, not free; pick a free point instead.
        jump_key = PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD)
        assert jump_key in states[jump_step].positions
        bumped = states[jump_step].copy()
        p = bumped.positions[jump_key]
        bumped.positions[jump_key] = Point3(p.data + np.array([0.0, 0.0, 50.0]))
        states[jump_step] = bumped

        diag = diagnose_sweep(axle, states, stats)
        jumps = [i for i in diag.issues if i.category == "jump"]
        assert jumps, "expected a jump warning"
        # The injected jump shows up leaving and returning: at jump_step and
        # jump_step+1. The step INTO the bumped state must be flagged.
        flagged_steps = {i.step for i in jumps}
        assert jump_step in flagged_steps
        assert all(i.severity == "warning" for i in jumps)

    def test_smooth_states_no_jump(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        heave = list(np.linspace(0.0, 20.0, 11))
        states, stats = solve_sweep(axle, _axle_sweep(heave, heave, [0.0] * len(heave)))
        diag = diagnose_sweep(axle, states, stats)
        assert not [i for i in diag.issues if i.category == "jump"]


# ----------------------------------------------------------------------
# 5. Transmission margin
# ----------------------------------------------------------------------


class TestTransmissionMargin:
    """Topping-out drives a transmission warning; gentle heave does not."""

    def test_near_limit_warns_naming_joint(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        # Drive the left corner into droop right up to its ~-72 mm limit, where
        # the pushrod approaches its rocker toggle (margin ~0.127 < 0.15).
        n = 30
        droop = list(np.linspace(0.0, 72.0, n))
        states, stats = solve_sweep(
            axle,
            _axle_sweep([-v for v in droop], [0.0] * n, [0.0] * n),
        )
        diag = diagnose_sweep(axle, states, stats)
        trans = [i for i in diag.issues if i.category == "transmission"]
        assert trans, "expected a transmission warning near the travel limit"
        # Message names a joint and a margin below the threshold.
        assert all(i.severity == "warning" for i in trans)
        assert all(
            i.value is not None and i.value < TRANSMISSION_MARGIN_WARN for i in trans
        )
        assert any(
            ("PUSHROD_INBOARD" in i.message)
            or ("ROCKER_DROPLINK" in i.message)
            or ("ARB_DROPLINK" in i.message)
            for i in trans
        )

    def test_small_heave_no_transmission_warning(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        heave = [5.0, 10.0]
        states, stats = solve_sweep(axle, _axle_sweep(heave, heave, [0.0, 0.0]))
        diag = diagnose_sweep(axle, states, stats)
        assert not [i for i in diag.issues if i.category == "transmission"]


# ----------------------------------------------------------------------
# 6. SweepDiagnostics API + CLI smoke
# ----------------------------------------------------------------------


class TestSweepDiagnosticsApi:
    """The aggregate helper's ok / warnings / errors partition correctly."""

    def test_partition(self) -> None:
        diag = SweepDiagnostics(
            issues=[
                DiagnosticIssue(0, "jump", "warning", "w", 1.0),
                DiagnosticIssue(1, "residual", "error", "e", 2.0),
            ]
        )
        assert not diag.ok
        assert len(diag.warnings) == 1
        assert len(diag.errors) == 1

    def test_ok_with_only_warnings(self) -> None:
        diag = SweepDiagnostics(
            issues=[DiagnosticIssue(0, "jump", "warning", "w", 1.0)]
        )
        assert diag.ok


class TestCliDiagnostics:
    """CLI prints diagnostics to stderr; a clean sweep prints none."""

    def test_cli_prints_warnings_to_stderr(
        self, tmp_path: Path, axle_rocker_file: Path, capsys
    ) -> None:
        import yaml

        from kinematics.cli import sweep as cli_sweep

        # A deep-droop sweep that stays feasible but tops out the pushrod
        # (transmission warnings), written as a temporary sweep file in the
        # loader's schema.
        def target(point: str, side: str, axis: str, stop: float) -> dict:
            return {
                "point": point,
                "side": side,
                "direction": {"axis": axis},
                "mode": "relative",
                "start": 0.0,
                "stop": stop,
            }

        sweep_yaml = {
            "version": 1,
            "steps": 30,
            "targets": [
                target("WHEEL_CENTER", "left", "Z", -72.0),
                target("WHEEL_CENTER", "right", "Z", 0.0),
                target("TRACKROD_INBOARD", "left", "Y", 0.0),
            ],
        }
        sweep_path = tmp_path / "droop_sweep.yaml"
        sweep_path.write_text(yaml.safe_dump(sweep_yaml))
        out = tmp_path / "out.csv"

        cli_sweep(
            geometry=axle_rocker_file,
            sweep=sweep_path,
            out=out,
            animation_out=None,
        )
        assert out.exists()
        captured = capsys.readouterr()
        assert "WARNING:" in captured.err
        assert "Diagnostics:" in captured.err

    def test_cli_clean_sweep_prints_no_diagnostics(
        self, tmp_path: Path, axle_rocker_file: Path, test_data_dir: Path, capsys
    ) -> None:
        from kinematics.cli import sweep as cli_sweep

        out = tmp_path / "clean.csv"
        cli_sweep(
            geometry=axle_rocker_file,
            sweep=test_data_dir / "axle_rocker_sweep.yaml",
            out=out,
            animation_out=None,
        )
        captured = capsys.readouterr()
        assert "WARNING:" not in captured.err
        assert "ERROR:" not in captured.err
        assert "Diagnostics:" not in captured.err
