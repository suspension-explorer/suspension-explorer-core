"""
Tests for the double wishbone axle model (two coupled corners).

Coverage:
1. Loading: mirror mode, explicit mode equivalence, validation errors.
2. Equivalence anchor: axle solve == two single-corner solves (the strongest
   correctness check).
3. Symmetry: pure heave produces mirror-symmetric metrics.
4. Steering: rack sweep moves both wheels monotonically.
5. Roll: opposite wheel heights give opposite camber deviations.
6. Jacobian: analytic vs finite-difference on the full axle system.
7. Writer/CLI smoke: CLI sweep output has the expected columns.

Sign conventions (derived and asserted here):

- Each corner's ``roadwheel_angle_deg`` (== toe) is centreline-relative:
  positive means the front of that wheel points toward the vehicle centreline
  (toe-in). In PURE HEAVE the left and right values are therefore EQUAL (mirror
  symmetry). Under STEERING they move with OPPOSITE sign but each is monotonic
  in rack displacement, because steering the axle one way is toe-in for one
  wheel and toe-out for the other in the per-side convention.
- ``total_toe_deg`` = left + right roadwheel angle, so it is the axle total
  toe-in and is an even (symmetric) function of rack displacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetPositionMode,
)
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.sweep_loader import SweepFile, build_sweep_config
from kinematics.main import solve_sweep
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import ResidualComputer, convert_targets_to_absolute
from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

# Tolerance for the equivalence anchor (differences seen are ~1e-6 mm).
POSITION_TOLERANCE = 1e-4
# Finite-difference Jacobian check (mirrors tests/test_jacobians.py).
FD_STEP = 1e-7
FD_TOLERANCE = 1e-6


# ----------------------------------------------------------------------
# Fixtures & helpers
# ----------------------------------------------------------------------


@pytest.fixture
def axle_geometry_file(test_data_dir: Path) -> Path:
    return test_data_dir / "axle_geometry.yaml"


@pytest.fixture
def axle_geometry_explicit_file(test_data_dir: Path) -> Path:
    return test_data_dir / "axle_geometry_explicit.yaml"


@pytest.fixture
def single_geometry_file(test_data_dir: Path) -> Path:
    return test_data_dir / "geometry.yaml"


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
    """Build an axle sweep from equal-length left/right/steer value lists."""
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


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------


class TestLoading:
    """Axle YAML loading, mirroring, and validation."""

    def test_registered_and_type(self, axle_geometry_file: Path) -> None:
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        assert axle.TYPE_KEY == "double_wishbone_axle"
        assert set(axle.corners) == {Side.LEFT, Side.RIGHT}

    def test_mirror_produces_exact_y_negated_right(
        self, axle_geometry_file: Path
    ) -> None:
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        left = axle.corners[Side.LEFT].hardpoints
        right = axle.corners[Side.RIGHT].hardpoints
        assert set(left) == set(right)
        for pid, left_pos in left.items():
            expected = left_pos.data.copy()
            expected[Axis.Y] = -expected[Axis.Y]
            np.testing.assert_allclose(right[pid].data, expected, atol=1e-12)

    def test_explicit_equivalent_to_mirror(
        self, axle_geometry_file: Path, axle_geometry_explicit_file: Path
    ) -> None:
        mirror = load_geometry(axle_geometry_file)
        explicit = load_geometry(axle_geometry_explicit_file)
        assert isinstance(explicit, DoubleWishboneAxleSuspension)
        for side in (Side.LEFT, Side.RIGHT):
            m_hp = mirror.corners[side].hardpoints  # type: ignore[attr-defined]
            e_hp = explicit.corners[side].hardpoints
            assert set(m_hp) == set(e_hp)
            for pid in m_hp:
                np.testing.assert_allclose(e_hp[pid].data, m_hp[pid].data, atol=1e-12)

    def test_wrong_sign_source_side_rejected(self, test_data_dir: Path) -> None:
        import yaml

        data = yaml.safe_load(open(test_data_dir / "axle_geometry.yaml"))
        data.pop("type")
        # Declare the +Y (left) points as the right side.
        data["hardpoints"]["side"] = "right"
        with pytest.raises(ValueError, match="requires AXLE_OUTBOARD Y"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_explicit_missing_side_rejected(self, test_data_dir: Path) -> None:
        import yaml

        data = yaml.safe_load(open(test_data_dir / "axle_geometry_explicit.yaml"))
        data.pop("type")
        del data["hardpoints"]["right"]
        with pytest.raises(ValueError, match="both 'left' and 'right'"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_explicit_same_sign_sides_rejected(self, test_data_dir: Path) -> None:
        import yaml

        data = yaml.safe_load(open(test_data_dir / "axle_geometry_explicit.yaml"))
        data.pop("type")
        # Make the 'right' block a copy of the 'left' (+Y) block.
        data["hardpoints"]["right"] = data["hardpoints"]["left"]
        with pytest.raises(ValueError, match="requires AXLE_OUTBOARD Y"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_side_on_single_corner_target_rejected(
        self, single_geometry_file: Path
    ) -> None:
        single = load_geometry(single_geometry_file)
        spec = SweepFile.model_validate(
            {
                "version": 1,
                "steps": 2,
                "targets": [
                    {
                        "point": "WHEEL_CENTER",
                        "side": "left",
                        "direction": {"axis": "Z"},
                        "start": -5,
                        "stop": 5,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="does not accept a side"):
            build_sweep_config(spec, single)

    def test_side_without_suspension_rejected(self) -> None:
        spec = SweepFile.model_validate(
            {
                "version": 1,
                "steps": 2,
                "targets": [
                    {
                        "point": "WHEEL_CENTER",
                        "side": "left",
                        "direction": {"axis": "Z"},
                        "start": -5,
                        "stop": 5,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="requires a suspension context"):
            build_sweep_config(spec, None)


# ----------------------------------------------------------------------
# 2. Equivalence anchor
# ----------------------------------------------------------------------


class TestEquivalence:
    """Axle solve equivalence to independent single-corner solves."""

    def test_axle_matches_two_single_corner_solves(
        self, single_geometry_file: Path, axle_geometry_file: Path
    ) -> None:
        """
        The axle solve equals two independent single-corner solves.

        The left corner matches a single-corner solve at steer +t; the right
        corner matches the y-mirror of a single-corner solve at steer -t (the
        rigid rack carries the same physical rack translation to both sides, so
        the mirror-frame steer sign flips).
        """
        single = load_geometry(single_geometry_file)
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)

        heave, steer = 25.0, 8.0

        def single_solve(steer_value: float):
            cfg = SweepConfig(
                [
                    [_rel(PointID.WHEEL_CENTER, Axis.Z, heave)],
                    [_rel(PointID.TRACKROD_INBOARD, Axis.Y, steer_value)],
                ]
            )
            states, _ = solve_sweep(single, cfg)
            return states[0]

        left_ref = single_solve(steer)
        right_ref = single_solve(-steer)

        axle_states, stats = solve_sweep(axle, _axle_sweep([heave], [heave], [steer]))
        assert stats[0].converged
        axle_state = axle_states[0]

        for pid in single.OUTPUT_POINTS:
            # Left corner: direct match.
            np.testing.assert_allclose(
                axle_state.positions[PointRef(Side.LEFT, pid)].data,
                left_ref.positions[pid].data,
                atol=POSITION_TOLERANCE,
                err_msg=f"left mismatch at {pid.name}",
            )
            # Right corner: y-mirror of the -steer single-corner solve.
            mirrored = right_ref.positions[pid].data.copy()
            mirrored[Axis.Y] = -mirrored[Axis.Y]
            np.testing.assert_allclose(
                axle_state.positions[PointRef(Side.RIGHT, pid)].data,
                mirrored,
                atol=POSITION_TOLERANCE,
                err_msg=f"right (mirror) mismatch at {pid.name}",
            )


# ----------------------------------------------------------------------
# 3. Symmetry (pure heave)
# ----------------------------------------------------------------------


class TestSymmetry:
    """Mirror symmetry of axle metrics under pure heave."""

    def test_pure_heave_is_mirror_symmetric(self, axle_geometry_file: Path) -> None:
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)

        heave = [10.0, 20.0]
        states, stats = solve_sweep(axle, _axle_sweep(heave, heave, [0.0, 0.0]))
        assert all(s.converged for s in stats)

        for state in states:
            row = axle.compute_state_metrics(state)

            # Camber equal on both sides.
            assert row["left_camber_deg"] == pytest.approx(
                row["right_camber_deg"], abs=1e-4
            )
            # Roadwheel angle EQUAL (same sign) in pure heave: the per-side toe
            # convention is centreline-relative, and the two corners are exact
            # mirror images, so the values coincide.
            assert row["left_roadwheel_angle_deg"] == pytest.approx(
                row["right_roadwheel_angle_deg"], abs=1e-4
            )
            # Rack stays at its design Y.
            assert row["rack_displacement_mm"] == pytest.approx(0.0, abs=1e-3)
            # Roll centre lies on the vehicle centreline (Y ~ 0).
            assert row["roll_center_y_mm"] is not None
            assert row["roll_center_y_mm"] == pytest.approx(0.0, abs=1e-3)
            # Track change is symmetric: both contact patches move equally in |Y|.
            left_cp_y = float(
                state.positions[PointRef(Side.LEFT, PointID.CONTACT_PATCH_CENTER)][
                    Axis.Y
                ]
            )
            right_cp_y = float(
                state.positions[PointRef(Side.RIGHT, PointID.CONTACT_PATCH_CENTER)][
                    Axis.Y
                ]
            )
            assert left_cp_y == pytest.approx(-right_cp_y, abs=1e-3)


# ----------------------------------------------------------------------
# 4. Steering
# ----------------------------------------------------------------------


class TestSteering:
    """Steering (rack) sweep behaviour."""

    def test_rack_sweep_moves_wheels_monotonically(
        self, axle_geometry_file: Path
    ) -> None:
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)

        rack = list(np.linspace(-15.0, 15.0, 7))
        zeros = [0.0] * len(rack)
        states, stats = solve_sweep(axle, _axle_sweep(zeros, zeros, rack))
        assert all(s.converged for s in stats)

        rows = [axle.compute_state_metrics(st) for st in states]
        left = [r["left_roadwheel_angle_deg"] for r in rows]
        right = [r["right_roadwheel_angle_deg"] for r in rows]
        total = [r["total_toe_deg"] for r in rows]

        # Both are well-defined at every step.
        assert all(v is not None for v in left + right + total)

        left_diffs = np.diff(left)
        right_diffs = np.diff(right)
        # Each wheel is strictly monotonic in rack displacement, and the two
        # move with OPPOSITE sign (documented per-side toe convention).
        assert np.all(left_diffs < 0), "left roadwheel angle not monotonic"
        assert np.all(right_diffs > 0), "right roadwheel angle not monotonic"

        # Total toe is a smooth, even function of rack (symmetric about centre).
        centre = len(rack) // 2
        assert total[centre] == pytest.approx(0.0, abs=1e-3)
        assert total[0] == pytest.approx(total[-1], abs=1e-3)


# ----------------------------------------------------------------------
# 5. Roll
# ----------------------------------------------------------------------


class TestRoll:
    """Roll (opposite wheel heights) behaviour."""

    def test_opposite_wheel_heights_give_opposite_camber(
        self, axle_geometry_file: Path
    ) -> None:
        axle = load_geometry(axle_geometry_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)

        # Design reference (zero roll).
        ref_states, _ = solve_sweep(axle, _axle_sweep([0.0], [0.0], [0.0]))
        ref = axle.compute_state_metrics(ref_states[0])
        left0 = ref["left_camber_deg"]
        right0 = ref["right_camber_deg"]
        assert left0 is not None and right0 is not None

        # Roll: left up, right down.
        roll = [8.0, 16.0]
        states, stats = solve_sweep(
            axle, _axle_sweep(roll, [-v for v in roll], [0.0, 0.0])
        )
        assert all(s.converged for s in stats)

        for state in states:
            row = axle.compute_state_metrics(state)
            left_camber = row["left_camber_deg"]
            right_camber = row["right_camber_deg"]
            assert left_camber is not None and right_camber is not None
            d_left = left_camber - left0
            d_right = right_camber - right0
            # Camber deviations from design have opposite sign in roll.
            assert d_left * d_right < 0, (
                f"camber deviations not opposite: {d_left} vs {d_right}"
            )
            # Roll-centre metrics remain finite and well-defined.
            rc_y = row["roll_center_y_mm"]
            rc_z = row["roll_center_z_mm"]
            assert rc_y is not None and rc_z is not None
            assert np.isfinite(rc_y)
            assert np.isfinite(rc_z)
            # Total toe is well-defined.
            assert row["total_toe_deg"] is not None
            assert np.isfinite(row["total_toe_deg"])


# ----------------------------------------------------------------------
# 6. Jacobian
# ----------------------------------------------------------------------


class TestJacobian:
    """Analytic vs finite-difference Jacobian for the axle system."""

    def test_analytic_matches_finite_difference(self, axle_geometry_file: Path) -> None:
        axle = load_geometry(axle_geometry_file)
        initial_state = axle.initial_state()
        derived_manager = DerivedPointsManager(axle.derived_spec())
        state = initial_state.copy()

        targets = convert_targets_to_absolute(
            [
                _rel(PointRef(Side.LEFT, PointID.WHEEL_CENTER), Axis.Z, 5.0),
                _rel(PointRef(Side.RIGHT, PointID.WHEEL_CENTER), Axis.Z, -5.0),
                _rel(PointRef(Side.LEFT, PointID.TRACKROD_INBOARD), Axis.Y, 3.0),
            ],
            initial_state,
        )

        rc = ResidualComputer(
            constraints=axle.constraints(),
            derived_manager=derived_manager,
            state_buffer=state,
            n_target_variables=len(targets),
        )

        x0 = state.get_free_array()
        analytic = rc.compute_jacobian(x0, targets)

        n_res, n_vars = analytic.shape
        numerical = np.zeros((n_res, n_vars))
        for j in range(n_vars):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[j] += FD_STEP
            x_minus[j] -= FD_STEP
            r_plus = rc.compute(x_plus, targets)
            r_minus = rc.compute(x_minus, targets)
            numerical[:, j] = (r_plus - r_minus) / (2.0 * FD_STEP)

        np.testing.assert_allclose(analytic, numerical, atol=FD_TOLERANCE)


# ----------------------------------------------------------------------
# 7. Writer / CLI smoke
# ----------------------------------------------------------------------


class TestCliSmoke:
    """CLI sweep output column smoke test for the axle."""

    def test_cli_sweep_writes_axle_columns(
        self,
        tmp_path: Path,
        axle_geometry_file: Path,
        test_data_dir: Path,
    ) -> None:
        import csv

        from kinematics.cli import sweep as cli_sweep

        out = tmp_path / "axle_out.csv"
        cli_sweep(
            geometry=axle_geometry_file,
            sweep=test_data_dir / "axle_sweep.yaml",
            out=out,
            animation_out=None,
        )
        assert out.exists()

        with open(out) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        reader = csv.DictReader(lines)
        rows = list(reader)
        headers = reader.fieldnames or []

        # Position columns for both sides.
        assert "LEFT_WHEEL_CENTER_x" in headers
        assert "RIGHT_WHEEL_CENTER_z" in headers
        # Per-side metric columns.
        assert "left_camber_deg" in headers
        assert "right_camber_deg" in headers
        # Axle-level metric columns.
        for col in (
            "roll_center_y_mm",
            "roll_center_z_mm",
            "total_toe_deg",
            "track_mm",
            "rack_displacement_mm",
        ):
            assert col in headers, f"missing axle metric column {col}"

        # Every step converged.
        assert all(r["solver_converged"] == "True" for r in rows)
        assert len(rows) == 5
