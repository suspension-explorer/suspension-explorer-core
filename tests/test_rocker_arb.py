"""
Tests for the pushrod / rocker / torsion-bar / ARB inboard actuation (Phase B).

Coverage:
1. ``signed_angle_about_axis`` unit tests (known rotations, sign flips, edges).
2. Load-time validation of the rocker group and the ARB group.
3. Corner-level rocker: bump sweep, length/radius conservation, monotonicity,
   and a hand-verified rocker-angle sign.
4. Axle heave: symmetric rocker angles, zero ARB twist.
5. Axle roll: antisymmetric rocker angles, non-zero ARB twist with a documented
   sign, droplink lengths conserved.
6. Analytic vs finite-difference Jacobian on the rocker+ARB axle.
7. CLI/writer smoke: new columns present and solver converged throughout.
8. Regression: non-rocker geometries produce no rocker/ARB columns.

Sign conventions (see ``metrics/main.py``):

- ``rocker_angle_deg`` is the raw signed angle of ``PUSHROD_INBOARD`` about the
  ROCKER_AXIS_FRONT -> ROCKER_AXIS_REAR direction, multiplied by the corner's
  ``side_sign`` (+1 left, -1 right). The normalisation makes symmetric heave read
  EQUAL on both sides and roll read equal-and-opposite.
- ``arb_arm_angle_deg`` (per side) and ``arb_twist_deg`` (= left - right) use RAW
  signed angles about the single shared ARB axis (ARB_AXIS_A -> ARB_AXIS_B), with
  no side normalisation, since both arms share one physical axis.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetPositionMode,
)
from kinematics.core.vector_utils.geometric import (
    rotate_point_about_axis,
    signed_angle_about_axis,
)
from kinematics.io.geometry_loader import load_geometry
from kinematics.main import solve_sweep
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import ResidualComputer, convert_targets_to_absolute
from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension

TEST_TOLERANCE = 1e-4
FD_STEP = 1e-7
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
def corner_rocker_file(test_data_dir: Path) -> Path:
    return test_data_dir / "corner_rocker_geometry.yaml"


@pytest.fixture
def axle_rocker_file(test_data_dir: Path) -> Path:
    return test_data_dir / "axle_geometry_rocker.yaml"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    data.pop("type", None)
    return data


# ----------------------------------------------------------------------
# 1. signed_angle_about_axis
# ----------------------------------------------------------------------


class TestSignedAngleAboutAxis:
    """Direct unit tests of the signed-angle primitive."""

    Z = Direction3([0.0, 0.0, 1.0])
    ORIGIN = Point3([0.0, 0.0, 0.0])

    def test_identity_is_zero(self) -> None:
        p = Point3([1.0, 0.0, 0.0])
        assert signed_angle_about_axis(p, p, self.ORIGIN, self.Z) == pytest.approx(0.0)

    def test_plus_ninety_about_z(self) -> None:
        # +90 deg about +Z carries +X onto +Y (right-hand rule).
        p0 = Point3([1.0, 0.0, 0.0])
        p1 = Point3([0.0, 1.0, 0.0])
        assert signed_angle_about_axis(p0, p1, self.ORIGIN, self.Z) == pytest.approx(
            math.pi / 2
        )

    def test_minus_ninety_about_z(self) -> None:
        p0 = Point3([1.0, 0.0, 0.0])
        p1 = Point3([0.0, -1.0, 0.0])
        assert signed_angle_about_axis(p0, p1, self.ORIGIN, self.Z) == pytest.approx(
            -math.pi / 2
        )

    def test_one_eighty_about_z(self) -> None:
        p0 = Point3([1.0, 0.0, 0.0])
        p1 = Point3([-1.0, 0.0, 0.0])
        assert abs(
            signed_angle_about_axis(p0, p1, self.ORIGIN, self.Z)
        ) == pytest.approx(math.pi)

    def test_axis_reversal_flips_sign(self) -> None:
        p0 = Point3([1.0, 0.0, 0.0])
        p1 = Point3([0.0, 1.0, 0.0])
        plus = signed_angle_about_axis(p0, p1, self.ORIGIN, self.Z)
        minus = signed_angle_about_axis(p0, p1, self.ORIGIN, -self.Z)
        assert minus == pytest.approx(-plus)

    def test_axial_offset_of_axis_point_is_irrelevant(self) -> None:
        # Only the radius component perpendicular to the axis matters, so moving
        # the axis point along the axis must not change the result.
        p0 = Point3([1.0, 0.0, 5.0])
        p1 = Point3([0.0, 1.0, 5.0])
        a = signed_angle_about_axis(p0, p1, self.ORIGIN, self.Z)
        b = signed_angle_about_axis(p0, p1, Point3([0.0, 0.0, -3.0]), self.Z)
        assert a == pytest.approx(b)
        assert a == pytest.approx(math.pi / 2)

    def test_oblique_axis_matches_rodrigues(self) -> None:
        # Rotate a point by a known angle about an oblique axis with Rodrigues,
        # then recover that same angle.
        axis = Direction3([1.0, 1.0, 1.0])
        axis_point = Point3([0.2, -0.3, 0.5])
        p0 = Point3([1.0, 0.5, -0.4])
        angle = 0.7  # radians
        p1 = rotate_point_about_axis(p0, axis_point, axis, angle)
        recovered = signed_angle_about_axis(p0, p1, axis_point, axis)
        assert recovered == pytest.approx(angle, abs=1e-9)


# ----------------------------------------------------------------------
# 2. Load-time validation
# ----------------------------------------------------------------------


class TestValidation:
    """Rocker- and ARB-group load-time validation."""

    def test_rocker_axis_unequal_y_rejected(self, corner_rocker_file: Path) -> None:
        data = _load_yaml(corner_rocker_file)
        data["hardpoints"]["rocker_axis_rear"]["y"] = 341.0  # break XZ parallelism
        with pytest.raises(ValueError, match="parallel to the XZ plane"):
            DoubleWishboneSuspension.from_yaml_data(data)

    def test_partial_rocker_group_rejected(self, corner_rocker_file: Path) -> None:
        data = _load_yaml(corner_rocker_file)
        del data["hardpoints"]["rocker_axis_rear"]
        with pytest.raises(ValueError, match="pushrod/rocker group"):
            DoubleWishboneSuspension.from_yaml_data(data)

    def test_rocker_droplink_without_group_rejected(self, test_data_dir: Path) -> None:
        # geometry.yaml has no rocker group; adding only ROCKER_DROPLINK is invalid.
        data = _load_yaml(test_data_dir / "geometry.yaml")
        data["hardpoints"]["rocker_droplink"] = {"x": 0, "y": 340, "z": 400}
        with pytest.raises(ValueError, match="ROCKER_DROPLINK requires"):
            DoubleWishboneSuspension.from_yaml_data(data)

    def test_arb_missing_center_rejected(self, axle_rocker_file: Path) -> None:
        data = _load_yaml(axle_rocker_file)
        del data["hardpoints"]["center"]
        with pytest.raises(ValueError, match="center 'arb_axis_a' and 'arb_axis_b'"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_arb_missing_droplink_rejected(self, axle_rocker_file: Path) -> None:
        data = _load_yaml(axle_rocker_file)
        del data["hardpoints"]["points"]["arb_droplink"]
        with pytest.raises(ValueError, match="'arb_droplink' must be given on both"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_explicit_partial_arb_droplink_rejected(self, test_data_dir: Path) -> None:
        # Explicit mode with the rocker/ARB group only on the left corner.
        data = _load_yaml(test_data_dir / "axle_geometry_rocker.yaml")
        left = data["hardpoints"]["points"]
        right = {k: {**v, "y": -v["y"]} for k, v in copy.deepcopy(left).items()}
        # Drop the right ARB droplink to make the ARB group partial.
        right.pop("arb_droplink")
        data["hardpoints"] = {
            "left": left,
            "right": right,
            "center": data["hardpoints"]["center"],
        }
        with pytest.raises(ValueError, match="ARB"):
            DoubleWishboneAxleSuspension.from_yaml_data(data)

    def test_single_corner_yaml_with_arb_droplink_rejected(
        self, corner_rocker_file: Path
    ) -> None:
        data = _load_yaml(corner_rocker_file)
        data["hardpoints"]["arb_droplink"] = {"x": 0, "y": 150, "z": 490}
        with pytest.raises(ValueError, match="Unknown hardpoint key"):
            DoubleWishboneSuspension.from_yaml_data(data)


# ----------------------------------------------------------------------
# 3. Corner-level rocker
# ----------------------------------------------------------------------


class TestCornerRocker:
    """Single-corner pushrod/rocker behavior."""

    def _bump_sweep(self, corner, heave: list[float]) -> list:
        # Pin the steering DOF (trackrod inboard Y) so the rocker angle is a clean
        # function of wheel travel, exactly as the base model requires a steering
        # target to be well posed.
        cfg = SweepConfig(
            [
                [_rel(PointID.WHEEL_CENTER, Axis.Z, v) for v in heave],
                [_rel(PointID.TRACKROD_INBOARD, Axis.Y, 0.0) for _ in heave],
            ]
        )
        states, stats = solve_sweep(corner, cfg)
        assert all(s.converged for s in stats)
        return states

    def test_lengths_conserved_and_monotonic(self, corner_rocker_file: Path) -> None:
        corner = load_geometry(corner_rocker_file)
        assert corner.has_rocker
        design = corner.initial_state()

        pushrod0 = (
            design.positions[PointID.PUSHROD_OUTBOARD]
            - design.positions[PointID.PUSHROD_INBOARD]
        ).norm()
        radius_front0 = (
            design.positions[PointID.PUSHROD_INBOARD]
            - design.positions[PointID.ROCKER_AXIS_FRONT]
        ).norm()
        radius_rear0 = (
            design.positions[PointID.PUSHROD_INBOARD]
            - design.positions[PointID.ROCKER_AXIS_REAR]
        ).norm()

        heave = [-20.0, -10.0, 0.0, 10.0, 20.0]
        states = self._bump_sweep(corner, heave)

        angles = []
        for st in states:
            pushrod = (
                st.positions[PointID.PUSHROD_OUTBOARD]
                - st.positions[PointID.PUSHROD_INBOARD]
            ).norm()
            rf = (
                st.positions[PointID.PUSHROD_INBOARD]
                - st.positions[PointID.ROCKER_AXIS_FRONT]
            ).norm()
            rr = (
                st.positions[PointID.PUSHROD_INBOARD]
                - st.positions[PointID.ROCKER_AXIS_REAR]
            ).norm()
            assert pushrod == pytest.approx(pushrod0, abs=TEST_TOLERANCE)
            assert rf == pytest.approx(radius_front0, abs=TEST_TOLERANCE)
            assert rr == pytest.approx(radius_rear0, abs=TEST_TOLERANCE)

            row = corner.compute_state_metrics(st)
            # Torsion-bar twist mirrors the rocker angle exactly.
            assert row["torsion_bar_twist_deg"] == pytest.approx(
                row["rocker_angle_deg"], abs=1e-9
            )
            angles.append(row["rocker_angle_deg"])

        # Zero at design (index 2) and strictly monotonic in wheel travel.
        # The tolerance allows for solver acceptance noise: a converged
        # solve can sit up to the residual tolerance away from the exact
        # design position, which maps to microdegrees of rocker angle.
        assert angles[2] == pytest.approx(0.0, abs=1e-5)
        diffs = np.diff(angles)
        assert np.all(diffs < 0.0), f"rocker angle not monotonic: {angles}"

    def test_hand_verified_bump_sign(self, corner_rocker_file: Path) -> None:
        """
        A positive (+Z) wheel bump gives a NEGATIVE raw rocker angle.

        Right-hand-rule argument for this geometry (all at x=0 for the rocker):

        - The rocker axis runs ROCKER_AXIS_FRONT (x=+100) -> ROCKER_AXIS_REAR
          (x=-100), so the authored axis direction is -X.
        - PUSHROD_INBOARD sits at the TOP of the rocker circle, directly above the
          pivot: its design radius vector is r0 = (0, 0, +50).
        - Under a +Z wheel bump the upright rises, the pushrod pushes the rocker,
          and PUSHROD_INBOARD swings toward -Y (verified: dy < 0), so
          r1 ~= (0, -m, +50) with m > 0.
        - r0 x r1 = (0,0,50) x (0,-m,50) = (+50 m, 0, 0), i.e. +X.
        - Dotting with the axis direction -X gives a NEGATIVE value, so the signed
          angle is negative. This is a LEFT corner (side_sign = +1), so the
          reported rocker angle keeps that negative sign.
        """
        corner = load_geometry(corner_rocker_file)
        states = self._bump_sweep(corner, [0.0, 20.0])

        # Confirm the premise: PUSHROD_INBOARD swings toward -Y under bump.
        dy = float(
            states[1].positions[PointID.PUSHROD_INBOARD][Axis.Y]
            - states[0].positions[PointID.PUSHROD_INBOARD][Axis.Y]
        )
        assert dy < 0.0

        bump_angle = corner.compute_state_metrics(states[1])["rocker_angle_deg"]
        assert bump_angle is not None
        assert bump_angle < 0.0


# ----------------------------------------------------------------------
# 4. Axle heave (symmetric)
# ----------------------------------------------------------------------


class TestAxleHeave:
    """Pure heave: mirror-symmetric rocker angles and zero ARB twist."""

    def test_symmetric_heave(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        assert axle.has_arb

        heave = [10.0, 20.0]
        states, stats = solve_sweep(axle, _axle_sweep(heave, heave, [0.0, 0.0]))
        assert all(s.converged for s in stats)

        design = axle.initial_state()
        for st in states:
            row = axle.compute_state_metrics(st)
            # Wheel up (+Z) must drive the droplink DOWN (-Z) on both sides:
            # the rocker's droplink lever sits on the opposite side of the pivot
            # axis from the pushrod lever, and the ARB lives below the rocker.
            for side in (Side.LEFT, Side.RIGHT):
                for pid in (PointID.ROCKER_DROPLINK, PointID.ARB_DROPLINK):
                    dz = float(st.positions[PointRef(side, pid)][Axis.Z]) - float(
                        design.positions[PointRef(side, pid)][Axis.Z]
                    )
                    assert dz < 0.0, (
                        f"{side.name} {pid.name} should move down in bump, got {dz}"
                    )
            # Rocker angles equal on both sides (side-normalised).
            assert row["left_rocker_angle_deg"] == pytest.approx(
                row["right_rocker_angle_deg"], abs=TEST_TOLERANCE
            )
            # Raw ARB arm angles equal on both sides.
            assert row["left_arb_arm_angle_deg"] == pytest.approx(
                row["right_arb_arm_angle_deg"], abs=TEST_TOLERANCE
            )
            # No relative twist of the ARB in pure heave.
            assert row["arb_twist_deg"] == pytest.approx(0.0, abs=TEST_TOLERANCE)


# ----------------------------------------------------------------------
# 5. Axle roll (antisymmetric)
# ----------------------------------------------------------------------


class TestAxleRoll:
    """Roll: antisymmetric rocker angles, non-zero ARB twist, conserved links."""

    def test_roll(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
        assert isinstance(axle, DoubleWishboneAxleSuspension)

        design = axle.initial_state()
        droplink0 = {
            side: (
                design.positions[PointRef(side, PointID.ROCKER_DROPLINK)]
                - design.positions[PointRef(side, PointID.ARB_DROPLINK)]
            ).norm()
            for side in (Side.LEFT, Side.RIGHT)
        }

        # Roll: left wheel up, right wheel down.
        roll = [8.0, 16.0]
        states, stats = solve_sweep(
            axle, _axle_sweep(roll, [-v for v in roll], [0.0, 0.0])
        )
        assert all(s.converged for s in stats)

        for st in states:
            row = axle.compute_state_metrics(st)
            left = row["left_rocker_angle_deg"]
            right = row["right_rocker_angle_deg"]
            twist = row["arb_twist_deg"]
            assert left is not None and right is not None and twist is not None

            # Side-normalised rocker angles are equal-and-opposite. Exact equality
            # of magnitude does not hold (the mechanism responds nonlinearly to
            # +Z vs -Z travel), so the check is on sign plus approximate magnitude.
            assert left * right < 0.0, f"rocker angles not opposite: {left}, {right}"
            assert abs(left) == pytest.approx(abs(right), rel=0.1)

            # Left-wheel-up produces NEGATIVE ARB twist for this fixture.
            # Derivation: the ARB axis direction is ARB_AXIS_A(+Y) ->
            # ARB_AXIS_B(-Y), i.e. -Y; each arm end sits on an 80 mm lever in
            # +X off the bar. An arm end moving down (-Z) at radius +X is a
            # NEGATIVE rotation about -Y (velocity of +ω about -Y at radius +X
            # is (0, 0, +ωr)). Left wheel up pushes the left droplink (and arm)
            # down => left_arm < 0; right wheel down lifts the right arm =>
            # right_arm > 0; arb_twist = left_arm - right_arm < 0.
            assert twist < 0.0

            # Droplink lengths conserved on both sides.
            for side in (Side.LEFT, Side.RIGHT):
                length = (
                    st.positions[PointRef(side, PointID.ROCKER_DROPLINK)]
                    - st.positions[PointRef(side, PointID.ARB_DROPLINK)]
                ).norm()
                assert length == pytest.approx(droplink0[side], abs=TEST_TOLERANCE)


# ----------------------------------------------------------------------
# 6. Jacobian
# ----------------------------------------------------------------------


class TestJacobian:
    """Analytic vs finite-difference Jacobian on the rocker+ARB axle."""

    def test_analytic_matches_finite_difference(self, axle_rocker_file: Path) -> None:
        axle = load_geometry(axle_rocker_file)
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
            numerical[:, j] = (
                rc.compute(x_plus, targets) - rc.compute(x_minus, targets)
            ) / (2.0 * FD_STEP)

        np.testing.assert_allclose(analytic, numerical, atol=FD_TOLERANCE)


# ----------------------------------------------------------------------
# 7. CLI / writer smoke
# ----------------------------------------------------------------------


class TestCliSmoke:
    """CLI sweep output for the rocker/ARB axle."""

    def test_cli_writes_rocker_arb_columns(
        self, tmp_path: Path, axle_rocker_file: Path, test_data_dir: Path
    ) -> None:
        import csv

        from kinematics.cli import sweep as cli_sweep

        out = tmp_path / "rocker_axle_out.csv"
        cli_sweep(
            geometry=axle_rocker_file,
            sweep=test_data_dir / "axle_rocker_sweep.yaml",
            out=out,
            animation_out=None,
        )
        assert out.exists()

        with open(out) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        reader = csv.DictReader(lines)
        rows = list(reader)
        headers = reader.fieldnames or []

        # Position columns for the new points, both sides.
        for col in (
            "LEFT_PUSHROD_INBOARD_x",
            "LEFT_PUSHROD_OUTBOARD_z",
            "LEFT_ROCKER_DROPLINK_y",
            "RIGHT_ARB_DROPLINK_x",
        ):
            assert col in headers, f"missing position column {col}"

        # New metric columns.
        for col in (
            "left_rocker_angle_deg",
            "right_rocker_angle_deg",
            "left_torsion_bar_twist_deg",
            "left_arb_arm_angle_deg",
            "right_arb_arm_angle_deg",
            "arb_twist_deg",
        ):
            assert col in headers, f"missing metric column {col}"

        assert all(r["solver_converged"] == "True" for r in rows)
        # Row count follows the steps value in axle_rocker_sweep.yaml.
        assert len(rows) == 16


# ----------------------------------------------------------------------
# 8. Regression: no rocker/ARB columns without the group
# ----------------------------------------------------------------------


class TestRegressionNoRocker:
    """Non-rocker geometries emit none of the new columns."""

    def test_plain_axle_has_no_rocker_arb_columns(self, test_data_dir: Path) -> None:
        axle = load_geometry(test_data_dir / "axle_geometry.yaml")
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        assert not axle.has_arb
        states, _ = solve_sweep(axle, _axle_sweep([0.0], [0.0], [0.0]))
        row = axle.compute_state_metrics(states[0])
        for col in row:
            assert "rocker" not in col
            assert "torsion" not in col
            assert "arb" not in col

    def test_plain_corner_has_no_rocker_columns(self, test_data_dir: Path) -> None:
        corner = load_geometry(test_data_dir / "geometry.yaml")
        assert not corner.has_rocker
        cfg = SweepConfig([[_rel(PointID.WHEEL_CENTER, Axis.Z, 0.0)]])
        states, _ = solve_sweep(corner, cfg)
        row = corner.compute_state_metrics(states[0])
        assert "rocker_angle_deg" not in row
        assert "torsion_bar_twist_deg" not in row
