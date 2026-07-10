"""
Tests for the per-state (non-derivative) suspension metrics.

Covers corner travel metrics (wheel travel, half-track change, recession),
the installed damper length, the side-view anti-geometry metrics (anti-dive,
anti-lift, anti-squat) and the axle-level modal metrics (heave, roll,
ride-height change, Ackermann).
"""

from __future__ import annotations

from math import atan2, degrees
from types import SimpleNamespace

import numpy as np
import pytest

from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import PointTarget, PointTargetAxis, SweepConfig
from kinematics.io import load_geometry
from kinematics.main import solve_sweep
from kinematics.metrics.anti_geometry import (
    calculate_anti_dive_pct,
    calculate_anti_lift_pct,
    calculate_anti_squat_pct,
    calculate_svsa_angle,
)
from kinematics.metrics.main import (
    compute_metrics_for_axle_state,
    compute_metrics_for_state_from_suspension,
)
from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

# --------------------------------------------------------------------------
# Sweep helpers (mirroring tests/test_sensitivity.py)
# --------------------------------------------------------------------------


def _bump_target(point_id, value: float) -> PointTarget:
    return PointTarget(
        point_id=point_id,
        direction=PointTargetAxis(axis=Axis.Z),
        value=value,
        mode=TargetPositionMode.ABSOLUTE,
    )


def _rack_target(point_id, value: float) -> PointTarget:
    return PointTarget(
        point_id=point_id,
        direction=PointTargetAxis(axis=Axis.Y),
        value=value,
        mode=TargetPositionMode.ABSOLUTE,
    )


def _solve_states(suspension, target_lists):
    """
    Solve one state per entry of target_lists; each entry is the full list of
    absolute targets for that step.
    """
    sweeps = [
        [target_lists[step][dim] for step in range(len(target_lists))]
        for dim in range(len(target_lists[0]))
    ]
    states, _stats = solve_sweep(suspension, SweepConfig(sweeps))
    return states


# --------------------------------------------------------------------------
# Corner travel metrics
# --------------------------------------------------------------------------


class TestCornerTravelMetrics:
    """Corner travel, half-track change, and recession against design."""

    def test_wheel_travel_matches_commanded_bump(
        self, double_wishbone_geometry_file
    ) -> None:
        corner = load_geometry(double_wishbone_geometry_file)
        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])

        bump = 15.0
        droop = -15.0

        def targets(z: float):
            return [
                _bump_target(PointID.WHEEL_CENTER, z),
                _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
            ]

        bump_state, droop_state = _solve_states(
            corner, [targets(design_z + bump), targets(design_z + droop)]
        )

        bump_metrics = compute_metrics_for_state_from_suspension(bump_state, corner)
        droop_metrics = compute_metrics_for_state_from_suspension(droop_state, corner)

        # Wheel travel equals the commanded wheel-centre displacement, and bump
        # (wheel up) is positive.
        bump_travel = bump_metrics["wheel_travel_mm"]
        droop_travel = droop_metrics["wheel_travel_mm"]
        assert bump_travel is not None and droop_travel is not None
        assert bump_travel == pytest.approx(bump, abs=1e-4)
        assert droop_travel == pytest.approx(droop, abs=1e-4)
        assert bump_travel > 0
        assert droop_travel < 0

    def test_half_track_and_recession_are_zero_at_design(
        self, double_wishbone_geometry_file
    ) -> None:
        corner = load_geometry(double_wishbone_geometry_file)
        metrics = compute_metrics_for_state_from_suspension(
            corner.initial_state(), corner
        )
        assert metrics["half_track_change_mm"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["wheel_recession_mm"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["wheel_travel_mm"] == pytest.approx(0.0, abs=1e-6)

    def test_half_track_and_recession_finite_under_bump(
        self, double_wishbone_geometry_file
    ) -> None:
        corner = load_geometry(double_wishbone_geometry_file)
        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])

        state = _solve_states(
            corner,
            [
                [
                    _bump_target(PointID.WHEEL_CENTER, design_z + 20.0),
                    _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
                ]
            ],
        )[0]
        metrics = compute_metrics_for_state_from_suspension(state, corner)

        # Both are defined and match a direct computation from the state.
        design_cp = initial.positions[PointID.CONTACT_PATCH_CENTER]
        cp = state.positions[PointID.CONTACT_PATCH_CENTER]
        expected_half_track = abs(float(cp[Axis.Y])) - abs(float(design_cp[Axis.Y]))
        expected_recession = -(float(cp[Axis.X]) - float(design_cp[Axis.X]))
        assert metrics["half_track_change_mm"] == pytest.approx(
            expected_half_track, abs=1e-6
        )
        assert metrics["wheel_recession_mm"] == pytest.approx(
            expected_recession, abs=1e-6
        )


# --------------------------------------------------------------------------
# Damper length
# --------------------------------------------------------------------------


class TestDamperLength:
    """Installed damper length equals the strut mount distance."""

    def test_damper_length_matches_strut_distance_and_shrinks_in_bump(
        self, test_data_dir
    ) -> None:
        corner = load_geometry(test_data_dir / "corner_strut_geometry.yaml")
        assert corner.has_strut

        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])

        design_metrics = compute_metrics_for_state_from_suspension(initial, corner)

        top = initial.positions[PointID.STRUT_TOP]
        bottom = initial.positions[PointID.STRUT_BOTTOM]
        expected = float((top - bottom).norm())
        assert design_metrics["damper_length_mm"] == pytest.approx(expected, abs=1e-6)

        bump_state = _solve_states(
            corner,
            [
                [
                    _bump_target(PointID.WHEEL_CENTER, design_z + 25.0),
                    _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
                ]
            ],
        )[0]
        bump_metrics = compute_metrics_for_state_from_suspension(bump_state, corner)

        # The strut foot rides up with the lower wishbone in bump, so the
        # installed damper length shrinks.
        bump_length = bump_metrics["damper_length_mm"]
        design_length = design_metrics["damper_length_mm"]
        assert bump_length is not None and design_length is not None
        assert bump_length < design_length

    def test_damper_length_none_without_strut(
        self, double_wishbone_geometry_file
    ) -> None:
        corner = load_geometry(double_wishbone_geometry_file)
        assert not corner.has_strut
        metrics = compute_metrics_for_state_from_suspension(
            corner.initial_state(), corner
        )
        assert metrics["damper_length_mm"] is None


# --------------------------------------------------------------------------
# Anti-geometry pure functions (synthetic contexts)
# --------------------------------------------------------------------------


def _stub_ctx(
    *,
    svic: Point3 | None,
    cp: Point3,
    wc: Point3,
    cg: Point3,
    wheelbase: float,
    axle_position=None,
    front_brake_bias=None,
    driven_axle=None,
):
    """
    Build a minimal duck-typed MetricContext for the anti-geometry functions.

    The anti functions only touch config (three fields), side_view_ic,
    contact_patch_center, wheel_center, cg_position and wheelbase, so a
    SimpleNamespace exercises them directly on synthetic geometry.
    """
    config = SimpleNamespace(
        axle_position=axle_position,
        front_brake_bias=front_brake_bias,
        driven_axle=driven_axle,
    )
    return SimpleNamespace(
        config=config,
        side_view_ic=svic,
        contact_patch_center=cp,
        wheel_center=wc,
        cg_position=cg,
        wheelbase=wheelbase,
    )


class TestAntiGeometry:
    """Side-view anti-dive/lift/squat and SVSA-angle sign conventions."""

    CP = Point3([0.0, 800.0, 0.0])
    WC = Point3([0.0, 800.0, 300.0])
    CG = Point3([1250.0, 0.0, 450.0])
    L = 2500.0

    def test_svsa_angle_sign_and_vertical(self) -> None:
        # SVIC ahead (+X) and above: line rises toward the front -> positive.
        ctx = _stub_ctx(
            svic=Point3([500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
        )
        angle = calculate_svsa_angle(ctx)
        expected = degrees(atan2(300.0, 500.0))
        assert angle is not None and angle == pytest.approx(expected)

        # Vertical side-view line -> None.
        vertical = _stub_ctx(
            svic=Point3([0.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
        )
        assert calculate_svsa_angle(vertical) is None

    def test_anti_metrics_none_when_config_unset(self) -> None:
        ctx = _stub_ctx(
            svic=Point3([-500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
        )
        assert calculate_anti_dive_pct(ctx) is None
        assert calculate_anti_lift_pct(ctx) is None
        assert calculate_anti_squat_pct(ctx) is None

    def test_anti_dive_sign_flips_with_svic_position(self) -> None:
        # Front axle, 60% front brake bias. SVIC above and BEHIND (-X) the
        # front contact patch is the classic anti-dive geometry -> positive.
        behind = _stub_ctx(
            svic=Point3([-500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="front",
            front_brake_bias=0.6,
        )
        ahead = _stub_ctx(
            svic=Point3([500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="front",
            front_brake_bias=0.6,
        )
        behind_pct = calculate_anti_dive_pct(behind)
        ahead_pct = calculate_anti_dive_pct(ahead)
        assert behind_pct is not None and behind_pct > 0
        assert ahead_pct is not None and ahead_pct < 0
        # A rear-only metric must stay None for a front axle.
        assert calculate_anti_lift_pct(behind) is None

    def test_anti_lift_sign_flips_with_svic_position(self) -> None:
        # Rear axle, 60% front bias (40% rear). SVIC above and AHEAD (+X) of
        # the rear contact patch -> positive anti-lift.
        ahead = _stub_ctx(
            svic=Point3([500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="rear",
            front_brake_bias=0.6,
        )
        behind = _stub_ctx(
            svic=Point3([-500.0, 800.0, 300.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="rear",
            front_brake_bias=0.6,
        )
        ahead_pct = calculate_anti_lift_pct(ahead)
        behind_pct = calculate_anti_lift_pct(behind)
        assert ahead_pct is not None and ahead_pct > 0
        assert behind_pct is not None and behind_pct < 0
        assert calculate_anti_dive_pct(ahead) is None

    def test_anti_squat_requires_matching_driven_axle(self) -> None:
        # Rear axle driven at the rear: uses the wheel-centre -> SVIC line.
        rear = _stub_ctx(
            svic=Point3([500.0, 800.0, 500.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="rear",
            driven_axle="rear",
        )
        squat = calculate_anti_squat_pct(rear)
        assert squat is not None and squat > 0

        # Driven axle does not match this axle -> None.
        mismatch = _stub_ctx(
            svic=Point3([500.0, 800.0, 500.0]),
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="rear",
            driven_axle="front",
        )
        assert calculate_anti_squat_pct(mismatch) is None

    def test_anti_metrics_none_when_svic_undefined(self) -> None:
        ctx = _stub_ctx(
            svic=None,
            cp=self.CP,
            wc=self.WC,
            cg=self.CG,
            wheelbase=self.L,
            axle_position="front",
            front_brake_bias=0.6,
            driven_axle="front",
        )
        assert calculate_anti_dive_pct(ctx) is None
        assert calculate_anti_squat_pct(ctx) is None
        assert calculate_svsa_angle(ctx) is None

    def test_anti_dive_none_when_cg_below_ground(self) -> None:
        # CG at or below the contact-patch plane is non-physical for the anti
        # formula -> None.
        below = _stub_ctx(
            svic=Point3([-500.0, 800.0, 300.0]),
            cp=Point3([0.0, 800.0, 500.0]),
            wc=self.WC,
            cg=Point3([1250.0, 0.0, 450.0]),
            wheelbase=self.L,
            axle_position="front",
            front_brake_bias=0.6,
        )
        assert calculate_anti_dive_pct(below) is None


# --------------------------------------------------------------------------
# Axle-level modal metrics
# --------------------------------------------------------------------------


class TestAxleStateMetrics:
    """Axle heave, roll, ride-height change, and Ackermann."""

    @pytest.fixture
    def axle(self, test_data_dir):
        axle = load_geometry(test_data_dir / "axle_geometry.yaml")
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        return axle

    def _design(self, axle):
        initial = axle.initial_state()
        left_wc = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
        right_wc = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
        rack = PointRef(Side.LEFT, PointID.TRACKROD_INBOARD)
        return (
            float(initial.positions[left_wc][Axis.Z]),
            float(initial.positions[right_wc][Axis.Z]),
            float(initial.positions[rack][Axis.Y]),
            left_wc,
            right_wc,
            rack,
        )

    def test_symmetric_heave(self, axle) -> None:
        left_z0, right_z0, rack_y0, left_wc, right_wc, rack = self._design(axle)
        delta = 12.0

        state = _solve_states(
            axle,
            [
                [
                    _bump_target(left_wc, left_z0 + delta),
                    _bump_target(right_wc, right_z0 + delta),
                    _rack_target(rack, rack_y0),
                ]
            ],
        )[0]
        metrics = compute_metrics_for_axle_state(state, axle, axle.config)

        assert metrics.axle["heave_mm"] == pytest.approx(delta, abs=1e-4)
        assert metrics.axle["roll_deg"] == pytest.approx(0.0, abs=1e-6)

        # Ride-height change equals minus the mean contact-patch rise.
        initial = axle.initial_state()
        cp_dz = []
        for side in (Side.LEFT, Side.RIGHT):
            ref = PointRef(side, PointID.CONTACT_PATCH_CENTER)
            now = float(state.positions[ref][Axis.Z])
            design = float(initial.positions[ref][Axis.Z])
            cp_dz.append(now - design)
        expected_rhc = -0.5 * (cp_dz[0] + cp_dz[1])
        rhc = metrics.axle["ride_height_change_mm"]
        assert rhc is not None
        assert rhc == pytest.approx(expected_rhc, abs=1e-6)
        assert rhc < 0

    def test_antisymmetric_roll(self, axle) -> None:
        left_z0, right_z0, rack_y0, left_wc, right_wc, rack = self._design(axle)
        delta = 10.0

        state = _solve_states(
            axle,
            [
                [
                    _bump_target(left_wc, left_z0 + delta),
                    _bump_target(right_wc, right_z0 - delta),
                    _rack_target(rack, rack_y0),
                ]
            ],
        )[0]
        metrics = compute_metrics_for_axle_state(state, axle, axle.config)

        assert metrics.axle["heave_mm"] == pytest.approx(0.0, abs=1e-4)

        # roll_deg = atan2(dz_left - dz_right, track) with the current track.
        left_cp = float(
            state.positions[PointRef(Side.LEFT, PointID.CONTACT_PATCH_CENTER)][Axis.Y]
        )
        right_cp = float(
            state.positions[PointRef(Side.RIGHT, PointID.CONTACT_PATCH_CENTER)][Axis.Y]
        )
        track = abs(left_cp - right_cp)
        expected_roll = degrees(atan2(2.0 * delta, track))
        roll = metrics.axle["roll_deg"]
        assert roll is not None
        assert roll == pytest.approx(expected_roll, abs=1e-4)
        assert roll > 0  # left wheel in bump

    def test_steer_step_ackermann_finite(self, axle) -> None:
        left_z0, right_z0, rack_y0, left_wc, right_wc, rack = self._design(axle)

        state = _solve_states(
            axle,
            [
                [
                    _bump_target(left_wc, left_z0),
                    _bump_target(right_wc, right_z0),
                    _rack_target(rack, rack_y0 + 50.0),
                ]
            ],
        )[0]
        metrics = compute_metrics_for_axle_state(state, axle, axle.config)

        ackermann = metrics.axle["ackermann_pct"]
        assert ackermann is not None
        assert np.isfinite(ackermann)

        # The per-side yaw-steer angles (change in roadwheel angle folded to a
        # common yaw sign) point the same way under a rack displacement.
        initial = axle.initial_state()
        design_left = compute_metrics_for_axle_state(initial, axle, axle.config)
        roadwheel_angle_left = metrics.corners["left"]["roadwheel_angle_deg"]
        roadwheel_angle_right = metrics.corners["right"]["roadwheel_angle_deg"]
        roadwheel_angle_left_d = design_left.corners["left"]["roadwheel_angle_deg"]
        roadwheel_angle_right_d = design_left.corners["right"]["roadwheel_angle_deg"]
        assert roadwheel_angle_left is not None and roadwheel_angle_right is not None
        assert (
            roadwheel_angle_left_d is not None and roadwheel_angle_right_d is not None
        )
        delta_left = -(roadwheel_angle_left - roadwheel_angle_left_d)
        delta_right = roadwheel_angle_right - roadwheel_angle_right_d
        assert delta_left * delta_right > 0
