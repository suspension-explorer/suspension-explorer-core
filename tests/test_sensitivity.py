"""
Validation of solution-manifold tangents against finite differences.

The sensitivity layer computes exact tangents analytically (implicit
function theorem + dual propagation). These tests are the only place finite
differencing appears: solve the mechanism at target +/- h and check that
central differences of positions and metrics agree with the analytical
tangents and rate metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.geometry import extract_array
from kinematics.core.point_ref import PointRef, Side
from kinematics.core.types import PointTarget, PointTargetAxis, SweepConfig
from kinematics.io.geometry_loader import load_geometry
from kinematics.main import solve_sweep
from kinematics.metrics.context import MetricContext
from kinematics.metrics.main import compute_metrics_for_state
from kinematics.metrics.rates import (
    compute_axle_rate_metrics,
    compute_corner_rate_metrics,
)
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.sensitivity import combine_tangents, compute_state_tangents
from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

# Central-difference step for the solved-mechanism comparisons (mm of target
# displacement). Small enough that curvature error is negligible, large
# enough that solver tolerance noise does not dominate the quotient.
FD_STEP = 0.25

# Tolerances for tangent (mm/mm) and rate (deg/mm etc.) comparisons.
TANGENT_RTOL = 1e-3
TANGENT_ATOL = 1e-5
RATE_RTOL = 1e-3
RATE_ATOL = 1e-6


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
    Solve one state per entry of target_lists; each entry is the full list
    of absolute targets for that step.
    """
    sweeps = [
        [target_lists[step][dim] for step in range(len(target_lists))]
        for dim in range(len(target_lists[0]))
    ]
    states, _stats = solve_sweep(suspension, SweepConfig(sweeps))
    return states


class TestCornerTangents:
    """Corner tangents and rate metrics against finite differences."""

    @pytest.fixture
    def corner(self, double_wishbone_geometry_file):
        return load_geometry(double_wishbone_geometry_file)

    def test_tangent_matches_finite_difference(self, corner) -> None:
        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])
        # Evaluate away from design so nothing is special about the point.
        z_0 = design_z + 10.0

        def targets(z: float) -> list[PointTarget]:
            return [
                _bump_target(PointID.WHEEL_CENTER, z),
                _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
            ]

        state = _solve_states(corner, [targets(z_0)])[0]

        derived_manager = DerivedPointsManager(corner.derived_spec())
        field = compute_state_tangents(
            state, corner.constraints(), derived_manager, targets(z_0)
        )[0]

        # Central difference of every point position across two re-solves.
        state_minus, state_plus = _solve_states(
            corner,
            [targets(z_0 - FD_STEP), targets(z_0 + FD_STEP)],
        )

        for point_id in state.positions:
            fd_velocity = (
                extract_array(state_plus.positions[point_id])
                - extract_array(state_minus.positions[point_id])
            ) / (2.0 * FD_STEP)
            np.testing.assert_allclose(
                field.velocity(point_id),
                fd_velocity,
                rtol=TANGENT_RTOL,
                atol=TANGENT_ATOL,
                err_msg=f"tangent mismatch for {point_id!r}",
            )

        # The driven coordinate must move 1:1 with the target by definition.
        assert field.velocity(PointID.WHEEL_CENTER)[Axis.Z] == pytest.approx(
            1.0, abs=1e-6
        )

    def test_rate_metrics_match_finite_difference_of_metrics(self, corner) -> None:
        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])
        z_0 = design_z + 5.0

        def targets(z: float) -> list[PointTarget]:
            return [
                _bump_target(PointID.WHEEL_CENTER, z),
                _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
            ]

        state = _solve_states(corner, [targets(z_0)])[0]

        derived_manager = DerivedPointsManager(corner.derived_spec())
        tangents = compute_state_tangents(
            state, corner.constraints(), derived_manager, targets(z_0)
        )
        rates = compute_corner_rate_metrics(state, corner, tangents)

        state_minus, state_plus = _solve_states(
            corner,
            [targets(z_0 - FD_STEP), targets(z_0 + FD_STEP)],
        )
        assert corner.config is not None
        row_minus = compute_metrics_for_state(state_minus, corner, corner.config)
        row_plus = compute_metrics_for_state(state_plus, corner, corner.config)

        def fd(column: str) -> float:
            hi = row_plus[column]
            lo = row_minus[column]
            assert hi is not None and lo is not None
            return (hi - lo) / (2.0 * FD_STEP)

        expected = {
            "camber_gain_deg_per_mm": fd("camber_deg"),
            "bump_steer_deg_per_mm": fd("roadwheel_angle_deg"),
            "caster_gain_deg_per_mm": fd("caster_deg"),
            "kpi_gain_deg_per_mm": fd("kpi_deg"),
        }
        for column, fd_value in expected.items():
            assert rates[column] == pytest.approx(
                fd_value, rel=RATE_RTOL, abs=RATE_ATOL
            ), column

        # Contact patch rates against direct position differences.
        cp_plus = extract_array(state_plus.positions[PointID.CONTACT_PATCH_CENTER])
        cp_minus = extract_array(state_minus.positions[PointID.CONTACT_PATCH_CENTER])
        side_sign = 1.0  # The corner fixture is a left corner (Y > 0).
        fd_half_track = (
            side_sign * (cp_plus[Axis.Y] - cp_minus[Axis.Y]) / (2.0 * FD_STEP)
        )
        fd_recession = -(cp_plus[Axis.X] - cp_minus[Axis.X]) / (2.0 * FD_STEP)
        # The contact patch lateral rate passes through zero near design, so
        # the comparison is dominated by FD truncation error rather than the
        # rate magnitude; use a looser absolute floor.
        assert rates["half_track_rate_mm_per_mm"] == pytest.approx(
            fd_half_track, rel=RATE_RTOL, abs=1e-5
        )
        assert rates["wheel_recession_rate_mm_per_mm"] == pytest.approx(
            fd_recession, rel=RATE_RTOL, abs=1e-5
        )

    def test_kernel_values_match_catalog_metrics(self, corner) -> None:
        from kinematics.metrics import kernels
        from kinematics.metrics.angles import (
            calculate_camber,
            calculate_caster,
            calculate_kpi,
            calculate_toe,
        )

        design_z = float(corner.initial_state().positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(
            corner.initial_state().positions[PointID.TRACKROD_INBOARD][Axis.Y]
        )
        state = _solve_states(
            corner,
            [
                [
                    _bump_target(PointID.WHEEL_CENTER, design_z + 12.0),
                    _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
                ]
            ],
        )[0]
        assert corner.config is not None
        ctx = MetricContext(state=state, suspension=corner, config=corner.config)
        side = ctx.side_sign

        positions = state.positions
        assert kernels.camber_deg(positions, side) == pytest.approx(
            calculate_camber(ctx), abs=1e-9
        )
        assert kernels.toe_deg(positions, side) == pytest.approx(
            calculate_toe(ctx), abs=1e-9
        )
        assert kernels.caster_deg(positions) == pytest.approx(
            calculate_caster(ctx), abs=1e-9
        )
        assert kernels.kpi_deg(positions, side) == pytest.approx(
            calculate_kpi(ctx), abs=1e-9
        )


class TestRockerMotionRatio:
    """Rocker/torsion-bar motion ratio against finite differences."""

    def test_rocker_ratio_matches_finite_difference(self, test_data_dir) -> None:
        corner = load_geometry(test_data_dir / "corner_rocker_geometry.yaml")
        initial = corner.initial_state()
        design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
        rack_y0 = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])
        z_0 = design_z + 4.0

        def targets(z: float) -> list[PointTarget]:
            return [
                _bump_target(PointID.WHEEL_CENTER, z),
                _rack_target(PointID.TRACKROD_INBOARD, rack_y0),
            ]

        state = _solve_states(corner, [targets(z_0)])[0]

        derived_manager = DerivedPointsManager(corner.derived_spec())
        tangents = compute_state_tangents(
            state, corner.constraints(), derived_manager, targets(z_0)
        )
        rates = compute_corner_rate_metrics(state, corner, tangents)

        state_minus, state_plus = _solve_states(
            corner,
            [targets(z_0 - FD_STEP), targets(z_0 + FD_STEP)],
        )
        assert corner.config is not None
        row_minus = compute_metrics_for_state(state_minus, corner, corner.config)
        row_plus = compute_metrics_for_state(state_plus, corner, corner.config)
        hi = row_plus["rocker_angle_deg"]
        lo = row_minus["rocker_angle_deg"]
        assert hi is not None and lo is not None
        fd_ratio = (hi - lo) / (2.0 * FD_STEP)

        assert rates["rocker_motion_ratio_deg_per_mm"] == pytest.approx(
            fd_ratio, rel=RATE_RTOL, abs=RATE_ATOL
        )
        assert rates["torsion_bar_motion_ratio_deg_per_mm"] == pytest.approx(
            rates["rocker_motion_ratio_deg_per_mm"]
        )


class TestAxleTangents:
    """Axle tangents and modal (roll) rates against finite differences."""

    def test_left_bump_tangent_matches_finite_difference(self, test_data_dir) -> None:
        axle = load_geometry(test_data_dir / "axle_geometry.yaml")
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        initial = axle.initial_state()
        left_wc = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
        right_wc = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
        rack = PointRef(Side.LEFT, PointID.TRACKROD_INBOARD)
        left_z0 = float(initial.positions[left_wc][Axis.Z]) + 3.0
        right_z0 = float(initial.positions[right_wc][Axis.Z])
        rack_y0 = float(initial.positions[rack][Axis.Y])

        def targets(left_z: float) -> list[PointTarget]:
            return [
                _bump_target(left_wc, left_z),
                _bump_target(right_wc, right_z0),
                _rack_target(rack, rack_y0),
            ]

        state = _solve_states(axle, [targets(left_z0)])[0]

        derived_manager = DerivedPointsManager(axle.derived_spec())
        fields = compute_state_tangents(
            state, axle.constraints(), derived_manager, targets(left_z0)
        )
        left_field = fields[0]

        state_minus, state_plus = _solve_states(
            axle,
            [targets(left_z0 - FD_STEP), targets(left_z0 + FD_STEP)],
        )

        for point_id in state.positions:
            fd_velocity = (
                extract_array(state_plus.positions[point_id])
                - extract_array(state_minus.positions[point_id])
            ) / (2.0 * FD_STEP)
            np.testing.assert_allclose(
                left_field.velocity(point_id),
                fd_velocity,
                rtol=TANGENT_RTOL,
                atol=TANGENT_ATOL,
                err_msg=f"tangent mismatch for {point_id!r}",
            )

    def test_modal_rates_match_finite_difference_roll(self, test_data_dir) -> None:
        axle = load_geometry(test_data_dir / "axle_geometry.yaml")
        assert isinstance(axle, DoubleWishboneAxleSuspension)
        initial = axle.initial_state()
        left_wc = PointRef(Side.LEFT, PointID.WHEEL_CENTER)
        right_wc = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
        rack = PointRef(Side.LEFT, PointID.TRACKROD_INBOARD)
        left_z0 = float(initial.positions[left_wc][Axis.Z])
        right_z0 = float(initial.positions[right_wc][Axis.Z])
        rack_y0 = float(initial.positions[rack][Axis.Y])

        def targets(dz_left: float, dz_right: float) -> list[PointTarget]:
            return [
                _bump_target(left_wc, left_z0 + dz_left),
                _bump_target(right_wc, right_z0 + dz_right),
                _rack_target(rack, rack_y0),
            ]

        state = _solve_states(axle, [targets(0.0, 0.0)])[0]
        derived_manager = DerivedPointsManager(axle.derived_spec())
        fields = compute_state_tangents(
            state, axle.constraints(), derived_manager, targets(0.0, 0.0)
        )
        rates = compute_axle_rate_metrics(state, axle, fields)

        # Finite-difference roll: displace the wheel pair antisymmetrically
        # by the same half-track lever the modal combination uses.
        left_cp = extract_array(
            state.positions[PointRef(Side.LEFT, PointID.CONTACT_PATCH_CENTER)]
        )
        right_cp = extract_array(
            state.positions[PointRef(Side.RIGHT, PointID.CONTACT_PATCH_CENTER)]
        )
        track = abs(float(left_cp[Axis.Y]) - float(right_cp[Axis.Y]))
        roll_step_deg = 0.05
        dz = (track / 2.0) * np.deg2rad(roll_step_deg)

        state_minus, state_plus = _solve_states(
            axle, [targets(-dz, dz), targets(dz, -dz)]
        )
        assert axle.config is not None
        row_minus = compute_metrics_for_state(
            axle.corner_state(state_minus, Side.LEFT),
            axle.corners[Side.LEFT],
            axle.config,
        )
        row_plus = compute_metrics_for_state(
            axle.corner_state(state_plus, Side.LEFT),
            axle.corners[Side.LEFT],
            axle.config,
        )

        def fd(column: str) -> float:
            hi = row_plus[column]
            lo = row_minus[column]
            assert hi is not None and lo is not None
            return (hi - lo) / (2.0 * roll_step_deg)

        assert rates["left_toe_vs_roll_deg_per_deg"] == pytest.approx(
            fd("roadwheel_angle_deg"), rel=5e-3, abs=1e-5
        )
        assert rates["left_camber_vs_roll_deg_per_deg"] == pytest.approx(
            fd("camber_deg"), rel=5e-3, abs=1e-5
        )


class TestCombineTangents:
    """Linear combination of tangent fields."""

    def test_linear_combination(self) -> None:
        from kinematics.core.types import PointTarget, PointTargetAxis
        from kinematics.sensitivity import TangentField

        target = PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(axis=Axis.Z),
            value=0.0,
            mode=TargetPositionMode.ABSOLUTE,
        )
        field_a = TangentField(
            target_index=0,
            target=target,
            velocities={PointID.WHEEL_CENTER: np.array([1.0, 0.0, 0.0])},
        )
        field_b = TangentField(
            target_index=1,
            target=target,
            velocities={
                PointID.WHEEL_CENTER: np.array([0.0, 2.0, 0.0]),
                PointID.CONTACT_PATCH_CENTER: np.array([0.0, 0.0, 3.0]),
            },
        )
        combined = combine_tangents([field_a, field_b], [2.0, -1.0])
        np.testing.assert_allclose(combined[PointID.WHEEL_CENTER], [2.0, -2.0, 0.0])
        np.testing.assert_allclose(
            combined[PointID.CONTACT_PATCH_CENTER], [0.0, 0.0, -3.0]
        )
