"""
Tests for the optional spring/damper (coilover) strut element.

The strut is an all-or-nothing pair of hardpoints (STRUT_TOP, STRUT_BOTTOM).
STRUT_TOP is chassis-fixed; STRUT_BOTTOM is held rigid to whichever body carries
the damper (the lower wishbone for an outboard layout, the rocker for an inboard
layout). The strut length itself is left free -- it is the coilover travel.

Coverage:
1. A partial strut group (only one end authored) is rejected at load time.
2. A heave sweep converges for both the outboard and inboard (rocker) layouts.
3. The three rigid attachment distances that fix STRUT_BOTTOM to its body stay
   constant across the whole sweep (tight tolerance).
4. The strut length changes monotonically over a bump sweep for the outboard
   layout (the coilover compresses smoothly through wheel travel).
5. The analytic Jacobian matches central finite differences on a strut-equipped
   system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics import build_suspension, parse_geometry_spec
from kinematics.core.enums import Axis, PointID
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetPositionMode,
)
from kinematics.io import load_geometry
from kinematics.main import solve_sweep
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import ResidualComputer, convert_targets_to_absolute

TEST_TOLERANCE = 1e-4
FD_STEP = 1e-7
FD_TOLERANCE = 1e-6

# The three body-reference points STRUT_BOTTOM is held rigid to, per layout.
OUTBOARD_REFS = (
    PointID.LOWER_WISHBONE_INBOARD_FRONT,
    PointID.LOWER_WISHBONE_INBOARD_REAR,
    PointID.LOWER_WISHBONE_OUTBOARD,
)
INBOARD_REFS = (
    PointID.ROCKER_AXIS_FRONT,
    PointID.ROCKER_AXIS_REAR,
    PointID.PUSHROD_INBOARD,
)


def _rel(point_id: PointID, axis: Axis, value: float) -> PointTarget:
    return PointTarget(
        point_id=point_id,
        direction=PointTargetAxis(axis),
        value=value,
        mode=TargetPositionMode.RELATIVE,
    )


@pytest.fixture
def corner_strut_file(test_data_dir: Path) -> Path:
    return test_data_dir / "corner_strut_geometry.yaml"


@pytest.fixture
def corner_strut_rocker_file(test_data_dir: Path) -> Path:
    return test_data_dir / "corner_strut_rocker_geometry.yaml"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    data.pop("type", None)
    return data


def _bump_sweep(corner, heave: list[float]) -> list:
    # Pin the steering DOF (trackrod inboard Y) so wheel travel is the only input,
    # exactly as the base model requires a steering target to be well posed.
    cfg = SweepConfig(
        [
            [_rel(PointID.WHEEL_CENTER, Axis.Z, v) for v in heave],
            [_rel(PointID.TRACKROD_INBOARD, Axis.Y, 0.0) for _ in heave],
        ]
    )
    states, stats = solve_sweep(corner, cfg)
    assert all(s.converged for s in stats)
    return states


# ----------------------------------------------------------------------
# 1. Load-time validation
# ----------------------------------------------------------------------


class TestValidation:
    """The strut group is all-or-nothing."""

    def test_partial_strut_group_missing_top_rejected(
        self, corner_strut_file: Path
    ) -> None:
        data = _load_yaml(corner_strut_file)
        del data["hardpoints"]["strut_top"]
        with pytest.raises(ValueError, match="Incomplete strut group"):
            build_suspension(
                parse_geometry_spec({"type": "double_wishbone", **data})
            )

    def test_partial_strut_group_missing_bottom_rejected(
        self, corner_strut_file: Path
    ) -> None:
        data = _load_yaml(corner_strut_file)
        del data["hardpoints"]["strut_bottom"]
        with pytest.raises(ValueError, match="Incomplete strut group"):
            build_suspension(
                parse_geometry_spec({"type": "double_wishbone", **data})
            )

    def test_has_strut_flags(
        self, corner_strut_file: Path, test_data_dir: Path
    ) -> None:
        assert load_geometry(corner_strut_file).has_strut
        # Plain corner geometry authors no strut.
        assert not load_geometry(test_data_dir / "geometry.yaml").has_strut

    def test_strut_bottom_is_free_top_is_fixed(self, corner_strut_file: Path) -> None:
        corner = load_geometry(corner_strut_file)
        free = set(corner.free_points())
        assert PointID.STRUT_BOTTOM in free
        assert PointID.STRUT_TOP not in free


# ----------------------------------------------------------------------
# 2 + 3. Convergence and rigid-distance conservation, both layouts
# ----------------------------------------------------------------------


class TestRigidBody:
    """STRUT_BOTTOM stays rigid to its carrying body through the sweep."""

    @pytest.mark.parametrize(
        "fixture_name, refs",
        [
            ("corner_strut_geometry.yaml", OUTBOARD_REFS),
            ("corner_strut_rocker_geometry.yaml", INBOARD_REFS),
        ],
    )
    def test_rigid_distances_conserved(
        self, test_data_dir: Path, fixture_name: str, refs: tuple[PointID, ...]
    ) -> None:
        corner = load_geometry(test_data_dir / fixture_name)
        assert corner.has_strut
        design = corner.initial_state()
        design_distances = {
            ref: (design.positions[PointID.STRUT_BOTTOM] - design.positions[ref]).norm()
            for ref in refs
        }

        heave = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        states = _bump_sweep(corner, heave)

        for st in states:
            for ref in refs:
                dist = (st.positions[PointID.STRUT_BOTTOM] - st.positions[ref]).norm()
                assert dist == pytest.approx(design_distances[ref], abs=TEST_TOLERANCE)


# ----------------------------------------------------------------------
# 4. Strut length monotonicity (outboard layout)
# ----------------------------------------------------------------------


class TestStrutLength:
    """The coilover length sweeps monotonically for the outboard layout."""

    def test_length_monotonic_over_bump(self, corner_strut_file: Path) -> None:
        corner = load_geometry(corner_strut_file)
        heave = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        states = _bump_sweep(corner, heave)

        lengths = [
            (
                st.positions[PointID.STRUT_TOP] - st.positions[PointID.STRUT_BOTTOM]
            ).norm()
            for st in states
        ]
        diffs = np.diff(lengths)
        assert np.all(diffs < 0.0), f"strut length not monotonic: {lengths}"


# ----------------------------------------------------------------------
# 5. Jacobian
# ----------------------------------------------------------------------


class TestJacobian:
    """Analytic vs finite-difference Jacobian on a strut-equipped corner."""

    def test_analytic_matches_finite_difference(self, corner_strut_file: Path) -> None:
        corner = load_geometry(corner_strut_file)
        initial_state = corner.initial_state()
        derived_manager = DerivedPointsManager(corner.derived_spec())
        state = initial_state.copy()

        targets = convert_targets_to_absolute(
            [
                _rel(PointID.WHEEL_CENTER, Axis.Z, 5.0),
                _rel(PointID.TRACKROD_INBOARD, Axis.Y, 3.0),
            ],
            initial_state,
        )

        rc = ResidualComputer(
            constraints=corner.constraints(),
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
