"""
The axle composer must accept any CornerSuspension, not one architecture.

These tests compose a deliberately minimal non-double-wishbone corner into
AxleSuspension and drive it through composition, solving, metrics, and
diagnostics. The stub intentionally has no AXLE_INBOARD or AXLE_OUTBOARD
points, so every metric path must resolve the wheel axis through the role
hooks rather than double-wishbone conventions. This file is the guard
against reintroducing a concrete corner dependency anywhere in that chain.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Sequence

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
)
from kinematics.core.elements import (
    ElementType,
    RackElement,
    RigidLinkElement,
    SuspensionElement,
)
from kinematics.core.enums import Axis, PointID, SuspensionType
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, PointRef, Side
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig

TEST_DATA = Path(__file__).parent / "data"

CHASSIS_FRONT = PointID.LOWER_WISHBONE_INBOARD_FRONT
CHASSIS_REAR = PointID.LOWER_WISHBONE_INBOARD_REAR
KNUCKLE = PointID.LOWER_WISHBONE_OUTBOARD

STUB_DIAGNOSTIC_MESSAGE = "stub corner diagnostic"


@dataclass
class TrailingArmCorner(CornerSuspension):
    """
    Minimal unsteered corner: a rigid knuckle body hinged on two chassis
    anchors. Deliberately not a double wishbone and deliberately without
    AXLE_INBOARD/AXLE_OUTBOARD points.
    """

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            CHASSIS_FRONT,
            CHASSIS_REAR,
            KNUCKLE,
            PointID.WHEEL_CENTER,
            PointID.CONTACT_PATCH_CENTER,
        }
    )
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        KNUCKLE,
        PointID.WHEEL_CENTER,
        PointID.CONTACT_PATCH_CENTER,
    )
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        KNUCKLE,
        PointID.WHEEL_CENTER,
        PointID.CONTACT_PATCH_CENTER,
    )

    def initial_state(self) -> SuspensionState:
        if self._initial_state is None:
            self._initial_state = SuspensionState(
                positions=self.get_hardpoints_copy(),
                free_points=set[PointKey](self.FREE_POINTS),
            )
        return self._initial_state

    def free_points(self) -> Sequence[PointID]:
        return self.FREE_POINTS

    def constraints(self) -> list[Constraint]:
        positions = self.initial_state().positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        # Each moving point rides on the hinge axis through both chassis
        # anchors; the pairwise distances make the three move as one body.
        constraints: list[Constraint] = [
            distance(moving, anchor)
            for moving in self.FREE_POINTS
            for anchor in (CHASSIS_FRONT, CHASSIS_REAR)
        ]
        constraints.append(distance(KNUCKLE, PointID.WHEEL_CENTER))
        constraints.append(distance(PointID.WHEEL_CENTER, PointID.CONTACT_PATCH_CENTER))
        constraints.append(distance(KNUCKLE, PointID.CONTACT_PATCH_CENTER))
        return constraints

    def derived_spec(self) -> DerivedPointsSpec:
        return DerivedPointsSpec({}, {})

    def compute_side_view_instant_center(self, state: SuspensionState) -> None:
        return None

    def compute_front_view_instant_center(self, state: SuspensionState) -> None:
        return None

    def elements(self) -> tuple[SuspensionElement, ...]:
        return (
            RigidLinkElement(
                label="Trailing Arm",
                type=ElementType.WISHBONE,
                point_a=CHASSIS_FRONT,
                point_b=KNUCKLE,
            ),
        )

    def wheel_axis_points(self) -> tuple[PointID, PointID]:
        # Not the double-wishbone convention: metrics must honor this or
        # fail on the missing AXLE_INBOARD/AXLE_OUTBOARD points.
        return (KNUCKLE, PointID.WHEEL_CENTER)

    def steering_axis_points(self) -> tuple[PointID, PointID]:
        return (KNUCKLE, CHASSIS_FRONT)

    def rack_attachment_point(self) -> PointID | None:
        return None

    def topology_diagnostics(
        self,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        return [
            DiagnosticIssue(
                None,
                DiagnosticCategory.CHIRALITY,
                DiagnosticSeverity.WARNING,
                STUB_DIAGNOSTIC_MESSAGE,
                None,
            )
        ]


@dataclass
class SteeredTrailingArmCorner(TrailingArmCorner):
    """Stub corner that claims a rack attachment for mixed-steering tests."""

    def rack_attachment_point(self) -> PointID | None:
        return PointID.TRACKROD_INBOARD


def build_stub_corner(
    side: Side,
    corner_class: type[TrailingArmCorner] = TrailingArmCorner,
    config: SuspensionConfig | None = None,
) -> TrailingArmCorner:
    lateral = 600.0 if side is Side.LEFT else -600.0
    return corner_class(
        name=f"stub_{side.name.lower()}",
        side=side,
        config=config,
        hardpoints={
            CHASSIS_FRONT: Point3([100.0, 0.3 * lateral, 150.0]),
            CHASSIS_REAR: Point3([-100.0, 0.3 * lateral, 150.0]),
            KNUCKLE: Point3([0.0, 0.9 * lateral, 50.0]),
            PointID.WHEEL_CENTER: Point3([0.0, lateral, 0.0]),
            PointID.CONTACT_PATCH_CENTER: Point3([0.0, lateral, -200.0]),
        },
    )


def build_stub_axle(config: SuspensionConfig | None = None) -> AxleSuspension:
    # The stub corners have no registered architecture; any member works as
    # the reported identity here.
    return AxleSuspension(
        type_key=SuspensionType.DOUBLE_WISHBONE,
        name="stub_axle",
        side=Side.CENTER,
        hardpoints={},
        config=config,
        corners={
            Side.LEFT: build_stub_corner(Side.LEFT, config=config),
            Side.RIGHT: build_stub_corner(Side.RIGHT, config=config),
        },
    )


def test_axle_composes_non_double_wishbone_corners():
    axle = build_stub_axle()

    state = axle.initial_state()
    assert state.free_points == {
        PointRef(side, point)
        for side in (Side.LEFT, Side.RIGHT)
        for point in TrailingArmCorner.FREE_POINTS
    }

    # Remapped corner constraints only: no rack coupling for unsteered
    # corners, and every involved point is side-qualified.
    constraints = axle.constraints()
    per_corner = len(build_stub_corner(Side.LEFT).constraints())
    assert len(constraints) == 2 * per_corner
    assert all(
        isinstance(point, PointRef)
        for constraint in constraints
        for point in constraint.involved_points
    )

    assert not any(isinstance(element, RackElement) for element in axle.elements())
    assert axle.rack_attachment_points() is None


def test_axle_rejects_mixed_rack_attachment():
    with pytest.raises(ValueError, match="disagree on rack attachment"):
        AxleSuspension(
            type_key=SuspensionType.DOUBLE_WISHBONE,
            name="stub_axle",
            side=Side.CENTER,
            hardpoints={},
            corners={
                Side.LEFT: build_stub_corner(Side.LEFT, SteeredTrailingArmCorner),
                Side.RIGHT: build_stub_corner(Side.RIGHT),
            },
        )


def test_stub_axle_solves_and_reports_metrics_through_role_hooks():
    donor = load_geometry(TEST_DATA / "geometry.yaml")
    axle = build_stub_axle(config=donor.config)

    bump_values = [-10.0, 0.0, 10.0]
    sweep = SweepConfig(
        target_sweeps=[
            [
                PointTarget(
                    PointRef(side, PointID.WHEEL_CENTER),
                    PointTargetAxis(Axis.Z),
                    value,
                )
                for value in bump_values
            ]
            for side in (Side.LEFT, Side.RIGHT)
        ]
    )

    states, infos = solve_sweep(axle, sweep)
    assert all(info.converged for info in infos)
    assert all(info.max_residual < 1e-6 for info in infos)

    metrics = compute_sweep_metrics(axle, sweep, states)
    assert metrics.derivative_error is None
    final = metrics.rows[-1]
    assert isinstance(final, AxleMetricRows)

    # Scalar and derivative metrics both resolved the stub's wheel axis and
    # steering axis through the role hooks; the double-wishbone point names
    # do not exist in this state.
    for location in ("left", "right"):
        row = final.corners[location]
        assert row["camber"] is not None
        assert row["caster"] is not None
        assert row["deriv_camber_wrt_hub_z"] is not None
        # Unsteered corners declare no bump-steer derivatives.
        assert "deriv_roadwheel_angle_wrt_rack_displacement" not in row

    assert final.axle["heave"] == pytest.approx(bump_values[-1], abs=1e-6)
    assert final.axle["rack_displacement"] is None

    # Corner-owned diagnostics survive axle composition.
    diagnostics = axle.topology_diagnostics(states)
    stub_issues = [
        issue for issue in diagnostics if issue.message == STUB_DIAGNOSTIC_MESSAGE
    ]
    assert len(stub_issues) == 2

    # The composer reports the identity its builder supplied.
    assert axle.reported_type_key() is SuspensionType.DOUBLE_WISHBONE


def test_double_wishbone_declares_expected_point_roles():
    suspension = load_geometry(TEST_DATA / "geometry.yaml")
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.wheel_axis_points() == (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    assert suspension.steering_axis_points() == (
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_OUTBOARD,
    )
    assert suspension.rack_attachment_point() is PointID.TRACKROD_INBOARD


def test_metric_context_resolves_steering_axis_through_role_hooks():
    donor = load_geometry(TEST_DATA / "geometry.yaml")
    corner = build_stub_corner(Side.LEFT, config=donor.config)
    assert corner.config is not None
    ctx = MetricContext(
        state=corner.initial_state(),
        suspension=corner,
        config=corner.config,
    )

    lower, upper = ctx.steering_axis_pivots
    # The stub declares KNUCKLE as the lower pivot and CHASSIS_FRONT as the
    # upper pivot.
    initial = corner.initial_state()
    assert (lower - initial.get(KNUCKLE)).norm() == pytest.approx(0.0)
    assert (upper - initial.get(CHASSIS_FRONT)).norm() == pytest.approx(0.0)
    expected_direction = (upper - lower).normalize()
    assert ctx.steering_axis.data == pytest.approx(expected_direction.data)
