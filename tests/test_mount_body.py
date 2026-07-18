"""Tests for selectable actuation mount bodies.

These tests cover the mount option that lets a corner actuation pickup be fixed
to a chosen rigid body: a direct coil spring or a pushrod outboard end may be
carried by the lower wishbone or by the upright. They exercise the schema, the
build-time body lookup, the solved physical invariants, and the mount-body
anchor validation.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from kinematics.core.enums import (
    ActuationType,
    Axis,
    MountBody,
    PointID,
    TargetPositionMode,
)
from kinematics.core.input import build_suspension, parse_geometry_spec
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.schema.geometry import (
    ActuationSpec,
    DoubleWishboneGeometrySpec,
)
from kinematics.core.suspensions.build import build_actuation
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.corner.mechanisms import (
    ActuationDirect,
    ActuationPushrodRocker,
)
from kinematics.core.sweep import solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig

DATA_DIR = Path(__file__).parent / "data"
STRUT_GEOMETRY = DATA_DIR / "corner_strut_geometry.yaml"
ROCKER_GEOMETRY = DATA_DIR / "corner_rocker_geometry.yaml"
ROCKER_COILOVER_GEOMETRY = DATA_DIR / "corner_strut_rocker_geometry.yaml"


def _build_corner_with_mount(
    geometry_path: Path,
    mount: MountBody,
    *,
    shim_setup_thickness: float | None = None,
):
    """Load a corner geometry file with the actuation mount overridden.

    The mount key is always written, so this works whether or not the stock
    YAML already carries one.
    """
    with open(geometry_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data["actuation"]["mount"] = mount.value
    if shim_setup_thickness is not None:
        data["config"]["camber_shim"] = {
            "shim_face_point_a": {"x": -25.0, "y": 750.0, "z": 510.0},
            "shim_face_point_b": {"x": -25.0, "y": 750.0, "z": 490.0},
            "shim_face_normal": {"x": 0.0, "y": 1.0, "z": 0.0},
            "design_thickness": 30.0,
            "setup_thickness": shim_setup_thickness,
        }
    return build_suspension(data)


def _heave_sweep(displacements: tuple[float, ...]) -> SweepConfig:
    """Build a wheel-center vertical travel sweep in relative millimetres.

    The steering rack point is held at its design lateral position across the
    sweep. Without this second dimension a corner keeps a free steering degree
    of freedom, leaving the solve underdetermined and the absolute pose
    non-unique.
    """
    heave_targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(Axis.Z),
            value=displacement,
            mode=TargetPositionMode.RELATIVE,
        )
        for displacement in displacements
    ]
    steer_hold_targets = [
        PointTarget(
            point_id=PointID.TRACKROD_INBOARD,
            direction=PointTargetAxis(Axis.Y),
            value=0.0,
            mode=TargetPositionMode.RELATIVE,
        )
        for _ in displacements
    ]
    return SweepConfig([heave_targets, steer_hold_targets])


def _distance(positions, point_a: PointID, point_b: PointID) -> float:
    """Return the straight-line distance between two solved positions."""
    return float(np.linalg.norm(positions[point_a] - positions[point_b]))


def _distance_between(point_a: Point3, point_b: Point3) -> float:
    """Return the straight-line distance between two solved points."""
    return float(np.linalg.norm(point_a - point_b))


class TestActuationMountSchema:
    """The actuation schema carries an explicitly required mount body."""

    def test_mount_parses_from_string(self):
        spec = ActuationSpec(type=ActuationType.DIRECT, mount="upright")
        assert spec.mount is MountBody.UPRIGHT

    def test_mount_is_required_on_actuation_spec(self):
        # mount has no default; omitting it is a validation error.
        with pytest.raises(ValidationError):
            ActuationSpec.model_validate({"type": ActuationType.DIRECT})

    def test_full_geometry_spec_carries_mount_through_loader(self):
        with open(STRUT_GEOMETRY, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        data["actuation"]["mount"] = MountBody.UPRIGHT.value
        spec = parse_geometry_spec(data)
        assert isinstance(spec, DoubleWishboneGeometrySpec)
        assert spec.actuation.mount is MountBody.UPRIGHT

    def test_full_geometry_spec_requires_mount(self):
        with open(STRUT_GEOMETRY, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        # Ensure no mount is present so the required field is genuinely missing.
        data["actuation"].pop("mount", None)
        with pytest.raises(ValueError):
            parse_geometry_spec(data)


class TestBuildActuationMountMapping:
    """build_actuation resolves the named mount to an architecture body."""

    def test_direct_lower_wishbone_uses_lower_wishbone_body(self):
        spec = ActuationSpec(type=ActuationType.DIRECT, mount=MountBody.LOWER_WISHBONE)
        actuation = build_actuation(
            spec, mount_bodies=DoubleWishboneSuspension.MOUNT_BODIES
        )
        assert isinstance(actuation, ActuationDirect)
        assert (
            actuation.spring_pickup_body == DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        )

    def test_direct_upright_uses_upright_body(self):
        spec = ActuationSpec(type=ActuationType.DIRECT, mount=MountBody.UPRIGHT)
        actuation = build_actuation(
            spec, mount_bodies=DoubleWishboneSuspension.MOUNT_BODIES
        )
        assert isinstance(actuation, ActuationDirect)
        assert actuation.spring_pickup_body == DoubleWishboneSuspension.UPRIGHT_BODY

    def test_pushrod_rocker_upright_uses_upright_body(self):
        spec = ActuationSpec(type=ActuationType.PUSHROD_ROCKER, mount=MountBody.UPRIGHT)
        actuation = build_actuation(
            spec, mount_bodies=DoubleWishboneSuspension.MOUNT_BODIES
        )
        assert isinstance(actuation, ActuationPushrodRocker)
        assert actuation.pushrod_outboard_body == DoubleWishboneSuspension.UPRIGHT_BODY

    def test_pushrod_rocker_lower_wishbone_uses_lower_wishbone_body(self):
        spec = ActuationSpec(
            type=ActuationType.PUSHROD_ROCKER, mount=MountBody.LOWER_WISHBONE
        )
        actuation = build_actuation(
            spec, mount_bodies=DoubleWishboneSuspension.MOUNT_BODIES
        )
        assert isinstance(actuation, ActuationPushrodRocker)
        assert (
            actuation.pushrod_outboard_body
            == DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        )

    def test_unknown_mount_body_is_rejected(self):
        spec = ActuationSpec(type=ActuationType.DIRECT, mount=MountBody.UPRIGHT)
        # The mapping lacks the requested upright body.
        mount_bodies = {
            MountBody.LOWER_WISHBONE: DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        }
        with pytest.raises(ValueError, match="does not provide"):
            build_actuation(spec, mount_bodies=mount_bodies)


class TestMountedSpringSolveInvariants:
    """Solved sweeps preserve the rigid attachment implied by the mount body."""

    # The stock strut_bottom (0, 780, 240) sits about 26 mm off the plane of
    # the first three upright anchors, well clear of MIN_CHIRALITY_VOLUME, so
    # no coordinate adjustment is needed for the upright-mounted coilover.
    HEAVE_TARGETS = (-20.0, 0.0, 20.0, 40.0)

    @pytest.mark.parametrize(
        ("geometry_path", "moving_pickup"),
        (
            (STRUT_GEOMETRY, PointID.STRUT_BOTTOM),
            (ROCKER_GEOMETRY, PointID.PUSHROD_OUTBOARD),
        ),
    )
    @pytest.mark.parametrize(
        ("mount", "should_move_with_shim"),
        (
            (MountBody.UPRIGHT, True),
            (MountBody.LOWER_WISHBONE, False),
        ),
    )
    def test_camber_shim_follows_actuation_mount_body(
        self,
        geometry_path: Path,
        moving_pickup: PointID,
        mount: MountBody,
        should_move_with_shim: bool,
    ):
        design = _build_corner_with_mount(geometry_path, mount)
        shimmed = _build_corner_with_mount(
            geometry_path,
            mount,
            shim_setup_thickness=40.0,
        )

        design_position = design.initial_state().positions[moving_pickup]
        shimmed_position = shimmed.initial_state().positions[moving_pickup]
        movement = _distance_between(design_position, shimmed_position)

        if should_move_with_shim:
            assert movement > 0.1
        else:
            assert movement == pytest.approx(0.0, abs=TEST_TOLERANCE)

    def test_camber_shim_preserves_upright_mounted_pushrod_length(self):
        design = _build_corner_with_mount(ROCKER_GEOMETRY, MountBody.UPRIGHT)
        shimmed = _build_corner_with_mount(
            ROCKER_GEOMETRY,
            MountBody.UPRIGHT,
            shim_setup_thickness=40.0,
        )
        design_positions = design.initial_state().positions
        shimmed_positions = shimmed.initial_state().positions

        design_length = _distance(
            design_positions,
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
        )
        shimmed_length = _distance(
            shimmed_positions,
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
        )

        assert shimmed_length == pytest.approx(
            design_length,
            abs=TEST_TOLERANCE,
        )
        assert (
            _distance_between(
                design_positions[PointID.PUSHROD_INBOARD],
                shimmed_positions[PointID.PUSHROD_INBOARD],
            )
            > TEST_TOLERANCE
        )

    def test_camber_shim_reindexes_all_rocker_mounted_pickups(self):
        design = _build_corner_with_mount(
            ROCKER_COILOVER_GEOMETRY,
            MountBody.UPRIGHT,
        )
        shimmed = _build_corner_with_mount(
            ROCKER_COILOVER_GEOMETRY,
            MountBody.UPRIGHT,
            shim_setup_thickness=40.0,
        )
        design_positions = design.initial_state().positions
        shimmed_positions = shimmed.initial_state().positions

        design_pickup_distance = _distance(
            design_positions,
            PointID.PUSHROD_INBOARD,
            PointID.STRUT_BOTTOM,
        )
        shimmed_pickup_distance = _distance(
            shimmed_positions,
            PointID.PUSHROD_INBOARD,
            PointID.STRUT_BOTTOM,
        )

        assert shimmed_pickup_distance == pytest.approx(
            design_pickup_distance,
            abs=TEST_TOLERANCE,
        )

    def test_upright_mounted_coilover_holds_rigid_to_upright_body(self):
        suspension = _build_corner_with_mount(STRUT_GEOMETRY, MountBody.UPRIGHT)
        design = suspension.initial_state().positions
        design_distances = {
            anchor: _distance(design, PointID.STRUT_BOTTOM, anchor)
            for anchor in DoubleWishboneSuspension.UPRIGHT_BODY
        }

        states, _ = solve_sweep(suspension, _heave_sweep(self.HEAVE_TARGETS))

        for state in states:
            for anchor, design_distance in design_distances.items():
                solved_distance = _distance(
                    state.positions, PointID.STRUT_BOTTOM, anchor
                )
                assert solved_distance == pytest.approx(
                    design_distance, abs=TEST_TOLERANCE
                )

    def test_lower_wishbone_mounted_coilover_holds_rigid_to_lower_wishbone(self):
        suspension = _build_corner_with_mount(STRUT_GEOMETRY, MountBody.LOWER_WISHBONE)
        design = suspension.initial_state().positions
        design_distances = {
            anchor: _distance(design, PointID.STRUT_BOTTOM, anchor)
            for anchor in DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        }

        states, _ = solve_sweep(suspension, _heave_sweep(self.HEAVE_TARGETS))

        for state in states:
            for anchor, design_distance in design_distances.items():
                solved_distance = _distance(
                    state.positions, PointID.STRUT_BOTTOM, anchor
                )
                assert solved_distance == pytest.approx(
                    design_distance, abs=TEST_TOLERANCE
                )

    def test_mount_choice_changes_strut_bottom_under_travel(self):
        upright = _build_corner_with_mount(STRUT_GEOMETRY, MountBody.UPRIGHT)
        lower = _build_corner_with_mount(STRUT_GEOMETRY, MountBody.LOWER_WISHBONE)

        sweep = _heave_sweep(self.HEAVE_TARGETS)
        upright_states, _ = solve_sweep(upright, sweep)
        lower_states, _ = solve_sweep(lower, sweep)

        design_index = self.HEAVE_TARGETS.index(0.0)
        displaced_index = self.HEAVE_TARGETS.index(40.0)

        # At the design state the pickup is authored identically for both mounts.
        design_separation = _distance_between(
            upright_states[design_index].positions[PointID.STRUT_BOTTOM],
            lower_states[design_index].positions[PointID.STRUT_BOTTOM],
        )
        assert design_separation == pytest.approx(0.0, abs=TEST_TOLERANCE)

        # Under travel the two rigid bodies move differently, so the pickup that
        # rides on them must separate. This proves the mount option is kinematic.
        travel_separation = _distance_between(
            upright_states[displaced_index].positions[PointID.STRUT_BOTTOM],
            lower_states[displaced_index].positions[PointID.STRUT_BOTTOM],
        )
        assert travel_separation > 1.0

    def test_lower_wishbone_mounted_pushrod_holds_rigid_and_keeps_length(self):
        # The stock pushrod_outboard (0, 870, 240) sits 40 mm off the plane of
        # the lower wishbone anchors, so the chiral attachment is well defined.
        suspension = _build_corner_with_mount(ROCKER_GEOMETRY, MountBody.LOWER_WISHBONE)
        design = suspension.initial_state().positions
        design_distances = {
            anchor: _distance(design, PointID.PUSHROD_OUTBOARD, anchor)
            for anchor in DoubleWishboneSuspension.LOWER_WISHBONE_BODY
        }
        design_pushrod_length = _distance(
            design, PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD
        )

        states, _ = solve_sweep(suspension, _heave_sweep(self.HEAVE_TARGETS))

        for state in states:
            for anchor, design_distance in design_distances.items():
                solved_distance = _distance(
                    state.positions, PointID.PUSHROD_OUTBOARD, anchor
                )
                assert solved_distance == pytest.approx(
                    design_distance, abs=TEST_TOLERANCE
                )
            pushrod_length = _distance(
                state.positions, PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD
            )
            assert pushrod_length == pytest.approx(
                design_pushrod_length, abs=TEST_TOLERANCE
            )


class TestActuationMountValidation:
    """Direct actuation rejects degenerate mounting-body anchor sets."""

    def test_direct_actuation_rejects_fewer_than_three_anchors(self):
        actuation = ActuationDirect(
            spring_pickup_body=(
                PointID.LOWER_WISHBONE_INBOARD_FRONT,
                PointID.LOWER_WISHBONE_INBOARD_REAR,
            )
        )
        with pytest.raises(ValueError, match="at least three mounting body anchors"):
            actuation.validate({})

    def test_direct_actuation_rejects_collinear_first_three_anchors(self):
        # Three anchors strung along world Y leave rotation about that line free.
        hardpoints: dict[PointKey, Point3] = {
            PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3([0.0, 0.0, 0.0]),
            PointID.LOWER_WISHBONE_INBOARD_REAR: Point3([0.0, 100.0, 0.0]),
            PointID.LOWER_WISHBONE_OUTBOARD: Point3([0.0, 200.0, 0.0]),
        }
        actuation = ActuationDirect(
            spring_pickup_body=(
                PointID.LOWER_WISHBONE_INBOARD_FRONT,
                PointID.LOWER_WISHBONE_INBOARD_REAR,
                PointID.LOWER_WISHBONE_OUTBOARD,
            )
        )
        with pytest.raises(ValueError, match="must not be collinear"):
            actuation.validate(hardpoints)
