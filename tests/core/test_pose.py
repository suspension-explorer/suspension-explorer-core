"""Tests for the post-solve chassis pose interpretation of an axle ground datum."""

from __future__ import annotations

from math import atan2, degrees
from pathlib import Path

import numpy as np
import pytest

from kinematics.core.enums import AxlePosition, PointID
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.pose import (
    ChassisPose,
    PoseAssumption,
    build_chassis_pose,
    chassis_pose_for_axle_state,
)
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side

WHEELBASE = 2500.0
AXLE_X = 1500.0
HALF_TRACK = 600.0
DOWN_WORLD = np.array((0.0, 0.0, -1.0))


def _datum(left_z: float, right_z: float, *, x: float = 0.0) -> GroundDatum:
    ground = GroundDatum.from_wheel_ground_tangents(
        Point3((x, HALF_TRACK, left_z)),
        Point3((x, -HALF_TRACK, right_z)),
    )
    assert ground is not None
    return ground


def _pose(
    ground: GroundDatum,
    design_ground: GroundDatum,
    *,
    assumption: PoseAssumption = PoseAssumption.PURE_HEAVE,
    axle_position: AxlePosition = AxlePosition.FRONT,
    wheelbase: float = WHEELBASE,
    axle_x: float = AXLE_X,
) -> ChassisPose:
    pose = build_chassis_pose(
        ground,
        design_ground,
        wheelbase=wheelbase,
        axle_position=axle_position,
        assumption=assumption,
        axle_x=axle_x,
    )
    assert pose is not None
    return pose


@pytest.mark.parametrize("assumption", list(PoseAssumption))
def test_flat_ground_at_design_height_is_a_level_pose(
    assumption: PoseAssumption,
) -> None:
    ground = _datum(20.0, 20.0)
    pose = _pose(ground, ground, assumption=assumption)

    assert pose.roll_deg == pytest.approx(0.0)
    assert pose.pitch_deg == pytest.approx(0.0)
    assert pose.assumption is assumption
    assert pose.anchor.data == pytest.approx((AXLE_X, 0.0, 20.0))
    assert pose.gravity_direction_chassis().data == pytest.approx((0.0, 0.0, -1.0))
    assert pose.rotation_chassis_to_world == pytest.approx(np.eye(3))


def test_pose_maps_the_anchor_to_the_world_origin_and_preserves_distances() -> None:
    # A rolled and pitched pose: the rotation must stay rigid.
    pose = _pose(
        _datum(60.0, 0.0),
        _datum(0.0, 0.0),
        assumption=PoseAssumption.OPPOSITE_AXLE_FIXED,
    )
    first = Point3((AXLE_X + 120.0, 700.0, 350.0))
    second = Point3((AXLE_X - 45.0, -180.0, 90.0))

    assert pose.to_world(pose.anchor).data == pytest.approx((0.0, 0.0, 0.0))
    assert (pose.to_world(first) - pose.to_world(second)).norm() == pytest.approx(
        (first - second).norm()
    )
    assert (pose.to_world(first) - pose.to_world(pose.anchor)).norm() == pytest.approx(
        (first - pose.anchor).norm()
    )


def test_banked_ground_rolls_the_chassis_opposite_the_ground_line() -> None:
    # The left wheel's contact sits higher in chassis space, so the road rises
    # to the left: relative to that road the chassis left side is *down*.
    ground = _datum(60.0, 0.0)
    design_ground = _datum(0.0, 0.0)
    pose = _pose(ground, design_ground)

    assert ground.angle_deg == pytest.approx(degrees(atan2(60.0, 1200.0)))
    assert pose.roll_deg == pytest.approx(-ground.angle_deg)
    assert pose.roll_deg < 0.0

    gravity = pose.gravity_direction_chassis().data
    matrix_derived = pose.rotation_chassis_to_world.T @ DOWN_WORLD
    physical = -np.asarray(ground.normal.data)

    # Matrix-derived and physical (anti-parallel to the ground normal) agree.
    assert gravity == pytest.approx(matrix_derived)
    assert gravity == pytest.approx(physical)
    # Downhill is toward vehicle right, i.e. gravity leans toward +Y.
    assert gravity[1] > 0.0


def test_left_side_up_puts_gravity_toward_negative_chassis_y() -> None:
    # Mirror image: the road rises to the right, so the chassis left side is up.
    ground = _datum(0.0, 60.0)
    pose = _pose(ground, _datum(0.0, 0.0))

    assert pose.roll_deg == pytest.approx(-ground.angle_deg)
    assert pose.roll_deg > 0.0

    gravity = pose.gravity_direction_chassis().data
    assert gravity == pytest.approx(pose.rotation_chassis_to_world.T @ DOWN_WORLD)
    assert gravity == pytest.approx(-np.asarray(ground.normal.data))
    assert gravity[1] < 0.0


@pytest.mark.parametrize(
    ("axle_position", "sign"),
    [(AxlePosition.FRONT, 1.0), (AxlePosition.REAR, -1.0)],
)
def test_compression_pitches_the_chassis_about_the_opposite_axle(
    axle_position: AxlePosition,
    sign: float,
) -> None:
    # The solved ground line sits 12 mm above the design line at the centreline,
    # which is 12 mm of compression at the modelled axle.
    compression = 12.0
    pose = _pose(
        _datum(compression, compression),
        _datum(0.0, 0.0),
        assumption=PoseAssumption.OPPOSITE_AXLE_FIXED,
        axle_position=axle_position,
    )

    assert pose.pitch_deg == pytest.approx(
        sign * degrees(atan2(compression, WHEELBASE))
    )
    assert pose.pitch_deg == pytest.approx(sign * 0.27502, abs=1e-5)
    assert pose.roll_deg == pytest.approx(0.0)


@pytest.mark.parametrize("axle_position", list(AxlePosition))
@pytest.mark.parametrize("centerline_z", [-25.0, 0.0, 25.0])
def test_pure_heave_never_pitches_the_chassis(
    axle_position: AxlePosition,
    centerline_z: float,
) -> None:
    pose = _pose(
        _datum(centerline_z, centerline_z),
        _datum(0.0, 0.0),
        assumption=PoseAssumption.PURE_HEAVE,
        axle_position=axle_position,
    )

    assert pose.pitch_deg == 0.0
    assert pose.anchor.data == pytest.approx((AXLE_X, 0.0, centerline_z))


def test_extension_at_a_front_axle_lifts_the_nose() -> None:
    pose = _pose(
        _datum(-12.0, -12.0),
        _datum(0.0, 0.0),
        assumption=PoseAssumption.OPPOSITE_AXLE_FIXED,
        axle_position=AxlePosition.FRONT,
    )

    assert pose.pitch_deg == pytest.approx(-degrees(atan2(12.0, WHEELBASE)))


@pytest.mark.parametrize("axle_x", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_axle_station_has_no_pose(axle_x: float) -> None:
    assert (
        build_chassis_pose(
            _datum(0.0, 0.0),
            _datum(0.0, 0.0),
            wheelbase=WHEELBASE,
            axle_position=AxlePosition.FRONT,
            assumption=PoseAssumption.PURE_HEAVE,
            axle_x=axle_x,
        )
        is None
    )


@pytest.mark.parametrize("wheelbase", [0.0, -2500.0, float("nan"), float("inf")])
def test_degenerate_wheelbase_only_defeats_the_pitching_assumption(
    wheelbase: float,
) -> None:
    arguments = {
        "wheelbase": wheelbase,
        "axle_position": AxlePosition.FRONT,
        "axle_x": AXLE_X,
    }

    assert (
        build_chassis_pose(
            _datum(10.0, 10.0),
            _datum(0.0, 0.0),
            assumption=PoseAssumption.OPPOSITE_AXLE_FIXED,
            **arguments,
        )
        is None
    )
    heave_pose = build_chassis_pose(
        _datum(10.0, 10.0),
        _datum(0.0, 0.0),
        assumption=PoseAssumption.PURE_HEAVE,
        **arguments,
    )
    assert heave_pose is not None
    assert heave_pose.pitch_deg == 0.0


@pytest.mark.parametrize("assumption", list(PoseAssumption))
def test_vertical_ground_datum_has_no_centreline_height_and_no_pose(
    assumption: PoseAssumption,
) -> None:
    vertical = GroundDatum(-1.0, 0.0, 0.0)
    assert vertical.z_at(0.0) is None
    flat = _datum(0.0, 0.0)

    for ground, design_ground in ((vertical, flat), (flat, vertical)):
        assert (
            build_chassis_pose(
                ground,
                design_ground,
                wheelbase=WHEELBASE,
                axle_position=AxlePosition.FRONT,
                assumption=assumption,
                axle_x=AXLE_X,
            )
            is None
        )


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg"),
    [
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
    ],
)
def test_chassis_pose_rejects_non_finite_angles(
    roll_deg: float, pitch_deg: float
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ChassisPose(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            anchor=Point3((AXLE_X, 0.0, 0.0)),
            assumption=PoseAssumption.PURE_HEAVE,
        )


@pytest.mark.parametrize("assumption", list(PoseAssumption))
def test_solved_axle_state_yields_a_pose_anchored_on_the_ground_line(
    test_data_dir: Path,
    assumption: PoseAssumption,
) -> None:
    pytest.importorskip("yaml", reason="Axle fixtures load through the CLI adapter.")
    from kinematics.cli.io.loaders import load_geometry
    from kinematics.cli.io.sweep_loader import load_sweep
    from kinematics.core.suspensions.axle import AxleSuspension
    from kinematics.core.sweep import solve_sweep

    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    sweep = load_sweep(test_data_dir / "axle_sweep.yaml", axle)

    states, stats = solve_sweep(axle, sweep)
    assert all(info.converged for info in stats)

    state = states[-1]
    pose = chassis_pose_for_axle_state(axle, state, assumption)

    assert pose is not None
    assert pose.assumption is assumption
    assert np.isfinite(pose.roll_deg)
    assert np.isfinite(pose.pitch_deg)

    ground = GroundDatum.from_wheel_ground_tangents(
        state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)),
        state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT)),
    )
    assert ground is not None
    assert ground.signed_distance(pose.anchor) == pytest.approx(0.0, abs=1e-9)
    assert pose.roll_deg == pytest.approx(-ground.angle_deg)
    assert pose.anchor.data[0] == pytest.approx(
        0.5
        * (
            state.get(PointRef(Side.LEFT, PointID.WHEEL_CENTER)).data[0]
            + state.get(PointRef(Side.RIGHT, PointID.WHEEL_CENTER)).data[0]
        )
    )
    assert pose.to_world(pose.anchor).data == pytest.approx((0.0, 0.0, 0.0))


def test_pure_heave_pose_of_a_solved_axle_state_has_no_pitch(
    test_data_dir: Path,
) -> None:
    pytest.importorskip("yaml", reason="Axle fixtures load through the CLI adapter.")
    from kinematics.cli.io.loaders import load_geometry
    from kinematics.core.suspensions.axle import AxleSuspension

    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)

    design_pose = chassis_pose_for_axle_state(
        axle, axle.initial_state(), PoseAssumption.PURE_HEAVE
    )

    # The design state is its own pitch reference, so both assumptions agree.
    opposite_axle_pose = chassis_pose_for_axle_state(
        axle, axle.initial_state(), PoseAssumption.OPPOSITE_AXLE_FIXED
    )
    assert design_pose is not None
    assert opposite_axle_pose is not None
    assert design_pose.pitch_deg == 0.0
    assert opposite_axle_pose.pitch_deg == pytest.approx(0.0, abs=1e-9)
    assert design_pose.roll_deg == pytest.approx(0.0, abs=1e-9)
