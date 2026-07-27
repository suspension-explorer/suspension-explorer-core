"""Unit tests for world-space construction from a gravity direction."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np
import pytest

from kinematics.core.pose import GravityModel, WorldSpace
from kinematics.core.primitives.geometry import Direction3, Point3

ANCHOR = Point3((1500.0, 0.0, 25.0))


def test_straight_down_gravity_reproduces_the_chassis_axes() -> None:
    space = WorldSpace.from_gravity((0.0, 0.0, -1.0), anchor=ANCHOR)

    assert space is not None
    np.testing.assert_allclose(space.x.data, (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(space.y.data, (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(space.z.data, (0.0, 0.0, 1.0), atol=1e-12)
    assert space.gravity_model is None


def test_world_up_opposes_gravity_and_heading_projects_chassis_forward() -> None:
    bank = radians(12.0)
    gravity = np.array((0.0, sin(bank), -cos(bank)))

    space = WorldSpace.from_gravity(gravity, anchor=ANCHOR)

    assert space is not None
    np.testing.assert_allclose(space.z.data, -gravity, atol=1e-12)
    np.testing.assert_allclose(space.gravity.data, gravity, atol=1e-12)
    # Chassis forward has no component along this gravity, so it projects to
    # itself: heading is measured from the vehicle's own forward axis.
    np.testing.assert_allclose(space.x.data, (1.0, 0.0, 0.0), atol=1e-12)
    rotation = space.rotation_chassis_to_world
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_pitched_gravity_keeps_the_frame_exact_rather_than_small_angle() -> None:
    # A large fore-aft gravity component: the frame must stay exactly
    # orthonormal because construction never composes angle approximations.
    gravity = np.array((sin(radians(40.0)), 0.0, -cos(radians(40.0))))

    space = WorldSpace.from_gravity(gravity, anchor=ANCHOR)

    assert space is not None
    rotation = space.rotation_chassis_to_world
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(space.z.data, -gravity, atol=1e-12)
    # World forward stays in the chassis XZ plane and ahead of the vehicle.
    assert space.x[1] == pytest.approx(0.0, abs=1e-12)
    assert space.x[0] > 0.0


def test_to_world_maps_the_anchor_to_the_origin_and_preserves_distance() -> None:
    gravity = Direction3((0.1, -0.2, -0.9))
    space = WorldSpace.from_gravity(gravity, anchor=ANCHOR)
    assert space is not None

    origin = space.to_world(ANCHOR)
    np.testing.assert_allclose(origin.data, (0.0, 0.0, 0.0), atol=1e-12)

    probe = Point3((1234.0, -456.0, 789.0))
    mapped = space.to_world(probe)
    assert float(np.linalg.norm(mapped.data)) == pytest.approx(
        float(np.linalg.norm((probe - ANCHOR).data))
    )


@pytest.mark.parametrize(
    "gravity",
    [
        (0.0, 0.0, 0.0),
        (float("nan"), 0.0, -1.0),
        (float("inf"), 0.0, -1.0),
        (1.0, 0.0, 0.0),  # Chassis forward vertical: heading undefined.
        (-1.0, 0.0, 0.0),
    ],
)
def test_degenerate_gravity_returns_no_space(gravity) -> None:
    assert WorldSpace.from_gravity(gravity, anchor=ANCHOR) is None


def test_direct_construction_rejects_a_non_orthonormal_triad() -> None:
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        WorldSpace(
            x=Direction3((1.0, 0.0, 0.0)),
            y=Direction3((1.0, 1e-3, 0.0)),
            z=Direction3((0.0, 0.0, 1.0)),
            anchor=ANCHOR,
            gravity_model=None,
        )


def test_direct_construction_rejects_a_left_handed_triad() -> None:
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        WorldSpace(
            x=Direction3((1.0, 0.0, 0.0)),
            y=Direction3((0.0, -1.0, 0.0)),
            z=Direction3((0.0, 0.0, 1.0)),
            anchor=ANCHOR,
            gravity_model=None,
        )


def test_gravity_assumption_strings_coerce_and_reject() -> None:
    assert GravityModel("road_level") is GravityModel.ROAD_LEVEL
    assert GravityModel("chassis_level") is GravityModel.CHASSIS_LEVEL
    assert GravityModel("opposite_axle_fixed") is GravityModel.OPPOSITE_AXLE_FIXED
    with pytest.raises(ValueError):
        GravityModel("pure_heave")
