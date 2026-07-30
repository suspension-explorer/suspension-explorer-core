"""Unit tests for the flat, level WorldSpace transform."""

from __future__ import annotations

import numpy as np
import pytest

from kinematics.core.pose import WorldSpace
from kinematics.core.primitives.geometry import Direction3, Point3

ORIGIN = Point3((1500.0, 0.0, 25.0))


def test_identity_space_maps_origin_and_preserves_distance() -> None:
    space = WorldSpace(
        x=Direction3((1.0, 0.0, 0.0)),
        y=Direction3((0.0, 1.0, 0.0)),
        z=Direction3((0.0, 0.0, 1.0)),
        origin=ORIGIN,
    )

    np.testing.assert_allclose(space.to_world(ORIGIN).data, (0.0, 0.0, 0.0))
    probe = Point3((1234.0, -456.0, 789.0))
    assert np.linalg.norm(space.to_world(probe).data) == pytest.approx(
        np.linalg.norm((probe - ORIGIN).data)
    )
    np.testing.assert_allclose(space.gravity.data, (0.0, 0.0, -1.0))


def test_world_gravity_is_always_minus_world_z() -> None:
    up = Direction3((-0.2, 0.1, 0.97))
    forward = Direction3((1.0, 0.0, float(-up[0] / up[2])))
    lateral = Direction3(up.cross(forward))
    space = WorldSpace(x=forward, y=lateral, z=up, origin=ORIGIN)

    np.testing.assert_allclose(
        space.rotation_chassis_to_world @ space.gravity.data,
        (0.0, 0.0, -1.0),
        atol=1e-12,
    )


def test_direct_construction_rejects_a_non_orthonormal_triad() -> None:
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        WorldSpace(
            x=Direction3((1.0, 0.0, 0.0)),
            y=Direction3((1.0, 1e-3, 0.0)),
            z=Direction3((0.0, 0.0, 1.0)),
            origin=ORIGIN,
        )


def test_direct_construction_rejects_a_left_handed_triad() -> None:
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        WorldSpace(
            x=Direction3((1.0, 0.0, 0.0)),
            y=Direction3((0.0, -1.0, 0.0)),
            z=Direction3((0.0, 0.0, 1.0)),
            origin=ORIGIN,
        )
