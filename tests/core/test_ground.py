"""Tests for the shared road-plane primitive."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.road import RoadPlane


def test_plane_through_point_has_full_3d_normal() -> None:
    normal = Direction3((-0.2, 0.1, 0.97))
    point = Point3((100.0, -20.0, 30.0))
    ground = RoadPlane.through(normal, point)

    assert ground.signed_distance(point) == pytest.approx(0.0)
    assert ground.signed_distance(point + normal * 12.0) == pytest.approx(12.0)
    assert ground.forward.dot(normal) == pytest.approx(0.0, abs=1e-12)
    assert ground.lateral.dot(normal) == pytest.approx(0.0, abs=1e-12)
    assert ground.forward.cross(ground.lateral).dot(normal) == pytest.approx(1.0)


def test_horizontal_datum_uses_local_tangent_height() -> None:
    ground = RoadPlane.horizontal_at(Point3((123.0, 456.0, 78.0)))

    np.testing.assert_allclose(ground.normal.data, (0.0, 0.0, 1.0))
    assert ground.signed_distance(Point3((0.0, 0.0, 80.0))) == pytest.approx(2.0)
    np.testing.assert_allclose(ground.forward.data, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(ground.lateral.data, (0.0, 1.0, 0.0))


def test_axle_tangents_define_one_zero_grade_road_plane() -> None:
    left = Point3((30.0, 2.0, 1.0))
    right = Point3((-40.0, -2.0, -1.0))

    road = RoadPlane.from_axle_contact_centres(left, right)

    np.testing.assert_allclose(
        road.normal.data,
        (0.0, -1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)),
    )
    assert road.signed_distance(left) == pytest.approx(0.0)
    assert road.signed_distance(right) == pytest.approx(0.0)


def test_collapsed_axle_tangents_do_not_define_a_road_plane() -> None:
    tangent = Point3((10.0, 20.0, 30.0))

    with pytest.raises(ValueError, match="do not define"):
        RoadPlane.from_axle_contact_centres(tangent, tangent)


def test_road_plane_is_immutable() -> None:
    ground = RoadPlane.horizontal_at(Point3((0.0, 0.0, 0.0)))
    with pytest.raises(FrozenInstanceError):
        setattr(ground, "offset_mm", 1.0)


@pytest.mark.parametrize(
    ("normal", "offset"),
    [
        ((0.0, 0.0, -1.0), 0.0),
        ((1.0, 0.0, 0.0), 0.0),
        ((0.0, 0.0, 1.0), np.inf),
        ((np.nan, 0.0, 1.0), 0.0),
    ],
)
def test_direct_construction_rejects_invalid_plane(normal, offset) -> None:
    with pytest.raises(ValueError):
        RoadPlane(Direction3(normal), offset)
