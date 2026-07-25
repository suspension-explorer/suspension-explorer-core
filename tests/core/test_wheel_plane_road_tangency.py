"""Tests for coupled axle wheel-plane/road tangency geometry."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.points.derived.manager import PositionValue
from kinematics.core.points.derived.road import (
    _road_normal,
    _shared_road_angle,
    get_axle_wheel_plane_road_tangent,
)
from kinematics.core.primitives.dual import DualScalar, DualVec3
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension

RADIUS_MM = 200.0


def _keys() -> dict[str, PointRef]:
    return {
        "left_center": PointRef(Side.LEFT, PointID.WHEEL_CENTER),
        "left_inboard": PointRef(Side.LEFT, PointID.AXLE_INBOARD),
        "left_outboard": PointRef(Side.LEFT, PointID.AXLE_OUTBOARD),
        "right_center": PointRef(Side.RIGHT, PointID.WHEEL_CENTER),
        "right_inboard": PointRef(Side.RIGHT, PointID.AXLE_INBOARD),
        "right_outboard": PointRef(Side.RIGHT, PointID.AXLE_OUTBOARD),
    }


def _positions() -> dict[PointRef, Point3]:
    keys = _keys()
    return {
        keys["left_center"]: Point3((0.0, 500.0, 300.0)),
        keys["left_inboard"]: Point3((0.0, 400.0, 300.0)),
        keys["left_outboard"]: Point3((0.0, 600.0, 300.0)),
        keys["right_center"]: Point3((0.0, -500.0, 200.0)),
        keys["right_inboard"]: Point3((0.0, -600.0, 200.0)),
        keys["right_outboard"]: Point3((0.0, -400.0, 200.0)),
    }


def _tangent(positions: dict[PointRef, Point3], side: str) -> Point3:
    keys = _keys()
    tangent = get_axle_wheel_plane_road_tangent(
        positions,
        left_center=keys["left_center"],
        left_axis_inboard=keys["left_inboard"],
        left_axis_outboard=keys["left_outboard"],
        left_radius=RADIUS_MM,
        right_center=keys["right_center"],
        right_axis_inboard=keys["right_inboard"],
        right_axis_outboard=keys["right_outboard"],
        right_radius=RADIUS_MM,
        side=side,
    )
    assert isinstance(tangent, Point3)
    return tangent


def test_coupled_tangents_lie_on_one_road_plane_and_each_wheel_plane():
    positions = _positions()
    left = _tangent(positions, "left")
    right = _tangent(positions, "right")
    keys = _keys()

    left_center = positions[keys["left_center"]].data
    right_center = positions[keys["right_center"]].data
    left_axis = np.array((0.0, 1.0, 0.0))
    right_axis = np.array((0.0, 1.0, 0.0))
    angle = _shared_road_angle(
        left_center,
        left_axis,
        RADIUS_MM,
        right_center,
        right_axis,
        RADIUS_MM,
    )
    assert isinstance(angle, float)
    normal = _road_normal(angle)
    assert isinstance(normal, np.ndarray)

    np.testing.assert_allclose(
        np.dot(left.data - left_center, left_axis), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.dot(right.data - right_center, right_axis), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(np.linalg.norm(left.data - left_center), RADIUS_MM)
    np.testing.assert_allclose(np.linalg.norm(right.data - right_center), RADIUS_MM)
    np.testing.assert_allclose(np.dot(normal, left.data), np.dot(normal, right.data))
    assert angle == pytest.approx(-np.arctan(0.1))


def test_dual_road_angle_uses_implicit_shared_plane_derivative():
    left_center = DualVec3(np.array((0.0, 500.0, 200.0)), np.array((0.0, 0.0, 1.0)))
    right_center = DualVec3(np.array((0.0, -500.0, 200.0)))
    left_axis = DualVec3(np.array((0.0, 1.0, 0.0)))
    right_axis = DualVec3(np.array((0.0, 1.0, 0.0)))

    angle = _shared_road_angle(
        left_center,
        left_axis,
        RADIUS_MM,
        right_center,
        right_axis,
        RADIUS_MM,
    )

    assert isinstance(angle, DualScalar)
    assert angle.val == pytest.approx(0.0)
    assert angle.deriv == pytest.approx(-1.0 / 1000.0)


def test_coupled_tangents_handle_camber_and_toe_axes():
    positions = _positions()
    keys = _keys()
    positions[keys["left_inboard"]] = Point3((-20.0, 404.0, 280.0))
    positions[keys["left_outboard"]] = Point3((20.0, 596.0, 320.0))
    positions[keys["right_inboard"]] = Point3((30.0, -595.0, 225.0))
    positions[keys["right_outboard"]] = Point3((-30.0, -405.0, 175.0))
    left = _tangent(positions, "left")
    right = _tangent(positions, "right")
    left_center = positions[keys["left_center"]].data
    right_center = positions[keys["right_center"]].data
    left_axis = (
        positions[keys["left_outboard"]].data - positions[keys["left_inboard"]].data
    )
    right_axis = (
        positions[keys["right_outboard"]].data - positions[keys["right_inboard"]].data
    )
    left_axis /= np.linalg.norm(left_axis)
    right_axis /= np.linalg.norm(right_axis)
    angle = _shared_road_angle(
        left_center,
        left_axis,
        RADIUS_MM,
        right_center,
        right_axis,
        RADIUS_MM,
    )
    assert isinstance(angle, float)
    normal = _road_normal(angle)
    assert isinstance(normal, np.ndarray)

    np.testing.assert_allclose(
        np.dot(left.data - left_center, left_axis), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.dot(right.data - right_center, right_axis), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(np.linalg.norm(left.data - left_center), RADIUS_MM)
    np.testing.assert_allclose(np.linalg.norm(right.data - right_center), RADIUS_MM)
    np.testing.assert_allclose(np.dot(normal, left.data), np.dot(normal, right.data))


def test_axle_derived_spec_replaces_corner_flat_tangents(test_data_dir):
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    spec = axle.derived_spec()
    left_tangent = PointRef(Side.LEFT, PointID.WHEEL_PLANE_ROAD_TANGENT)
    right_center = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
    assert right_center in spec.dependencies[left_tangent]

    state = axle.initial_state()
    expected = spec.functions[left_tangent](
        cast(dict[PointKey, PositionValue], state.positions)
    )
    assert isinstance(expected, Point3)
    assert state.get(left_tangent).almost_equals(expected, atol=1e-9)


def test_plain_coincident_wheels_are_rejected_as_nonunique_road_geometry():
    positions = _positions()
    keys = _keys()
    positions[keys["right_center"]] = Point3((0.0, 500.0, 300.0))
    positions[keys["right_inboard"]] = Point3((0.0, 400.0, 300.0))
    positions[keys["right_outboard"]] = Point3((0.0, 600.0, 300.0))

    with pytest.raises(ValueError, match="collapsed track"):
        _tangent(positions, "left")
