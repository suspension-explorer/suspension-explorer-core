"""Tests for coupled axle wheel-plane/ground tangency geometry."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.metrics.ground import AxleGroundLine
from kinematics.core.points.derived import ground
from kinematics.core.points.derived.ground import (
    _GROUND_SOLVE_LIMIT,
    _flat_ground_normal_angle_estimate,
    _ground_normal,
    _GroundNormalContinuation,
    _shared_ground_normal_angle,
    _solve_ground_normal_angle,
    get_axle_wheel_ground_tangent,
)
from kinematics.core.points.derived.manager import PositionValue
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


def _tangent(
    positions: dict[PointRef, Point3],
    side: Literal["left", "right"],
    continuation: _GroundNormalContinuation | None = None,
) -> Point3:
    keys = _keys()
    tangent = get_axle_wheel_ground_tangent(
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
        continuation=continuation,
    )
    assert isinstance(tangent, Point3)
    return tangent


def test_coupled_tangents_lie_on_one_ground_plane_and_each_wheel_plane():
    positions = _positions()
    left = _tangent(positions, "left")
    right = _tangent(positions, "right")
    keys = _keys()

    left_center = positions[keys["left_center"]].data
    right_center = positions[keys["right_center"]].data
    left_axis = np.array((0.0, 1.0, 0.0))
    right_axis = np.array((0.0, 1.0, 0.0))
    angle = _shared_ground_normal_angle(
        left_center,
        left_axis,
        RADIUS_MM,
        right_center,
        right_axis,
        RADIUS_MM,
    )
    assert isinstance(angle, float)
    normal = _ground_normal(angle)
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


def test_dual_ground_angle_uses_implicit_shared_plane_derivative():
    left_center = DualVec3(np.array((0.0, 500.0, 200.0)), np.array((0.0, 0.0, 1.0)))
    right_center = DualVec3(np.array((0.0, -500.0, 200.0)))
    left_axis = DualVec3(np.array((0.0, 1.0, 0.0)))
    right_axis = DualVec3(np.array((0.0, 1.0, 0.0)))

    angle = _shared_ground_normal_angle(
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
    angle = _shared_ground_normal_angle(
        left_center,
        left_axis,
        RADIUS_MM,
        right_center,
        right_axis,
        RADIUS_MM,
    )
    assert isinstance(angle, float)
    normal = _ground_normal(angle)
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
    left_tangent = PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)
    right_center = PointRef(Side.RIGHT, PointID.WHEEL_CENTER)
    assert right_center in spec.dependencies[left_tangent]

    state = axle.initial_state()
    expected = spec.functions[left_tangent](
        cast(dict[PointKey, PositionValue], state.positions)
    )
    assert isinstance(expected, Point3)
    assert state.get(left_tangent).almost_equals(expected, atol=1e-9)


def test_plain_coincident_wheels_are_rejected_as_nonunique_ground_geometry():
    positions = _positions()
    keys = _keys()
    positions[keys["right_center"]] = Point3((0.0, 500.0, 300.0))
    positions[keys["right_inboard"]] = Point3((0.0, 400.0, 300.0))
    positions[keys["right_outboard"]] = Point3((0.0, 600.0, 300.0))

    with pytest.raises(ValueError, match="collapsed track"):
        _tangent(positions, "left")


# A narrow, tall, heavily toed axle whose flat-ground seed saturates the clamp while
# its true shared-plane root sits comfortably inside the resolvable domain.
_SATURATING_LEFT_CENTER = np.array((0.0, 50.0, 800.0))
_SATURATING_RIGHT_CENTER = np.array((0.0, -50.0, 0.0))
_SATURATING_LEFT_AXIS = np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0)
_SATURATING_RIGHT_AXIS = np.array((0.0, -1.0, 0.0))
_SATURATING_RADIUS_MM = 400.0


def _solve_saturating() -> float:
    return _solve_ground_normal_angle(
        _SATURATING_LEFT_CENTER,
        _SATURATING_LEFT_AXIS,
        _SATURATING_RADIUS_MM,
        _SATURATING_RIGHT_CENTER,
        _SATURATING_RIGHT_AXIS,
        _SATURATING_RADIUS_MM,
    )


def test_saturated_flat_ground_seed_still_solves():
    seed = _flat_ground_normal_angle_estimate(
        _SATURATING_LEFT_CENTER,
        _SATURATING_LEFT_AXIS,
        _SATURATING_RADIUS_MM,
        _SATURATING_RIGHT_CENTER,
        _SATURATING_RIGHT_AXIS,
        _SATURATING_RADIUS_MM,
    )
    # The seed must genuinely be clamped, or the test would not exercise the
    # boundary at all.
    assert seed == pytest.approx(-_GROUND_SOLVE_LIMIT)

    angle = _solve_saturating()

    assert abs(angle) < _GROUND_SOLVE_LIMIT
    residual = ground._shared_plane_residual(
        _SATURATING_LEFT_CENTER,
        _SATURATING_LEFT_AXIS,
        _SATURATING_RADIUS_MM,
        _SATURATING_RIGHT_CENTER,
        _SATURATING_RIGHT_AXIS,
        _SATURATING_RADIUS_MM,
        angle,
    )
    assert isinstance(residual, float)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_flat_ground_seed_is_robust_to_crossed_lateral_orientation():
    # Left support below right in Y: the raw atan2 lands near -pi and the clamp
    # would otherwise hand the solver a seed unrelated to the geometry.
    crossed_left_center = np.array((0.0, -500.0, 300.0))
    crossed_right_center = np.array((0.0, 500.0, 200.0))
    axis = np.array((0.0, 1.0, 0.0))

    seed = _flat_ground_normal_angle_estimate(
        crossed_left_center, axis, RADIUS_MM, crossed_right_center, axis, RADIUS_MM
    )
    angle = _solve_ground_normal_angle(
        crossed_left_center, axis, RADIUS_MM, crossed_right_center, axis, RADIUS_MM
    )

    assert seed == pytest.approx(np.arctan(0.1))
    assert angle == pytest.approx(np.arctan(0.1))


def test_shared_angle_is_solved_once_for_both_sides(monkeypatch):
    calls: list[float] = []
    original = ground._search_ground_normal_angle

    def counting_search(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(ground, "_search_ground_normal_angle", counting_search)

    positions = _positions()
    continuation = _GroundNormalContinuation()
    left = _tangent(positions, "left", continuation)
    right = _tangent(positions, "right", continuation)

    assert len(calls) == 1
    # The reused scalar must still place both sides on one plane.
    normal = _ground_normal(calls[0])
    assert isinstance(normal, np.ndarray)
    np.testing.assert_allclose(np.dot(normal, left.data), np.dot(normal, right.data))


def test_continuation_uses_previous_root_as_next_state_seed(monkeypatch):
    seeds: list[float | None] = []

    def fake_search(*args, seed=None):
        seeds.append(seed)
        return 0.25 if seed is None else seed + 0.01

    monkeypatch.setattr(ground, "_search_ground_normal_angle", fake_search)
    continuation = _GroundNormalContinuation()
    axis = np.array((0.0, 1.0, 0.0))

    first = continuation.solve(
        np.array((0.0, 500.0, 300.0)),
        axis,
        RADIUS_MM,
        np.array((0.0, -500.0, 200.0)),
        axis,
        RADIUS_MM,
    )
    second = continuation.solve(
        np.array((0.0, 500.0, 301.0)),
        axis,
        RADIUS_MM,
        np.array((0.0, -500.0, 200.0)),
        axis,
        RADIUS_MM,
    )

    assert first == pytest.approx(0.25)
    assert second == pytest.approx(0.26)
    assert seeds == [None, pytest.approx(0.25)]


def test_internal_normal_angle_negates_the_public_ground_line_angle():
    positions = _positions()
    left = _tangent(positions, "left")
    right = _tangent(positions, "right")
    ground_line = AxleGroundLine.from_wheel_ground_tangents(left, right)
    assert ground_line is not None

    angle = _shared_ground_normal_angle(
        positions[_keys()["left_center"]].data,
        np.array((0.0, 1.0, 0.0)),
        RADIUS_MM,
        positions[_keys()["right_center"]].data,
        np.array((0.0, 1.0, 0.0)),
        RADIUS_MM,
    )
    assert isinstance(angle, float)
    assert ground_line.angle_deg == pytest.approx(-np.degrees(angle))
