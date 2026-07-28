"""Tests for coupled axle wheel-plane/ground tangency geometry."""

from __future__ import annotations

import numpy as np
import pytest

from kinematics.core.enums import PointID
from kinematics.core.input import build_suspension
from kinematics.core.points.derived import ground
from kinematics.core.points.derived.ground import (
    _GROUND_SOLVE_LIMIT,
    AxleGroundTangency,
    _flat_ground_normal_angle_estimate,
    _ground_normal,
    _search_ground_normal_angle,
    _shared_ground_normal_angle,
    seed_from_tangent_points,
    solve_axle_wheel_ground_tangents,
)
from kinematics.core.primitives.dual import DualScalar, DualVec3
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension

RADIUS_MM = 200.0


def _axle_suspension() -> AxleSuspension:
    """Build the test axle through the transport-neutral core input facade."""
    suspension = build_suspension(
        {
            "type": "double_wishbone",
            "scope": "axle",
            "name": "test axle",
            "version": "0.0.1",
            "units": "millimeters",
            "vehicle_config": {
                "cg_position": {"x": 1250, "y": 0, "z": 450},
                "wheelbase": 2500.0,
            },
            "axle_config": {
                "axle_position": "front",
                "steering": {"type": "rack"},
                "actuation": {"type": "direct", "mount": "lower_wishbone"},
                "spring": {"type": "none"},
                "anti_roll": {"type": "none"},
                "heave_link": {"type": "none"},
                "wheel": {
                    "offset": 0,
                    "tire": {
                        "aspect_ratio": 0.55,
                        "section_width": 270,
                        "rim_diameter": 13,
                    },
                },
            },
            "hardpoints": {
                "left": {
                    "lower_wishbone_inboard_front": {"x": 250, "y": 400, "z": 200},
                    "lower_wishbone_inboard_rear": {"x": -250, "y": 450, "z": 200},
                    "lower_wishbone_outboard": {"x": 0, "y": 900, "z": 200},
                    "upper_wishbone_inboard_front": {"x": 225, "y": 350, "z": 500},
                    "upper_wishbone_inboard_rear": {"x": -275, "y": 350, "z": 500},
                    "upper_wishbone_outboard": {"x": -25, "y": 750, "z": 500},
                    "trackrod_inboard": {"x": 50, "y": 200, "z": 250},
                    "trackrod_outboard": {"x": 150, "y": 800, "z": 275},
                    "axle_inboard": {"x": -20, "y": 800, "z": 308.426},
                    "axle_outboard": {"x": -20, "y": 950, "z": 313.426},
                }
            },
        }
    )
    assert isinstance(suspension, AxleSuspension)
    return suspension


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


def _solve(
    positions: dict[PointRef, Point3],
    seed: float | None = None,
) -> AxleGroundTangency:
    keys = _keys()
    return solve_axle_wheel_ground_tangents(
        positions,
        left_center=keys["left_center"],
        left_axis_inboard=keys["left_inboard"],
        left_axis_outboard=keys["left_outboard"],
        left_radius=RADIUS_MM,
        right_center=keys["right_center"],
        right_axis_inboard=keys["right_inboard"],
        right_axis_outboard=keys["right_outboard"],
        right_radius=RADIUS_MM,
        seed=seed,
    )


def test_coupled_tangents_lie_on_one_ground_plane_and_each_wheel_plane():
    positions = _positions()
    keys = _keys()
    tangency = _solve(positions)
    left = tangency.left
    right = tangency.right
    assert isinstance(left, Point3)
    assert isinstance(right, Point3)

    left_center = positions[keys["left_center"]].data
    right_center = positions[keys["right_center"]].data
    left_axis = np.array((0.0, 1.0, 0.0))
    right_axis = np.array((0.0, 1.0, 0.0))
    normal = _ground_normal(tangency.normal_angle)
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
    assert tangency.normal_angle == pytest.approx(-np.arctan(0.1))


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
    tangency = _solve(positions)
    left = tangency.left
    right = tangency.right
    assert isinstance(left, Point3)
    assert isinstance(right, Point3)

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
    normal = _ground_normal(tangency.normal_angle)
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


def test_solve_returns_one_shared_plane_and_is_stateless():
    positions = _positions()

    first = _solve(positions, seed=-0.05)
    repeat = _solve(_positions(), seed=-0.05)

    assert isinstance(first.normal_angle, float)
    normal = _ground_normal(first.normal_angle)
    assert isinstance(normal, np.ndarray)
    assert isinstance(first.left, Point3)
    assert isinstance(first.right, Point3)
    np.testing.assert_allclose(
        np.dot(normal, first.left.data), np.dot(normal, first.right.data)
    )
    # Statelessness: the only cross-call coupling is the explicit seed, so the
    # same inputs and seed must reproduce the result bit for bit.
    assert isinstance(repeat.left, Point3)
    assert isinstance(repeat.right, Point3)
    assert repeat.normal_angle == first.normal_angle, (
        "Identical inputs and seed must reproduce the same ground-normal angle"
    )
    np.testing.assert_array_equal(repeat.left.data, first.left.data)
    np.testing.assert_array_equal(repeat.right.data, first.right.data)


def test_seed_from_tangent_points_recovers_the_lateral_contact_angle():
    left = Point3((0.0, 620.0, -12.0))
    right = Point3((0.0, -580.0, 18.0))
    lateral_separation = 620.0 - (-580.0)
    vertical_separation = -12.0 - 18.0

    seed = seed_from_tangent_points(left, right)
    reversed_seed = seed_from_tangent_points(right, left)

    expected = -np.arctan2(vertical_separation, lateral_separation)
    assert seed == pytest.approx(expected)
    # The recovery orients the separation from vehicle right to left, so a
    # crossed argument order cannot flip the reported ground-line angle.
    assert reversed_seed == pytest.approx(expected)


def test_seed_from_tangent_points_matches_the_solved_ground_normal_angle():
    tangency = _solve(_positions())

    seed = seed_from_tangent_points(tangency.left, tangency.right)

    assert seed == pytest.approx(tangency.normal_angle)


def test_seed_from_tangent_points_returns_none_for_unusable_points():
    collapsed = seed_from_tangent_points(
        Point3((0.0, 100.0, 10.0)), Point3((0.0, 100.0, -10.0))
    )
    non_finite = seed_from_tangent_points(
        np.array((0.0, 600.0, np.nan)), np.array((0.0, -600.0, 0.0))
    )

    assert collapsed is None, "A collapsed track cannot orient a ground line"
    assert non_finite is None, "Non-finite tangents cannot orient a ground line"


def test_plain_coincident_wheels_are_rejected_as_nonunique_ground_geometry():
    positions = _positions()
    keys = _keys()
    positions[keys["right_center"]] = Point3((0.0, 500.0, 300.0))
    positions[keys["right_inboard"]] = Point3((0.0, 400.0, 300.0))
    positions[keys["right_outboard"]] = Point3((0.0, 600.0, 300.0))

    with pytest.raises(ValueError, match="collapsed track"):
        _solve(positions)


# A narrow, tall, heavily toed axle whose flat-ground seed saturates the clamp while
# its true shared-plane root sits comfortably inside the resolvable domain.
_SATURATING_LEFT_CENTER = np.array((0.0, 50.0, 800.0))
_SATURATING_RIGHT_CENTER = np.array((0.0, -50.0, 0.0))
_SATURATING_LEFT_AXIS = np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0)
_SATURATING_RIGHT_AXIS = np.array((0.0, -1.0, 0.0))
_SATURATING_RADIUS_MM = 400.0


def _solve_saturating() -> float:
    return _search_ground_normal_angle(
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
    angle = _search_ground_normal_angle(
        crossed_left_center, axis, RADIUS_MM, crossed_right_center, axis, RADIUS_MM
    )

    assert seed == pytest.approx(np.arctan(0.1))
    assert angle == pytest.approx(np.arctan(0.1))


def test_axle_derived_spec_omits_the_coupled_tangents() -> None:
    axle = _axle_suspension()

    spec = axle.derived_spec()

    for side in (Side.LEFT, Side.RIGHT):
        tangent = PointRef(side, PointID.WHEEL_GROUND_TANGENT)
        assert tangent not in spec.functions, (
            "Coupled wheel-ground tangents are closure outputs, not derived points"
        )
        assert tangent not in spec.dependencies


def test_ground_closure_reproduces_the_stored_initial_state_tangents() -> None:
    axle = _axle_suspension()
    state = axle.initial_state()
    positions = dict(state.positions)

    axle.apply_ground_closure(positions)

    for side in (Side.LEFT, Side.RIGHT):
        tangent = PointRef(side, PointID.WHEEL_GROUND_TANGENT)
        recomputed = positions[tangent]
        assert isinstance(recomputed, Point3)
        assert state.get(tangent).almost_equals(recomputed, atol=1e-9), (
            f"Closure must reproduce the stored {side.name.lower()} tangent"
        )


def test_ground_closure_recovers_its_seed_from_the_stored_tangents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axle = _axle_suspension()
    state = axle.initial_state()
    positions = dict(state.positions)
    stored_seed = seed_from_tangent_points(
        state.get(PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)),
        state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT)),
    )
    assert stored_seed is not None

    original_search = ground._search_ground_normal_angle
    recorded_seeds: list[float | None] = []

    def record_search(*args, seed=None):
        recorded_seeds.append(seed)
        return original_search(*args, seed=seed)

    monkeypatch.setattr(ground, "_search_ground_normal_angle", record_search)

    axle.apply_ground_closure(positions)

    assert len(recorded_seeds) == 1, "The shared plane is solved once per closure"
    assert recorded_seeds[0] is not None, (
        "The closure must recover a seed from the stored tangent values"
    )
    assert recorded_seeds[0] == pytest.approx(stored_seed)
