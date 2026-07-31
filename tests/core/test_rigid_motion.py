import numpy as np
import pytest

from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.rigid_motion import (
    ScrewAxisStatus,
    compute_upright_screw_axis,
    extract_screw_axis,
    fit_rigid_body_twist,
)
from kinematics.core.sensitivity import TangentField
from kinematics.core.targeting import PointTarget, PointTargetAxis

POINT_KEYS: tuple[PointKey, ...] = (
    PointID.LOWER_WISHBONE_OUTBOARD,
    PointID.UPPER_WISHBONE_OUTBOARD,
    PointID.TRACKROD_OUTBOARD,
    PointID.WHEEL_CENTER,
    PointID.STRUT_BOTTOM,
)
POINTS = np.array(
    [
        [12.0, -8.0, 5.0],
        [-3.0, 7.0, 11.0],
        [9.0, 4.0, -6.0],
        [-5.0, -2.0, 3.0],
        [2.0, 13.0, 1.0],
    ]
)


def _tangent(velocities: dict[PointKey, np.ndarray]) -> TangentField:
    target = PointTarget(
        point_id=PointID.TRACKROD_INBOARD,
        direction=PointTargetAxis(Axis.X),
        value=0.0,
        mode=TargetPositionMode.ABSOLUTE,
    )
    return TangentField(target_index=0, target=target, velocities=velocities)


def _motion(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    angular_rate: float,
    pitch: float,
) -> tuple[dict[PointKey, Point3], TangentField]:
    direction = axis_direction / np.linalg.norm(axis_direction)
    angular_velocity = angular_rate * direction
    velocities = (
        np.cross(angular_velocity, points - axis_point) + pitch * angular_velocity
    )
    positions = {key: Point3(point) for key, point in zip(POINT_KEYS, points)}
    tangent = _tangent({key: velocity for key, velocity in zip(POINT_KEYS, velocities)})
    return positions, tangent


def _line_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
    direction: np.ndarray,
) -> float:
    return float(np.linalg.norm(np.cross(point_a - point_b, direction)))


def test_fit_recovers_arbitrary_pure_rotation() -> None:
    axis_point = np.array([4.0, -1.5, 8.0])
    direction = np.array([2.0, -3.0, 5.0])
    positions, tangent = _motion(POINTS, axis_point, direction, 0.35, 0.0)

    twist = fit_rigid_body_twist(positions, tangent, POINT_KEYS)
    axis, status, message = extract_screw_axis(twist)

    assert status is ScrewAxisStatus.VALID
    assert message is None
    assert axis is not None
    expected_direction = direction / np.linalg.norm(direction)
    assert np.dot(axis.direction.data, expected_direction) == pytest.approx(1.0)
    assert _line_distance(axis.point.data, axis_point, expected_direction) < 1e-12
    assert axis.pitch == pytest.approx(0.0, abs=1e-12)
    assert axis.angular_rate == pytest.approx(0.35)
    assert axis.fit_rms < 1e-12
    assert axis.fit_max < 1e-12


def test_fit_recovers_nonzero_pitch_screw() -> None:
    axis_point = np.array([-7.0, 4.0, 3.0])
    direction = np.array([1.0, 4.0, -2.0])
    positions, tangent = _motion(POINTS, axis_point, direction, 0.12, 7.5)

    axis, status, _message = extract_screw_axis(
        fit_rigid_body_twist(positions, tangent, POINT_KEYS)
    )

    assert status is ScrewAxisStatus.VALID
    assert axis is not None
    expected_direction = direction / np.linalg.norm(direction)
    assert _line_distance(axis.point.data, axis_point, expected_direction) < 1e-12
    assert axis.pitch == pytest.approx(7.5, abs=1e-12)


def test_axis_is_invariant_to_translated_reference_coordinates() -> None:
    axis_point = np.array([4.0, -1.5, 8.0])
    direction = np.array([2.0, -3.0, 5.0])
    translation = np.array([3_000.0, -70.0, 1_200.0])
    positions, tangent = _motion(POINTS, axis_point, direction, 0.35, 1.25)
    translated_positions, translated_tangent = _motion(
        POINTS + translation,
        axis_point + translation,
        direction,
        0.35,
        1.25,
    )

    axis, status, _message = extract_screw_axis(
        fit_rigid_body_twist(positions, tangent, POINT_KEYS)
    )
    translated_axis, translated_status, _message = extract_screw_axis(
        fit_rigid_body_twist(translated_positions, translated_tangent, POINT_KEYS)
    )

    assert status is ScrewAxisStatus.VALID
    assert translated_status is ScrewAxisStatus.VALID
    assert axis is not None
    assert translated_axis is not None
    assert translated_axis.point.data == pytest.approx(axis.point.data + translation)
    assert translated_axis.direction.data == pytest.approx(axis.direction.data)
    assert translated_axis.pitch == pytest.approx(axis.pitch)


def test_axis_is_invariant_to_tangent_derivative_scale() -> None:
    axis_point = np.array([4.0, -1.5, 8.0])
    direction = np.array([2.0, -3.0, 5.0])
    positions, tangent = _motion(POINTS, axis_point, direction, 0.35, 1.25)
    scaled_tangent = _tangent(
        {key: 17.0 * velocity for key, velocity in tangent.velocities.items()}
    )

    axis, status, _message = extract_screw_axis(
        fit_rigid_body_twist(positions, tangent, POINT_KEYS)
    )
    scaled_axis, scaled_status, _message = extract_screw_axis(
        fit_rigid_body_twist(positions, scaled_tangent, POINT_KEYS)
    )

    assert status is ScrewAxisStatus.VALID
    assert scaled_status is ScrewAxisStatus.VALID
    assert axis is not None
    assert scaled_axis is not None
    assert scaled_axis.point.data == pytest.approx(axis.point.data)
    assert scaled_axis.direction.data == pytest.approx(axis.direction.data)
    assert scaled_axis.pitch == pytest.approx(axis.pitch)
    assert scaled_axis.angular_rate == pytest.approx(17.0 * axis.angular_rate)


def test_fit_accepts_a_combined_velocity_mapping() -> None:
    positions, tangent = _motion(
        POINTS,
        np.array([4.0, -1.5, 8.0]),
        np.array([2.0, -3.0, 5.0]),
        0.35,
        1.25,
    )

    axis, status, _message = extract_screw_axis(
        fit_rigid_body_twist(positions, tangent.velocities, POINT_KEYS)
    )

    assert status is ScrewAxisStatus.VALID
    assert axis is not None


def test_twist_reconstructs_selected_point_velocities() -> None:
    axis_point = np.array([4.0, -1.5, 8.0])
    positions, tangent = _motion(
        POINTS,
        axis_point,
        np.array([2.0, -3.0, 5.0]),
        0.35,
        1.25,
    )
    twist = fit_rigid_body_twist(positions, tangent, POINT_KEYS)

    assert twist.valid
    assert twist.reference_point is not None
    assert twist.reference_velocity is not None
    assert twist.angular_velocity is not None
    for key in POINT_KEYS:
        offset = positions[key].data - twist.reference_point.data
        recovered = twist.reference_velocity.data + np.cross(
            twist.angular_velocity.data,
            offset,
        )
        assert recovered == pytest.approx(tangent.velocities[key], abs=1e-12)


def test_overdetermined_fit_reports_small_injected_noise() -> None:
    positions, tangent = _motion(
        POINTS,
        np.array([4.0, -1.5, 8.0]),
        np.array([2.0, -3.0, 5.0]),
        0.35,
        0.0,
    )
    noisy_velocities = {
        key: velocity.copy() for key, velocity in tangent.velocities.items()
    }
    noisy_velocities[POINT_KEYS[-1]] += np.array([1e-4, -2e-4, 3e-4])
    noisy_tangent = _tangent(noisy_velocities)

    twist = fit_rigid_body_twist(positions, noisy_tangent, POINT_KEYS)
    axis, status, _message = extract_screw_axis(twist)

    assert twist.valid
    assert 0.0 < twist.fit_rms < 1e-3
    assert twist.fit_max < 1e-3
    assert status is ScrewAxisStatus.EXCESSIVE_FIT_ERROR
    assert axis is None


def test_collinear_points_return_degenerate_upright_status() -> None:
    collinear_points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    keys = POINT_KEYS[:3]
    positions, tangent = _motion(
        collinear_points,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        0.3,
        0.0,
    )

    result = compute_upright_screw_axis("upright", positions, tangent, keys)

    assert result.axis is None
    assert result.status is ScrewAxisStatus.DEGENERATE_UPRIGHT
    assert result.point_count == 3


def test_pure_translation_returns_no_finite_axis() -> None:
    positions = {key: Point3(point) for key, point in zip(POINT_KEYS, POINTS)}
    tangent = _tangent({key: np.array([2.0, -1.0, 3.0]) for key in POINT_KEYS})

    result = compute_upright_screw_axis("upright", positions, tangent, POINT_KEYS)

    assert result.axis is None
    assert result.status is ScrewAxisStatus.NEAR_PURE_TRANSLATION
