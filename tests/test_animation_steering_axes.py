"""Animation alignment and persistent steering-axis artist tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("matplotlib")

from kinematics.cli.visualization.animation import (  # noqa: E402
    aligned_animation_frames,
    pingpong_animation_frames,
)
from kinematics.cli.visualization.clipping import Bounds3D  # noqa: E402
from kinematics.cli.visualization.main import SuspensionVisualizer  # noqa: E402
from kinematics.core.enums import PointID  # noqa: E402
from kinematics.core.primitives.geometry import Point3, Vector3  # noqa: E402
from kinematics.core.screw_axis import (  # noqa: E402
    InstantaneousScrewAxis,
    ScrewAxisResult,
    ScrewAxisStatus,
)
from kinematics.core.steering_axis import (  # noqa: E402
    SteeringResponseAxisResult,
    SteeringResponseStatus,
)


def _axis_result(
    label: str,
    *,
    point: tuple[float, float, float] = (5.0, 5.0, 5.0),
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> SteeringResponseAxisResult:
    screw_axis = ScrewAxisResult(
        point_keys=(PointID.WHEEL_CENTER,),
        axis=InstantaneousScrewAxis(
            point=Point3(point),
            direction=Vector3(direction),
            pitch=0.0,
            angular_rate=1.0,
            fit_rms=0.0,
            fit_max=0.0,
        ),
        status=ScrewAxisStatus.VALID,
    )
    return SteeringResponseAxisResult(
        upright_label=label,
        point_keys=(PointID.WHEEL_CENTER,),
        screw_axis=screw_axis,
        status=SteeringResponseStatus.VALID,
    )


def _unavailable_result(label: str) -> SteeringResponseAxisResult:
    return SteeringResponseAxisResult(
        upright_label=label,
        point_keys=(PointID.WHEEL_CENTER,),
        status=SteeringResponseStatus.TANGENT_UNAVAILABLE,
    )


@dataclass
class _FakeLine:
    xy: tuple[list[float], list[float]] = field(default_factory=lambda: ([], []))
    z: list[float] = field(default_factory=list)

    def set_data(self, x, y) -> None:
        self.xy = (list(x), list(y))

    def set_3d_properties(self, z) -> None:
        self.z = list(z)


def test_pingpong_keeps_valid_unavailable_valid_axes_with_their_positions() -> None:
    positions = [{"p": (float(index), 0.0, 0.0)} for index in range(3)]
    axis_frames = [
        (_axis_result("Left"), _axis_result("Right")),
        (_unavailable_result("Left"), _unavailable_result("Right")),
        (_axis_result("Left"), _axis_result("Right")),
    ]

    frames = aligned_animation_frames(positions, axis_frames)
    pingpong = pingpong_animation_frames(frames)

    assert [frame.positions["p"][0] for frame in pingpong] == [0.0, 1.0, 2.0, 1.0]
    assert [frame.steering_response_axes[0].status for frame in pingpong] == [
        SteeringResponseStatus.VALID,
        SteeringResponseStatus.TANGENT_UNAVAILABLE,
        SteeringResponseStatus.VALID,
        SteeringResponseStatus.TANGENT_UNAVAILABLE,
    ]


def test_alignment_rejects_mismatched_axis_frame_count() -> None:
    with pytest.raises(ValueError, match="frame counts must match"):
        aligned_animation_frames([{"p": (0.0, 0.0, 0.0)}], [])


def test_persistent_artist_updates_valid_then_hides_unavailable_and_outside() -> None:
    artist = _FakeLine()
    bounds = Bounds3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))

    SuspensionVisualizer.update_steering_response_axes(
        [artist],
        ["Upright"],
        [_axis_result("Upright")],
        bounds,
    )
    assert artist.xy == ([5.0, 5.0], [5.0, 5.0])
    assert artist.z == [0.0, 10.0]

    SuspensionVisualizer.update_steering_response_axes(
        [artist],
        ["Upright"],
        [_unavailable_result("Upright")],
        bounds,
    )
    assert artist.xy == ([], [])
    assert artist.z == []

    SuspensionVisualizer.update_steering_response_axes(
        [artist],
        ["Upright"],
        [_axis_result("Upright", point=(20.0, 5.0, 5.0))],
        bounds,
    )
    assert artist.xy == ([], [])
    assert artist.z == []


def test_multiple_upright_artists_update_independently() -> None:
    left_artist = _FakeLine()
    right_artist = _FakeLine()

    SuspensionVisualizer.update_steering_response_axes(
        [left_artist, right_artist],
        ["Left", "Right"],
        [_axis_result("Right", direction=(1.0, 0.0, 0.0))],
        Bounds3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
    )

    assert left_artist.xy == ([], [])
    assert right_artist.xy == ([0.0, 10.0], [5.0, 5.0])
