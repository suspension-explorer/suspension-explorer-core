"""
Animation utilities for suspension visualization.

This module provides animation functionality for suspension systems, making use of the
common plotting utilities from plots.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from kinematics.cli.visualization.clipping import Bounds3D
from kinematics.cli.visualization.main import SuspensionVisualizer
from kinematics.cli.visualization.plots import (
    compute_bounds_from_states,
    configure_3d_axis,
    create_four_view_axes,
)
from kinematics.core.screw_axis import UprightScrewAxisResult


@dataclass(frozen=True)
class AnimationFrame:
    """Position and steering-axis data that must remain frame-aligned."""

    positions: dict[str, tuple[float, float, float]]
    steering_response_axes: tuple[UprightScrewAxisResult, ...] = ()


def aligned_animation_frames(
    position_states: Sequence[dict[str, tuple[float, float, float]]],
    steering_response_axes: Sequence[Sequence[UprightScrewAxisResult]] | None = None,
) -> list[AnimationFrame]:
    """Bundle positions and axes before any animation sequencing is applied."""
    axis_frames = (
        [()] * len(position_states)
        if steering_response_axes is None
        else steering_response_axes
    )
    if len(position_states) != len(axis_frames):
        raise ValueError(
            "Animation position and steering-axis frame counts must match: "
            f"{len(position_states)} positions, {len(axis_frames)} axis frames."
        )
    return [
        AnimationFrame(positions, tuple(axes))
        for positions, axes in zip(position_states, axis_frames, strict=True)
    ]


def pingpong_animation_frames(frames: Sequence[AnimationFrame]) -> list[AnimationFrame]:
    """Return forward then interior-reverse playback without breaking alignment."""
    return [*frames, *frames[-2:0:-1]]


def create_animation(
    position_states: list[dict[str, tuple[float, float, float]]],
    initial_positions: dict[str, tuple[float, float, float]],
    visualizer: SuspensionVisualizer,
    output_path: Path,
    fps: int = 20,
    writer: str | None = None,
    codec: str = "libx264",
    dpi: int = 200,
    show_live: bool = True,
    steering_response_axes: Sequence[Sequence[UprightScrewAxisResult]] | None = None,
) -> None:
    """
    Create an animation showing suspension movement through multiple states.

    Args:
        position_states: List of position dictionaries for each frame.
        initial_positions: Initial position state for reference.
        visualizer: Suspension visualizer with links and wheel config.
        output_path: Path where the animation will be saved.
        fps: Frames per second for the animation.
        writer: Animation writer to use ('ffmpeg', 'pillow', etc.).
        codec: Video codec to use (for ffmpeg writer).
        dpi: DPI for the output animation.
        show_live: Whether to show the animation live during creation.
        steering_response_axes: Per-frame isolated upright axes aligned with positions.
    """
    frames = aligned_animation_frames(position_states, steering_response_axes)
    if not frames:
        raise ValueError("Animation requires at least one position state")

    # Create figure with four subplots using common function.
    fig, axes = create_four_view_axes()

    # Compute global bounds solely from suspension and wheel geometry.
    min_bounds, max_bounds, (x_mid, y_mid, z_mid, max_range) = (
        compute_bounds_from_states([frame.positions for frame in frames])
    )
    clipping_bounds = Bounds3D(min_bounds, max_bounds)

    # Configure axes once using common function.
    for view_name, ax in axes.items():
        configure_3d_axis(ax, view_name, x_mid, y_mid, z_mid, max_range)

    # Use unified draw_links to create link artists.
    link_artists: dict[str, list] = {k: [] for k in axes.keys()}
    for view_name, ax in axes.items():
        link_artists[view_name] = visualizer.draw_links(ax, initial_positions)

    upright_labels = tuple(
        dict.fromkeys(
            result.upright_label
            for frame in frames
            for result in frame.steering_response_axes
        )
    )
    steering_axis_artists: dict[str, list] = {k: [] for k in axes}
    for view_name, ax in axes.items():
        steering_axis_artists[view_name] = visualizer.draw_steering_response_axes(
            ax, upright_labels
        )

    # Add legend once on iso view.
    axes["iso"].legend(loc="upper left")

    # Use unified draw_wheel to create wheel artists.
    num_bands = 36
    wheel_artists: dict[str, list[dict]] = {k: [] for k in axes.keys()}
    for view_name, ax in axes.items():
        wheel_artists[view_name] = visualizer.draw_wheel(
            ax, initial_positions, num_bands=num_bands
        )

    # Layout.
    plt.subplots_adjust(
        left=0.0, right=1, bottom=0.025, top=0.95, wspace=0.01, hspace=0.01
    )

    # Persistent title updated each frame (cheaper than re-creating)
    title_artist = fig.suptitle("", fontsize=16)
    title_center_key = (
        visualizer.wheel_references[0].center if visualizer.wheel_references else None
    )

    # Update function that only updates artist data (no clears/plots)
    def update(frame_index: int):
        frame = frames[frame_index]
        positions = frame.positions

        # Update links.
        for view_name in axes.keys():
            visualizer.update_links(link_artists[view_name], positions)

        # Update wheel geometry.
        for view_name in axes.keys():
            visualizer.update_wheel(
                wheel_artists[view_name], positions, num_bands=num_bands
            )
            visualizer.update_steering_response_axes(
                steering_axis_artists[view_name],
                upright_labels,
                frame.steering_response_axes,
                clipping_bounds,
            )

        # Update global title.
        if title_center_key is None:
            title_artist.set_text(f"Frame {frame_index}")
        else:
            wheel_center_z = positions[title_center_key][2]
            initial_wheel_center_z = initial_positions[title_center_key][2]
            title_artist.set_text(
                f"Wheel Center Z: {wheel_center_z - initial_wheel_center_z:.1f} [mm]"
            )

        artists = []
        for view_name in axes.keys():
            artists.extend(link_artists[view_name])
            for wheel in wheel_artists[view_name]:
                artists.extend(wheel["rims"])
                artists.extend(wheel["bands"])
            artists.extend(steering_axis_artists[view_name])
        return artists

    # Play forward then reverse (ping-pong).
    pingpong_frames = pingpong_animation_frames(frames)
    frame_indices = range(0, len(pingpong_frames), 1)

    # Choose writer automatically if not provided.
    out_suffix = output_path.suffix.lower()
    chosen_writer: str
    if writer is not None:
        chosen_writer = writer
    elif out_suffix in {".mp4", ".m4v", ".mov"}:
        chosen_writer = "ffmpeg"
    else:
        chosen_writer = "pillow"

    try:
        if chosen_writer == "ffmpeg":
            Writer = animation.writers["ffmpeg"]
            writer_inst = Writer(fps=fps, codec=codec)
        else:
            Writer = animation.writers[chosen_writer]
            writer_inst = Writer(fps=fps)
    except Exception:
        # Fallback to pillow.
        Writer = animation.writers["pillow"]
        writer_inst = Writer(fps=fps)

    if show_live:
        plt.ion()
        plt.show(block=False)

    try:
        with writer_inst.saving(fig, str(output_path), dpi):
            for frame_index in frame_indices:
                frame = pingpong_frames[frame_index]
                positions = frame.positions
                # Update all artists for this frame.
                for view_name in axes.keys():
                    visualizer.update_links(link_artists[view_name], positions)
                    visualizer.update_wheel(
                        wheel_artists[view_name], positions, num_bands=num_bands
                    )
                    visualizer.update_steering_response_axes(
                        steering_axis_artists[view_name],
                        upright_labels,
                        frame.steering_response_axes,
                        clipping_bounds,
                    )
                # Update global title.
                if title_center_key is None:
                    title_artist.set_text(f"Frame {frame_index}")
                else:
                    wheel_center_z = positions[title_center_key][2]
                    initial_wheel_center_z = initial_positions[title_center_key][2]
                    title_artist.set_text(
                        "Wheel Center Z: "
                        f"{wheel_center_z - initial_wheel_center_z:.1f} [mm]"
                    )
                if show_live:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                writer_inst.grab_frame()
    finally:
        if show_live:
            plt.ioff()
        plt.close(fig)
