"""
Standard plotting functions for suspension visualization.

This module provides reusable plotting functionality for both single states and
animation sequences.
"""

from pathlib import Path
from typing import Mapping, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from kinematics.cli.visualization.main import SuspensionVisualizer, build_render_model
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension


def compute_bounds_from_positions(
    positions: Mapping[object, object],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Compute axis bounds and limits from position data.

    Args:
        positions: Dictionary mapping PointID to 3D positions.

    Returns:
        Tuple of (min_bounds, max_bounds, (x_mid, y_mid, z_mid, max_range))
    """
    all_points = np.asarray(
        [extract_array(position) for position in positions.values()]
    )
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
    y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
    z_mid = (max_bounds[2] + min_bounds[2]) * 0.5
    max_range = max(
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    )

    return min_bounds, max_bounds, (x_mid, y_mid, z_mid, max_range)


def compute_bounds_from_states(
    states: list[dict[str, tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Compute axis bounds and limits from multiple position states.

    Args:
        states: List of position dictionaries.

    Returns:
        Tuple of (min_bounds, max_bounds, (x_mid, y_mid, z_mid, max_range))
    """
    all_points = np.asarray(
        [extract_array(position) for state in states for position in state.values()]
    )
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
    y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
    z_mid = (max_bounds[2] + min_bounds[2]) * 0.5
    max_range = max(
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    )

    return min_bounds, max_bounds, (x_mid, y_mid, z_mid, max_range)


def configure_3d_axis(
    ax: Axes3D,
    view_name: str,
    x_mid: float,
    y_mid: float,
    z_mid: float,
    max_range: float,
) -> None:
    """
    Configure a 3D axis with standard view settings.

    Args:
        ax: The 3D axis to configure.
        view_name: View type ("front", "top", "side", "iso").
        x_mid: X center coordinate for the view.
        y_mid: Y center coordinate for the view.
        z_mid: Z center coordinate for the view.
        max_range: Range for consistent scaling.
    """
    # Set view-specific properties.
    if view_name == "top":
        ax.view_init(elev=90, azim=0)
        ax.set_title("Top View [X-Y]")
        ax.set_proj_type("ortho")
        ax.set_zticklabels([])  # type: ignore[attr-defined]
    elif view_name == "front":
        ax.view_init(elev=0, azim=0)
        ax.set_title("Front View [Y-Z]")
        ax.set_proj_type("ortho")
        ax.set_xticklabels([])
    elif view_name == "side":
        ax.view_init(elev=0, azim=90)
        ax.set_title("Side View [X-Z]")
        ax.set_proj_type("ortho")
        ax.set_yticklabels([])
    else:  # isometric
        ax.view_init(elev=20, azim=45)
        ax.set_title("Isometric View")
        ax.set_proj_type("ortho")

    # Set consistent axis limits and aspect ratio.
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
    ax.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")


def create_four_view_axes() -> tuple[Figure, dict[str, Axes3D]]:
    """
    Create a figure with four 3D subplots for standard views.

    Returns:
        Tuple of (figure, axes_dict) where axes_dict maps view names to 3D axes.
    """
    fig_scalar = 1.25
    fig = plt.figure(figsize=(16 * fig_scalar, 10 * fig_scalar))
    gs = fig.add_gridspec(2, 2)

    axes = {
        "front": cast(Axes3D, fig.add_subplot(gs[0, 0], projection="3d")),
        "top": cast(Axes3D, fig.add_subplot(gs[1, 0], projection="3d")),
        "side": cast(Axes3D, fig.add_subplot(gs[0, 1], projection="3d")),
        "iso": cast(Axes3D, fig.add_subplot(gs[1, 1], projection="3d")),
    }

    return fig, axes


def plot_suspension_on_axis(
    ax: Axes3D,
    visualizer: SuspensionVisualizer,
    positions: dict[str, tuple[float, float, float]],
    view_name: str,
    show_labels: bool = True,
) -> None:
    """
    Plot suspension links and wheel on a 3D axis.

    Args:
        ax: The 3D axis to plot on.
        visualizer: Suspension visualizer with links and wheel config.
        positions: Position data for all points.
        view_name: View name for label filtering.
        show_labels: Whether to show labels on this view.
    """
    # Plot suspension links.
    for link in visualizer.links:
        if len(link.points) > 1:
            pts = np.asarray([positions[name] for name in link.points])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=link.style.color,
                linewidth=link.style.linewidth,
                linestyle=link.style.linestyle,
                marker=link.style.marker,
                markersize=link.style.markersize,
                label=link.label if show_labels else None,
            )
        else:
            # Single point. matplotlib's mplot3d scatter stubs insist on int
            # for `zs`, even though the runtime accepts array-like; ignore
            # the pyright complaint on the third arg.
            pt = np.asarray(positions[link.points[0]])
            ax.scatter(
                pt[0:1],
                pt[1:2],
                pt[2:3],  # pyright: ignore[reportArgumentType]
                color=link.style.color,
                s=int(link.style.markersize**2),
                marker=link.style.marker,
                label=link.label if show_labels else None,
            )

    # Draw the wheel.
    visualizer.draw_wheel(ax, positions)


def create_four_view_plot(
    state: SuspensionState,
    suspension: Suspension,
    output_path: Path,
    title: str = "Suspension Geometry Visualization",
    dpi: int = 150,
) -> None:
    """
    Create a four-view plot (front, top, side, isometric) of a suspension state.

    Args:
        state: The suspension state to visualize.
        suspension: The Suspension instance for getting visualization links.
        output_path: Path where the plot image will be saved.
        title: Main title for the plot.
        dpi: DPI for the saved image.
    """
    render_model = build_render_model(suspension)
    positions = render_model.positions(state)

    # Create figure with four subplots.
    fig, axes = create_four_view_axes()

    # Compute global bounds for consistent scaling.
    _, _, (x_mid, y_mid, z_mid, max_range) = compute_bounds_from_positions(positions)

    # Configure each view and plot suspension.
    for view_name, ax in axes.items():
        configure_3d_axis(ax, view_name, x_mid, y_mid, z_mid, max_range)
        plot_suspension_on_axis(
            ax,
            render_model.visualizer,
            positions,
            view_name,
            view_name == "iso",
        )

    # Add legend only to isometric view.
    axes["iso"].legend(loc="upper left")

    # Set main title and layout.
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(
        left=0.0, right=1, bottom=0.025, top=0.95, wspace=0.01, hspace=0.01
    )

    # Save the plot.
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def create_single_view_plot(
    state: SuspensionState,
    suspension: Suspension,
    output_path: Path,
    view: str = "iso",
    title: str = "Suspension Geometry Visualization",
    dpi: int = 150,
) -> None:
    """
    Create a single view plot of a suspension state.

    Args:
        state: The suspension state to visualize.
        suspension: The Suspension instance for getting visualization links.
        output_path: Path where the plot image will be saved.
        view: View type ("front", "top", "side", "iso").
        title: Title for the plot.
        dpi: DPI for the saved image.
    """
    render_model = build_render_model(suspension)
    positions = render_model.positions(state)

    # Create single plot.
    fig = plt.figure(figsize=(12, 8))
    ax_raw = fig.add_subplot(111, projection="3d")
    ax = cast(Axes3D, ax_raw)

    # Compute bounds and configure axis.
    _, _, (x_mid, y_mid, z_mid, max_range) = compute_bounds_from_positions(positions)

    # Set custom title for isometric view.
    if view == "iso":
        configure_3d_axis(ax, view, x_mid, y_mid, z_mid, max_range)
        ax.set_title(title)  # Override default iso title.
    else:
        configure_3d_axis(ax, view, x_mid, y_mid, z_mid, max_range)

    # Plot suspension.
    plot_suspension_on_axis(
        ax,
        render_model.visualizer,
        positions,
        view,
        show_labels=True,
    )

    ax.legend()

    # Save the plot.
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
