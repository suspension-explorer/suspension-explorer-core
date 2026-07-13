"""
Visualization utilities for geometric computations.
"""

import importlib
from typing import Any, Optional, cast

import numpy as np

from kinematics.core.primitives.geometry import Direction3, Point3

PLOTTING_ENABLED = False


def should_plot() -> bool:
    """
    Check if plotting is enabled via environment variable (constant for now).
    """
    return PLOTTING_ENABLED


def plot_plane_from_points(
    a: Point3,
    b: Point3,
    c: Point3,
    normal: Optional[Direction3] = None,
    d: Optional[float] = None,
    title: str = "Plane from Three Points",
) -> None:
    """
    Plot three points and the plane they define.

    Args:
        a: First point defining the plane.
        b: Second point defining the plane.
        c: Third point defining the plane.
        normal: Optional unit normal direction of the plane.
        d: Optional distance parameter of the plane.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt

        # Import needed for 3D plotting.
        importlib.import_module("mpl_toolkits.mplot3d")
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax3d = cast(Any, ax)

    # Plot the three points.
    points = np.array([a.data, b.data, c.data])
    ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=["red", "green", "blue"],
        s=100,
        alpha=0.8,
    )

    # Label the points.
    ax3d.text(a[0], a[1], a[2], "A", fontsize=12)
    ax3d.text(b[0], b[1], b[2], "B", fontsize=12)
    ax3d.text(c[0], c[1], c[2], "C", fontsize=12)

    # Draw lines between points to show the triangle.
    triangle = np.array([a.data, b.data, c.data, a.data])  # Close the triangle.
    ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], "gray", alpha=0.5)

    if normal is not None and d is not None:
        # Plot normal vector from triangle centroid.
        # Use an affine combination since Point3 + Point3 is not allowed:
        # centroid = a + ((b - a) + (c - a)) / 3.
        centroid = a + ((b - a) + (c - a)) / 3
        span = max((b - a).norm(), (c - a).norm(), (c - b).norm())
        normal_scale = span * 0.5

        ax.quiver(
            centroid[0],
            centroid[1],
            centroid[2],
            normal[0] * normal_scale,
            normal[1] * normal_scale,
            normal[2] * normal_scale,
            color="purple",
            arrow_length_ratio=0.1,
            linewidth=3,
        )

        ax3d.text(
            centroid[0] + normal[0] * normal_scale * 1.1,
            centroid[1] + normal[1] * normal_scale * 1.1,
            centroid[2] + normal[2] * normal_scale * 1.1,
            "Normal",
            fontsize=10,
            color="purple",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_title(title)

    # Make axes equal.
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore

    plt.tight_layout()
    plt.show()


def plot_plane_intersection(
    n1: Direction3,
    d1: float,
    n2: Direction3,
    d2: float,
    line_point: Optional[Point3] = None,
    line_direction: Optional[Direction3] = None,
    title: str = "Plane Intersection",
) -> None:
    """
    Plot two planes and their line of intersection.

    Args:
        n1: Unit normal direction of first plane.
        d1: Distance parameter of first plane.
        n2: Unit normal direction of second plane.
        d2: Distance parameter of second plane.
        line_point: Optional point on intersection line.
        line_direction: Optional direction of intersection line.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt

        importlib.import_module("mpl_toolkits.mplot3d")
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax3d = cast(Any, ax)

    # Plot intersection line if provided.
    if line_point is not None and line_direction is not None:
        plot_range = 5.0
        t_vals = np.linspace(-plot_range, plot_range, 100)
        line_points = line_point.data + t_vals[:, np.newaxis] * line_direction.data
        ax3d.plot(
            line_points[:, 0],
            line_points[:, 1],
            line_points[:, 2],
            "green",
            linewidth=4,
            label="Intersection Line",
        )

        # Plot direction vector at line point.
        ax3d.quiver(
            line_point[0],
            line_point[1],
            line_point[2],
            line_direction[0],
            line_direction[1],
            line_direction[2],
            color="green",
            arrow_length_ratio=0.1,
            linewidth=2,
        )

    # Plot normal vectors from origin.
    ax3d.quiver(
        0.0,
        0.0,
        0.0,
        n1[0] * 2,
        n1[1] * 2,
        n1[2] * 2,
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
        label="Normal 1",
    )
    ax3d.quiver(
        0.0,
        0.0,
        0.0,
        n2[0] * 2,
        n2[1] * 2,
        n2[2] * 2,
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
        label="Normal 2",
    )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(title)

    plot_range = 5.0
    ax3d.set_xlim(-plot_range, plot_range)
    ax3d.set_ylim(-plot_range, plot_range)
    ax3d.set_zlim(-plot_range, plot_range)

    ax3d.legend()
    plt.tight_layout()
    plt.show()


def plot_line_plane_intersection(
    line_point: Point3,
    line_direction: Direction3,
    plane_y: float,
    intersection: Optional[Point3] = None,
    title: str = "Line-Plane Intersection",
) -> None:
    """
    Plot a line and vertical plane (Y = constant) with their intersection.

    Args:
        line_point: Point on the line.
        line_direction: Direction of the line (unit vector).
        plane_y: Y-coordinate defining the vertical plane.
        intersection: Optional intersection point.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt

        importlib.import_module("mpl_toolkits.mplot3d")
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax3d = cast(Any, ax)

    # Plot the line.
    plot_range = 10.0
    t_vals = np.linspace(-plot_range, plot_range, 100)
    line_points = line_point.data + t_vals[:, np.newaxis] * line_direction.data
    ax3d.plot(
        line_points[:, 0],
        line_points[:, 1],
        line_points[:, 2],
        "blue",
        linewidth=3,
        label="Line",
    )

    # Create vertical plane representation (just edges for clarity).
    x_edges = [-plot_range, plot_range, plot_range, -plot_range, -plot_range]
    z_edges = [-plot_range, -plot_range, plot_range, plot_range, -plot_range]
    y_edges = [plane_y] * len(x_edges)
    ax3d.plot(
        x_edges,
        y_edges,
        z_edges,
        "yellow",
        linewidth=2,
        alpha=0.7,
        label=f"Plane Y={plane_y}",
    )

    # Plot intersection point if provided.
    if intersection is not None:
        ax3d.scatter(
            intersection[0],
            intersection[1],
            intersection[2],
            c="red",
            s=100,
            label="Intersection",
        )
        ax3d.text(
            intersection[0],
            intersection[1],
            intersection[2],
            f"  ({intersection[0]:.1f}, {intersection[1]:.1f}, {intersection[2]:.1f})",
            fontsize=10,
        )

    # Plot line direction vector.
    ax3d.quiver(
        line_point[0],
        line_point[1],
        line_point[2],
        line_direction[0],
        line_direction[1],
        line_direction[2],
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
    )

    # Plot line starting point.
    ax3d.scatter(
        line_point[0],
        line_point[1],
        line_point[2],
        c="green",
        s=80,
        label="Line Point",
    )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(title)

    # Set reasonable axis limits.
    all_points = line_points
    if intersection is not None:
        all_points = np.vstack([all_points, intersection.data])

    x_range = (all_points[:, 0].min() - 2, all_points[:, 0].max() + 2)
    y_range = (
        min(all_points[:, 1].min() - 2, plane_y - 2),
        max(all_points[:, 1].max() + 2, plane_y + 2),
    )
    z_range = (all_points[:, 2].min() - 2, all_points[:, 2].max() + 2)

    ax3d.set_xlim(x_range)
    ax3d.set_ylim(y_range)
    ax3d.set_zlim(z_range)

    ax3d.legend()
    plt.tight_layout()
    plt.show()
