"""
Visualization script for camber shim effects.

Generates side-by-side comparisons of stock vs. shimmed suspension geometry,
demonstrating how camber shims rotate the upright about the lower ball joint.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from kinematics import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.geometry import extract_array
from kinematics.schema import CamberShimConfig
from kinematics.suspensions.base import Suspension
from kinematics.visualization.api import visualize_geometry
from kinematics.visualization.main import SuspensionVisualizer, WheelVisualization
from kinematics.visualization.plots import (
    compute_bounds_from_positions,
    configure_3d_axis,
)

# Shim configuration constants.
DESIGN_SHIM_THICKNESS = 30.0  # mm - design/baseline shim stack thickness.
SETUP_SHIM_THICKNESS = 0.0  # mm - as-built/setup shim stack thickness.


def main():
    # Load the base geometry.
    geometry_path = Path("tests/data/geometry.yaml")
    suspension = load_geometry(geometry_path)

    # Visualize design configuration.
    print("\nGenerating design (baseline) visualization...")
    design_output = Path("camber_shim_design.png")
    visualize_geometry(suspension, design_output)
    print(f"Design visualization saved to: {design_output}")

    # Create setup configuration with shim change.
    # The shim sits between chassis and upper wishbone bracket.
    # The shim face center should be at/near the upper ball joint for maximum effect.
    # This represents the mounting face of the upright-side bracket.
    # Normal points outboard (positive Y).
    shim_config = CamberShimConfig(
        shim_face_point_a={
            "x": -25.0,  # Near upper ball joint X.
            "y": 750.0,  # Near upper ball joint Y.
            "z": 510.0,  # 10mm above mid-plane center.
        },
        shim_face_point_b={
            "x": -25.0,  # Near upper ball joint X.
            "y": 750.0,  # Near upper ball joint Y.
            "z": 490.0,  # 10mm below mid-plane center.
        },
        shim_face_normal={
            "x": 0.0,
            "y": 1.0,  # Unit vector pointing outboard.
            "z": 0.0,
        },
        design_thickness=DESIGN_SHIM_THICKNESS,
        setup_thickness=SETUP_SHIM_THICKNESS,
    )

    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    setup_suspension = create_setup_suspension(suspension, shim_config)

    # Visualize setup configuration.
    shim_delta = SETUP_SHIM_THICKNESS - DESIGN_SHIM_THICKNESS
    print(f"Generating setup ({shim_delta:+.1f}mm shim change) visualization...")
    setup_output = Path("camber_shim_setup.png")
    visualize_geometry(setup_suspension, setup_output)
    print(f"Setup visualization saved to: {setup_output}")

    # Overlay comparison plot: both suspensions on a single front view.
    comparison_output = Path("camber_shim_comparison.png")
    print("\nGenerating front-view comparison...")
    plot_front_view_comparison(
        suspension, setup_suspension, comparison_output, shim_delta
    )
    print(f"Comparison visualization saved to: {comparison_output}")


def create_setup_suspension(
    suspension: Suspension,
    shim_config: CamberShimConfig,
) -> Suspension:
    """Create a fresh suspension with the requested camber-shim setup."""
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    setup_config = suspension.config.model_copy(update={"camber_shim": shim_config})
    return replace(
        suspension,
        hardpoints=suspension.get_hardpoints_copy(),
        config=setup_config,
    )


def plot_front_view_comparison(
    design_suspension,
    setup_suspension,
    output_path: Path,
    shim_delta: float,
) -> None:
    """
    Plot both suspensions overlaid on a single front-view (Y-Z) axis.

    The design configuration is drawn in blue and the setup configuration
    in red so the geometric difference from the shim change is visible.
    """
    design_state = design_suspension.initial_state()
    setup_state = setup_suspension.initial_state()

    # Merge positions from both states to compute shared axis bounds.
    merged_positions = {**design_state.positions, **setup_state.positions}
    _, _, (x_mid, y_mid, z_mid, max_range) = compute_bounds_from_positions(
        merged_positions
    )

    fig = plt.figure(figsize=(12, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    configure_3d_axis(ax, "front", x_mid, y_mid, z_mid, max_range)

    # Helper: draw all elements for a suspension in a single color.
    def _draw_suspension(suspension, state, color: str, label: str) -> None:
        wheel_cfg = suspension.config.wheel
        wheel_config = WheelVisualization(
            diameter=wheel_cfg.tire.nominal_radius * 2,
            width=wheel_cfg.tire.section_width,
        )
        vis = SuspensionVisualizer(suspension.get_visualization_links(), wheel_config)

        # Draw links.
        first = True
        for link in vis.links:
            pts = np.stack([extract_array(state.positions[pid]) for pid in link.points])
            if len(link.points) > 1:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    linewidth=link.linewidth,
                    linestyle=link.linestyle,
                    marker=link.marker,
                    markersize=link.markersize,
                    label=label if first else None,
                )
            else:
                ax.scatter(
                    pts[0, 0],
                    pts[0, 1],
                    pts[0, 2],
                    color=color,
                    s=int(link.markersize**2),
                    marker=link.marker,
                    label=label if first else None,
                )
            first = False

        # Draw the wheel (rim circles and cross-tire bands) in the same color.
        positions = state.positions
        wheel_center = extract_array(positions[PointID.WHEEL_CENTER])
        wheel_inboard = extract_array(positions[PointID.WHEEL_INBOARD])
        wheel_outboard = extract_array(positions[PointID.WHEEL_OUTBOARD])
        axle_vec = extract_array(
            positions[PointID.AXLE_OUTBOARD] - positions[PointID.AXLE_INBOARD]
        )
        axle_vec = axle_vec / np.linalg.norm(axle_vec)

        # Build a local frame perpendicular to the axle.
        e2 = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(axle_vec, e2)) > 0.9:
            e2 = np.array([0.0, 1.0, 0.0])
        e2 = e2 - np.dot(e2, axle_vec) * axle_vec
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(axle_vec, e2)

        radius = wheel_config.diameter / 2
        theta = np.linspace(0, 2 * np.pi, wheel_config.num_points)

        # Rim circles for center, inboard, and outboard planes.
        for center, alpha in [
            (wheel_center, 0.25),
            (wheel_inboard, wheel_config.alpha),
            (wheel_outboard, wheel_config.alpha),
        ]:
            rim = np.array(
                [center + radius * (np.cos(t) * e2 + np.sin(t) * e3) for t in theta]
            )
            ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=color, alpha=alpha)

        # Cross-tire bands connecting inboard and outboard rims.
        num_bands = 48
        band_in, band_out = SuspensionVisualizer.get_band_endpoints(
            wheel_inboard, wheel_outboard, e2, e3, num_bands, radius
        )
        for i in range(num_bands):
            ax.plot(
                [band_in[i, 0], band_out[i, 0]],
                [band_in[i, 1], band_out[i, 1]],
                [band_in[i, 2], band_out[i, 2]],
                color=color,
                alpha=wheel_config.alpha,
            )

        # Contact patch marker.
        if PointID.CONTACT_PATCH_CENTER in positions:
            cp = extract_array(positions[PointID.CONTACT_PATCH_CENTER])
            ax.scatter(cp[0], cp[1], cp[2], color=color, s=100, marker="o")

    _draw_suspension(design_suspension, design_state, "#1f77b4", "Design [Shim = 30mm]")
    _draw_suspension(
        setup_suspension,
        setup_state,
        "#d62728",
        f"Setup [Shim = {SETUP_SHIM_THICKNESS:.1f} mm]",
    )

    ax.legend(loc="upper left")
    ax.set_title("Front View [Y-Z] - Design vs. Setup")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
