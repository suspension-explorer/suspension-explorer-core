"""
Public API for visualization features with lazy imports for optional dependencies.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import typer

from kinematics.core.enums import Axis, PointID
from kinematics.visualization.plots import create_four_view_plot

if TYPE_CHECKING:
    from kinematics.state import SuspensionState
    from kinematics.suspensions.base import Suspension


def visualize_suspension_sweep(
    suspension: "Suspension",
    solution_states: list["SuspensionState"],
    output_path: Path,
    wheel_diameter: float,
    wheel_width: float,
    fps: int = 20,
    show_live: bool = False,
) -> None:
    """
    Create an animation of a suspension sweep.

    This function requires matplotlib and related visualization dependencies.
    Install with: pip install "kinematics[viz]"

    Args:
        suspension: The Suspension instance used to generate the solutions.
        solution_states: List of solved suspension states to animate.
        output_path: Path where the animation file will be saved.
        wheel_diameter: Wheel diameter in millimeters.
        wheel_width: Wheel width in millimeters.
        fps: Frames per second for the animation.
        show_live: Whether to show the animation during creation.

    Raises:
        ImportError: If visualization dependencies are not installed.
    """
    try:
        from kinematics.visualization.animation import create_animation
        from kinematics.visualization.main import (
            SuspensionVisualizer,
            WheelVisualization,
        )
    except ImportError as e:
        raise ImportError(
            "Visualization dependencies not found. "
            'Install with: pip install "kinematics[viz]"\n'
            f"Original error: {e}"
        ) from e

    # Configure wheel visualisation.
    wheel_config = WheelVisualization(
        diameter=wheel_diameter,
        width=wheel_width,
    )

    # Get visualisation links from suspension.
    visualization_links = suspension.get_visualization_links()

    # Create visualizer.
    visualizer = SuspensionVisualizer(
        visualization_links,
        wheel_config,
        wheel_anchors=suspension.wheel_visualization_anchors(),
    )

    # Get initial positions for animation baseline.
    initial_state = suspension.initial_state()
    initial_positions = initial_state.positions.copy()

    # Extract position dictionaries from states.
    position_states = [state.positions for state in solution_states]

    # Create the animation.
    create_animation(
        position_states,
        initial_positions,
        visualizer,
        output_path,
        fps=fps,
        show_live=show_live,
    )


def visualize_geometry(
    suspension: "Suspension",
    output_path: Path,
) -> None:
    """
    Creates a debug plot for a single suspension state and checks ground tangency.

    Args:
        suspension: The Suspension instance for the geometry.
        output_path: Path where the plot image will be saved.
    """
    try:
        # Test for matplotlib availability; plotting is handled by plots module.
        import matplotlib.pyplot as plt

        plt.figure()  # Test that matplotlib works.
        plt.close()
    except ImportError as e:
        raise ImportError(
            "Visualization dependencies not found. "
            'Install with: pip install "kinematics[viz]"\n'
            f"Original error: {e}"
        ) from e

    # Check for supported suspension types.
    supported_types = (
        "double_wishbone",
        "double_wishbone_front",
        "double_wishbone_rear",
        "double_wishbone_axle",
    )

    if suspension.TYPE_KEY not in supported_types:
        raise NotImplementedError(
            "Geometry visualization only supported for double wishbone suspensions."
        )

    typer.secho(
        "Checking and visualizing suspension geometry...",
    )

    state = suspension.initial_state()

    # Ground tangency: single-corner reads the corner's contact patch directly;
    # the axle checks each side via its stripped corner state (PointRef keys).
    if suspension.TYPE_KEY == "double_wishbone_axle":
        from kinematics.core.point_ref import Side
        from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

        assert isinstance(suspension, DoubleWishboneAxleSuspension)
        ground_checks = [
            (
                side.name.title(),
                float(
                    suspension.corner_state(state, side).get(
                        PointID.CONTACT_PATCH_CENTER
                    )[Axis.Z]
                ),
            )
            for side in (Side.LEFT, Side.RIGHT)
        ]
    else:
        ground_checks = [("", float(state.get(PointID.CONTACT_PATCH_CENTER)[Axis.Z]))]

    # Report the final status per checked corner.
    for label, z_offset in ground_checks:
        prefix = f"{label} " if label else ""
        if np.isclose(z_offset, 0.0, atol=1e-2):
            typer.secho(
                f"Geometry Check: OK. {prefix}Contact patch at ground "
                f"(Z = {z_offset:.3f} mm).",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"Geometry Check: WARNING. {prefix}Contact patch center is not "
                "on the ground.",
                fg=typer.colors.RED,
            )
            typer.echo("-" * 60)
            typer.secho(
                f"The {prefix.lower()}contact patch center currently located at "
                f"Z = {z_offset:.3f}mm."
            )
            typer.echo("-" * 60)

    # Get wheel configuration from the suspension.
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    wheel_cfg = suspension.config.wheel

    # Create the four-view plot.
    create_four_view_plot(
        state=state,
        suspension=suspension,
        output_path=output_path,
        wheel_diameter=wheel_cfg.tire.nominal_radius * 2,
        wheel_width=wheel_cfg.tire.section_width,
        title="Suspension Geometry Visualization",
        dpi=150,
    )

    typer.secho(f"Visualization saved to: {output_path}", fg=typer.colors.GREEN)
