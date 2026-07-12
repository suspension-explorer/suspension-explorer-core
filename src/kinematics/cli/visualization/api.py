"""
Public API for visualization features with lazy imports for optional dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kinematics.cli.visualization.plots import create_four_view_plot
from kinematics.core.types import Axis, PointID

if TYPE_CHECKING:
    from kinematics.core.types import Suspension, SuspensionState


@dataclass(frozen=True)
class GeometryVisualizationResult:
    """Ground-plane check returned after rendering a static geometry."""

    output_path: Path
    contact_patch_z: float
    contact_patch_on_ground: bool


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
    Install with: pip install "kinematics[cli,viz]"

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
        from kinematics.cli.visualization.animation import create_animation
        from kinematics.cli.visualization.main import (
            SuspensionVisualizer,
            WheelVisualization,
            renderer_links,
        )
    except ImportError as e:
        raise ImportError(
            "Visualization dependencies not found. "
            'Install with: pip install "kinematics[cli,viz]"\n'
            f"Original error: {e}"
        ) from e

    # Configure wheel visualization.
    wheel_config = WheelVisualization(
        diameter=wheel_diameter,
        width=wheel_width,
    )

    topology = suspension.topology()

    # Create visualizer.
    visualizer = SuspensionVisualizer(
        renderer_links(topology),
        wheel_config,
        topology.wheels,
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
) -> GeometryVisualizationResult:
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
            'Install with: pip install "kinematics[cli,viz]"\n'
            f"Original error: {e}"
        ) from e

    # Check for supported suspension types.
    is_double_wishbone = suspension.TYPE_KEY in (
        "double_wishbone",
        "double_wishbone_front",
        "double_wishbone_rear",
    )

    if not is_double_wishbone:
        raise NotImplementedError(
            "Geometry visualization only supported for double wishbone suspensions."
        )

    state = suspension.initial_state()
    contact_patch_z = float(state.get(PointID.CONTACT_PATCH_CENTER)[Axis.Z])

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

    return GeometryVisualizationResult(
        output_path=output_path,
        contact_patch_z=contact_patch_z,
        contact_patch_on_ground=bool(np.isclose(contact_patch_z, 0.0, atol=1e-2)),
    )
