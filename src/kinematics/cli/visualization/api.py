"""
Public API for optional visualization features.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kinematics.cli.visualization.animation import create_animation
from kinematics.cli.visualization.main import build_render_model
from kinematics.cli.visualization.plots import create_four_view_plot

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.base import Suspension


@dataclass(frozen=True)
class GeometryVisualizationResult:
    """
    Ground-plane check returned after rendering a static geometry.
    """

    output_path: Path
    contact_patch_z: tuple[float, ...]
    contact_patch_on_ground: bool


def visualize_suspension_sweep(
    suspension: "Suspension",
    solution_states: list["SuspensionState"],
    output_path: Path,
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
        fps: Frames per second for the animation.
        show_live: Whether to show the animation during creation.

    """
    render_model = build_render_model(suspension)

    # Get initial positions for animation baseline.
    initial_positions = render_model.positions(suspension.initial_state())

    # Extract position dictionaries from states.
    position_states = [render_model.positions(state) for state in solution_states]

    # Create the animation.
    create_animation(
        position_states,
        initial_positions,
        render_model.visualizer,
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
    state = suspension.initial_state()
    render_model = build_render_model(suspension)
    positions = render_model.positions(state)
    contact_patch_z = tuple(
        float(positions[references.contact_patch][2])
        for references in render_model.visualizer.wheel_references
    )
    if not contact_patch_z:
        raise ValueError("Suspension assembly has no wheel contact patches")

    # Create the four-view plot.
    create_four_view_plot(
        state=state,
        suspension=suspension,
        output_path=output_path,
        title="Suspension Geometry Visualization",
        dpi=150,
    )

    return GeometryVisualizationResult(
        output_path=output_path,
        contact_patch_z=contact_patch_z,
        contact_patch_on_ground=bool(
            np.all(np.isclose(contact_patch_z, 0.0, atol=1e-2))
        ),
    )
