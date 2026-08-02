"""
Public API for optional visualization features.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from kinematics.cli.visualization.main import SuspensionVisualizer, build_render_model
from kinematics.core.primitives.geometry import Point3
from kinematics.core.road import RoadPlane

if TYPE_CHECKING:
    from kinematics.core.rigid_motion import UprightScrewAxisResult
    from kinematics.core.state import SuspensionState
    from kinematics.core.suspensions.base import Suspension


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
    steering_response_axes: Sequence[Sequence["UprightScrewAxisResult"]] | None = None,
) -> None:
    """Load the optional animation renderer only when animation is requested."""
    from kinematics.cli.visualization.animation import create_animation as render

    render(
        position_states,
        initial_positions,
        visualizer,
        output_path,
        fps=fps,
        writer=writer,
        codec=codec,
        dpi=dpi,
        show_live=show_live,
        steering_response_axes=steering_response_axes,
    )


def create_four_view_plot(
    state: "SuspensionState",
    suspension: "Suspension",
    output_path: Path,
    title: str = "Suspension Geometry Visualization",
    dpi: int = 150,
) -> None:
    """Load the optional static renderer only when a plot is requested."""
    from kinematics.cli.visualization.plots import create_four_view_plot as render

    render(
        state=state,
        suspension=suspension,
        output_path=output_path,
        title=title,
        dpi=dpi,
    )


@dataclass(frozen=True)
class GeometryVisualizationResult:
    """
    Contact-plane check returned after rendering a static geometry.

    ``wheel_contact_centre_z`` remains available as the raw chassis-coordinate
    diagnostic.  It is not used to decide whether a point is on the road: the
    chassis origin may be vertically translated or rolled relative to the
    reconstructed road plane.
    """

    output_path: Path
    wheel_contact_centre_z: tuple[float, ...]
    wheel_contact_centre_road_distance_mm: tuple[float, ...]
    wheel_contact_centres_on_road: bool


def _road_plane_for_wheel_contact_centres(
    contact_centres: tuple[Point3, ...],
) -> RoadPlane:
    """Reconstruct the supported road datum from rendered contact points.

    A standalone corner has one contact point, so its equivalent road plane is
    horizontal through that point.  A two-wheel axle uses the same
    longitudinally-extruded plane as the axle contact closure.  The renderer
    has no supported topology with any other wheel count.
    """
    if len(contact_centres) == 1:
        return RoadPlane.horizontal_at(contact_centres[0])
    if len(contact_centres) == 2:
        return RoadPlane.from_axle_contact_centres(*contact_centres)
    raise ValueError(
        "Suspension assembly must expose one corner or two axle wheel contact centres"
    )


def visualize_suspension_sweep(
    suspension: "Suspension",
    solution_states: list["SuspensionState"],
    output_path: Path,
    fps: int = 20,
    show_live: bool = False,
    steering_response_axes: Sequence[Sequence["UprightScrewAxisResult"]] | None = None,
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
        steering_response_axes: Per-frame isolated steering-response axes.

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
        steering_response_axes=steering_response_axes,
    )


def visualize_geometry(
    suspension: "Suspension",
    output_path: Path,
) -> GeometryVisualizationResult:
    """
    Creates a debug plot for a single suspension state and checks contact support.

    Args:
        suspension: The Suspension instance for the geometry.
        output_path: Path where the plot image will be saved.
    """
    state = suspension.initial_state()
    render_model = build_render_model(suspension)
    positions = render_model.positions(state)
    contact_centres = tuple(
        Point3(positions[references.wheel_contact_centre])
        for references in render_model.visualizer.wheel_references
    )
    if not contact_centres:
        raise ValueError("Suspension assembly has no wheel contact centres")
    road = _road_plane_for_wheel_contact_centres(contact_centres)
    wheel_contact_centre_z = tuple(
        float(contact_centre[2]) for contact_centre in contact_centres
    )
    wheel_contact_centre_road_distance_mm = tuple(
        road.signed_distance(contact_centre) for contact_centre in contact_centres
    )

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
        wheel_contact_centre_z=wheel_contact_centre_z,
        wheel_contact_centre_road_distance_mm=wheel_contact_centre_road_distance_mm,
        wheel_contact_centres_on_road=bool(
            np.all(np.isclose(wheel_contact_centre_road_distance_mm, 0.0, atol=1e-2))
        ),
    )
