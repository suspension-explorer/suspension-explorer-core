"""Tests for renderer-neutral suspension geometry resolution."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.visualization.main import (
    ELEMENT_STYLES,
    build_render_model,
    renderer_elements,
)
from kinematics.core.elements import (
    ElementType,
    RackElement,
    RockerElement,
    TorsionElement,
    WheelElement,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.presentation import (
    AxisProjection,
    PointMidpoint,
    axis_projection_name,
    element_paths,
    named_element_paths,
    named_point_keys,
    point_midpoint_name,
    resolve_positions,
    wheel_references,
)
from kinematics.core.primitives.point_ref import PointRef, Side, point_key_name


def test_corner_rocker_paths_include_perpendicular_arm_projection(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "corner_strut_rocker_geometry.yaml")
    assembly = suspension.assembly()
    positions = resolve_positions(suspension.initial_state().positions, assembly)
    paths = named_element_paths(assembly)

    rocker = next(
        element for element in assembly.elements if isinstance(element, RockerElement)
    )
    projection = next(
        point
        for path in element_paths(assembly)
        for point in path.points
        if isinstance(point, AxisProjection)
    )

    assert rocker.rotation_axis == (
        PointID.ROCKER_AXIS_A,
        PointID.ROCKER_AXIS_B,
    )
    assert all(isinstance(path.type, ElementType) for path in paths)
    assert all(not hasattr(path, "color") for path in paths)
    assert all(not hasattr(path, "style") for path in paths)
    assert {path.label for path in paths if "Rocker" in path.label} == {
        "Rocker Axis",
        "Rocker Pushrod Arm",
    }

    axis_start = np.asarray(positions[point_key_name(projection.rotation_axis[0])])
    axis_end = np.asarray(positions[point_key_name(projection.rotation_axis[1])])
    pickup = np.asarray(positions[point_key_name(projection.point)])
    projected = np.asarray(positions[axis_projection_name(projection)])

    assert set(named_point_keys(assembly)) == set(positions)
    assert np.dot(axis_end - axis_start, pickup - projected) == pytest.approx(0.0)


def test_axle_geometry_includes_shared_arb_and_side_keyed_rockers(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assembly = suspension.assembly()
    positions = resolve_positions(suspension.initial_state().positions, assembly)

    assert (
        PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)
        in assembly.referenced_point_keys
    )
    assert (
        PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)
        in assembly.referenced_point_keys
    )
    assert sum(isinstance(element, WheelElement) for element in assembly.elements) == 2
    rockers = [
        element for element in assembly.elements if isinstance(element, RockerElement)
    ]
    assert {rocker.label for rocker in rockers} == {"Left Rocker", "Right Rocker"}
    rack = next(
        element for element in assembly.elements if isinstance(element, RackElement)
    )
    assert rack.translation_axis is Axis.Y
    assert any(isinstance(element, TorsionElement) for element in assembly.elements)
    for rocker in rockers:
        assert len(rocker.pickups) == 2
        for pickup in rocker.pickups:
            projection = AxisProjection(pickup.point, rocker.rotation_axis)
            assert axis_projection_name(projection) in positions


def test_t_bar_crossbar_midpoint_is_resolved_as_presentation_geometry(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "axle_geometry_t_bar.yaml")
    assembly = suspension.assembly()
    positions = resolve_positions(suspension.initial_state().positions, assembly)
    midpoint = PointMidpoint(
        PointRef(Side.LEFT, PointID.DROPLINK_T_BAR),
        PointRef(Side.RIGHT, PointID.DROPLINK_T_BAR),
    )

    midpoint_name = point_midpoint_name(midpoint)
    assert midpoint_name in named_point_keys(assembly)
    assert midpoint_name in positions
    assert midpoint not in assembly.points.all
    assert np.asarray(positions[midpoint_name]) == pytest.approx(
        (
            np.asarray(positions[point_key_name(midpoint.point_a)])
            + np.asarray(positions[point_key_name(midpoint.point_b)])
        )
        / 2.0
    )


def test_cli_renderer_adds_styles_to_unstyled_element_paths(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assembly = suspension.assembly()
    positions = resolve_positions(suspension.initial_state().positions, assembly)
    paths = named_element_paths(assembly)

    rendered = renderer_elements(paths)

    assert [link.element for link in rendered] == list(paths)
    assert all(point in positions for link in rendered for point in link.points)
    assert all(link.style.color for link in rendered)
    assert ElementType.WHEEL_GROUND_TANGENT in ELEMENT_STYLES
    torsion_bar_paths = [
        path.points for path in paths if path.type is ElementType.TORSION_BAR
    ]
    assert torsion_bar_paths
    assert all(
        sum(path.points == torsion_path for path in paths) == 1
        for torsion_path in torsion_bar_paths
    )
    for label in {path.label for path in paths}:
        assert sum(link.label == label for link in rendered) == 1


def test_cli_renderer_has_distinct_heave_link_style() -> None:
    heave_style = ELEMENT_STYLES[ElementType.HEAVE_LINK]

    assert heave_style != ELEMENT_STYLES[ElementType.SPRING_DAMPER]
    assert heave_style.color
    assert heave_style.linestyle == "--"


def test_wheel_reference_uses_ground_tangent_name(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "geometry.yaml")
    reference = wheel_references(suspension.assembly())[0]

    assert reference.wheel_ground_tangent == "wheel_ground_tangent"


def test_wheel_element_and_path_use_canonical_ground_tangent_name(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "geometry.yaml")
    assembly = suspension.assembly()
    wheel = next(
        element for element in assembly.elements if isinstance(element, WheelElement)
    )
    path = next(
        path
        for path in named_element_paths(assembly)
        if path.type is ElementType.WHEEL_GROUND_TANGENT
    )

    assert wheel.wheel_ground_tangent is PointID.WHEEL_GROUND_TANGENT
    assert path.type.value == "wheel_ground_tangent"
    assert path.label == "Wheel Ground Tangent"


def test_cli_renderer_styles_wheel_ground_tangent_as_a_marker(
    test_data_dir: Path,
) -> None:
    suspension = load_geometry(test_data_dir / "geometry.yaml")
    render_model = build_render_model(suspension)
    visualizer = render_model.visualizer

    tangents = [
        link
        for link in visualizer.links
        if link.element.type is ElementType.WHEEL_GROUND_TANGENT
    ]

    assert len(tangents) == 1
    assert all(len(link.points) == 1 for link in tangents)
    assert all(link.style.linewidth == 0.0 for link in tangents)
    assert all(link.style.markersize > 0.0 for link in tangents)
    assert all(
        len(link.points) > 1
        for link in visualizer.links
        if link.element.type is not ElementType.WHEEL_GROUND_TANGENT
    )


def test_cli_renderer_static_and_animation_topology_includes_ground_tangent(
    test_data_dir: Path,
) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from kinematics.cli.visualization.plots import plot_suspension_on_axis

    suspension = load_geometry(test_data_dir / "geometry.yaml")
    state = suspension.initial_state()
    render_model = build_render_model(suspension)
    visualizer = render_model.visualizer
    positions = render_model.positions(state)

    fig = plt.figure()
    try:
        static_axis = fig.add_subplot(121, projection="3d")
        plot_suspension_on_axis(
            static_axis,
            visualizer,
            positions,
            view_name="iso",
        )
        # The single-point ground tangent is drawn as a scatter collection.
        assert len(static_axis.collections) == 1

        animation_axis = fig.add_subplot(122, projection="3d")
        artists = visualizer.draw_links(animation_axis, positions)
        visualizer.update_links(artists, positions)
        assert len(artists) == len(visualizer.links)
        assert len(animation_axis.lines) == len(visualizer.links)
    finally:
        plt.close(fig)
