"""Focused tests for the axle-owned anti-roll-bar coupling."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.constraints import DistanceConstraint
from kinematics.core.elements import (
    ElementType,
    TorsionElement,
    VariableLengthLinkElement,
)
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import (
    ArbUBar,
    AxleSuspension,
    HeaveLinkRockerToRocker,
)
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep


@pytest.fixture
def rocker_axle(test_data_dir: Path) -> AxleSuspension:
    """Load the explicit rocker/ARB axle fixture."""
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(suspension, AxleSuspension)
    assert isinstance(suspension.anti_roll, ArbUBar)
    return suspension


def _load_heave_axle(
    tmp_path: Path,
    test_data_dir: Path,
) -> AxleSuspension:
    data = cast(
        "dict[str, object]",
        yaml.safe_load(
            (test_data_dir / "axle_geometry_rocker.yaml").read_text(encoding="utf-8")
        ),
    )
    data["heave_link"] = {"type": "rocker_to_rocker"}
    hardpoints = cast("dict[str, object]", data["hardpoints"])
    points = cast("dict[str, object]", hardpoints["points"])
    points["heave_link_rocker"] = {"x": 0, "y": 300, "z": 400}
    geometry_path = tmp_path / "axle_geometry_heave.yaml"
    geometry_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    suspension = load_geometry(geometry_path)
    assert isinstance(suspension, AxleSuspension)
    assert isinstance(suspension.anti_roll, ArbUBar)
    assert isinstance(suspension.heave_link, HeaveLinkRockerToRocker)
    return suspension


def test_arb_points_are_owned_by_axle(
    rocker_axle: AxleSuspension,
) -> None:
    """Shared-axis and arm points use their explicit axle namespaces."""
    state = rocker_axle.initial_state()

    assert PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A) in state.positions
    assert PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B) in state.positions
    assembly = rocker_axle.assembly()
    assert PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A) in assembly.points.fixed
    assert PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B) in assembly.points.fixed
    for side in (Side.LEFT, Side.RIGHT):
        arm_point = PointRef(side, PointID.DROPLINK_U_BAR)
        assert arm_point in state.positions
        assert arm_point in state.free_points
        assert arm_point in assembly.points.free


def test_roll_sweep_conserves_both_droplink_lengths(
    rocker_axle: AxleSuspension,
    test_data_dir: Path,
) -> None:
    """Opposed wheel travel moves the ARB without stretching its droplinks."""
    design = rocker_axle.initial_state()
    design_lengths = {
        side: (
            design.positions[PointRef(side, PointID.DROPLINK_ROCKER)]
            - design.positions[PointRef(side, PointID.DROPLINK_U_BAR)]
        ).norm()
        for side in (Side.LEFT, Side.RIGHT)
    }
    sweep = load_sweep(test_data_dir / "axle_rocker_sweep.yaml", rocker_axle)

    states, stats = solve_sweep(rocker_axle, sweep)

    assert all(info.converged for info in stats)
    for state in states:
        for side in (Side.LEFT, Side.RIGHT):
            length = (
                state.positions[PointRef(side, PointID.DROPLINK_ROCKER)]
                - state.positions[PointRef(side, PointID.DROPLINK_U_BAR)]
            ).norm()
            assert length == pytest.approx(design_lengths[side], abs=1e-5)


def test_arb_assembly_is_one_bar_and_two_droplinks(
    rocker_axle: AxleSuspension,
) -> None:
    """Render the shared ARB as one continuous series."""
    elements = rocker_axle.elements()
    arb_elements = [element for element in elements if element.label == "Anti-Roll Bar"]
    droplinks = [element for element in elements if element.label.endswith("Droplink")]

    assert len(arb_elements) == 1
    assert isinstance(arb_elements[0], TorsionElement)
    assert arb_elements[0].attachments == (
        PointRef(Side.LEFT, PointID.DROPLINK_U_BAR),
        PointRef(Side.RIGHT, PointID.DROPLINK_U_BAR),
    )
    assert not hasattr(arb_elements[0], "paths")
    assert {element.label for element in droplinks} == {
        "Left Droplink",
        "Right Droplink",
    }


def test_heave_link_composes_with_u_bar_without_rigid_length_constraint(
    tmp_path: Path,
    test_data_dir: Path,
) -> None:
    axle = _load_heave_axle(tmp_path, test_data_dir)
    endpoints = {
        PointRef(Side.LEFT, PointID.HEAVE_LINK_ROCKER),
        PointRef(Side.RIGHT, PointID.HEAVE_LINK_ROCKER),
    }

    assert endpoints <= axle.initial_state().positions.keys()
    assert not any(
        isinstance(constraint, DistanceConstraint)
        and constraint.involved_points == endpoints
        for constraint in axle.constraints()
    )
    heave_element = next(
        element
        for element in axle.elements()
        if isinstance(element, VariableLengthLinkElement)
        and element.type is ElementType.HEAVE_LINK
    )
    assert set(heave_element.point_keys) == endpoints
    derivative_names = {
        definition.column_name for definition in axle.derivative_metric_definitions()
    }
    assert {
        "deriv_heave_link_length_wrt_hub_z_left",
        "deriv_heave_link_length_wrt_hub_z_right",
    } <= derivative_names


def test_heave_link_length_changes_in_same_direction_wheel_travel(
    tmp_path: Path,
    test_data_dir: Path,
) -> None:
    axle = _load_heave_axle(tmp_path, test_data_dir)
    sweep_data = cast(
        "dict[str, object]",
        yaml.safe_load(
            (test_data_dir / "axle_rocker_sweep.yaml").read_text(encoding="utf-8")
        ),
    )
    targets = cast("list[dict[str, object]]", sweep_data["targets"])
    targets[1]["start"] = targets[0]["start"]
    targets[1]["stop"] = targets[0]["stop"]
    sweep_path = tmp_path / "axle_heave_sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep_data), encoding="utf-8")

    sweep = load_sweep(sweep_path, axle)
    states, stats = solve_sweep(axle, sweep)
    assert all(info.converged for info in stats)
    lengths = [
        (
            state.get(PointRef(Side.LEFT, PointID.HEAVE_LINK_ROCKER))
            - state.get(PointRef(Side.RIGHT, PointID.HEAVE_LINK_ROCKER))
        ).norm()
        for state in states
    ]
    assert max(lengths) - min(lengths) > 1.0

    result = compute_sweep_metrics(axle, sweep, states)
    assert result.derivative_error is None
    for row in result.rows:
        assert isinstance(row, AxleMetricRows)
        assert "heave_link_length" in row.axle
