"""Focused tests for the axle-owned anti-roll-bar coupling."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.elements import TorsionElement
from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import (
    DoubleWishbonePushrodRockerAxleSuspension,
)
from kinematics.core.sweep import solve_sweep


@pytest.fixture
def rocker_axle(test_data_dir: Path) -> DoubleWishbonePushrodRockerAxleSuspension:
    """Load the explicit rocker/ARB axle fixture."""
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(suspension, DoubleWishbonePushrodRockerAxleSuspension)
    return suspension


def test_arb_points_are_owned_by_axle(
    rocker_axle: DoubleWishbonePushrodRockerAxleSuspension,
) -> None:
    """Shared-axis and arm points use their explicit axle namespaces."""
    state = rocker_axle.initial_state()

    assert PointRef(Side.CENTER, PointID.ARB_AXIS_A) in state.positions
    assert PointRef(Side.CENTER, PointID.ARB_AXIS_B) in state.positions
    assembly = rocker_axle.assembly()
    assert PointRef(Side.CENTER, PointID.ARB_AXIS_A) in assembly.points.fixed
    assert PointRef(Side.CENTER, PointID.ARB_AXIS_B) in assembly.points.fixed
    for side in (Side.LEFT, Side.RIGHT):
        arm_point = PointRef(side, PointID.DROPLINK_ARB)
        assert arm_point in state.positions
        assert arm_point in state.free_points
        assert arm_point in assembly.points.free


def test_roll_sweep_conserves_both_droplink_lengths(
    rocker_axle: DoubleWishbonePushrodRockerAxleSuspension,
    test_data_dir: Path,
) -> None:
    """Opposed wheel travel moves the ARB without stretching its droplinks."""
    design = rocker_axle.initial_state()
    design_lengths = {
        side: (
            design.positions[PointRef(side, PointID.DROPLINK_ROCKER)]
            - design.positions[PointRef(side, PointID.DROPLINK_ARB)]
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
                - state.positions[PointRef(side, PointID.DROPLINK_ARB)]
            ).norm()
            assert length == pytest.approx(design_lengths[side], abs=1e-5)


def test_arb_assembly_is_one_bar_and_two_droplinks(
    rocker_axle: DoubleWishbonePushrodRockerAxleSuspension,
) -> None:
    """Render the shared ARB as one continuous series."""
    elements = rocker_axle.elements()
    arb_elements = [element for element in elements if element.label == "Anti-Roll Bar"]
    droplinks = [element for element in elements if element.label.endswith("Droplink")]

    assert len(arb_elements) == 1
    assert isinstance(arb_elements[0], TorsionElement)
    assert arb_elements[0].path[0] == PointRef(Side.LEFT, PointID.DROPLINK_ARB)
    assert arb_elements[0].path[-1] == PointRef(Side.RIGHT, PointID.DROPLINK_ARB)
    assert {element.label for element in droplinks} == {
        "Left Droplink",
        "Right Droplink",
    }
