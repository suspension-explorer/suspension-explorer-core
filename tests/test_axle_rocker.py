"""Focused tests for the axle-owned anti-roll-bar coupling."""

from pathlib import Path

import pytest

from kinematics.core.enums import PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.io import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep
from kinematics.suspensions.axle import (
    DoubleWishbonePushrodRockerAxleSuspension,
)


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
    for side in (Side.LEFT, Side.RIGHT):
        arm_point = PointRef(side, PointID.DROPLINK_ARB)
        assert arm_point in state.positions
        assert arm_point in state.free_points


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
    sweep = parse_sweep_file(test_data_dir / "axle_rocker_sweep.yaml", rocker_axle)

    states, stats = solve_sweep(rocker_axle, sweep)

    assert all(info.converged for info in stats)
    for state in states:
        for side in (Side.LEFT, Side.RIGHT):
            length = (
                state.positions[PointRef(side, PointID.DROPLINK_ROCKER)]
                - state.positions[PointRef(side, PointID.DROPLINK_ARB)]
            ).norm()
            assert length == pytest.approx(design_lengths[side], abs=1e-5)


def test_arb_visualization_is_one_bar_and_two_droplinks(
    rocker_axle: DoubleWishbonePushrodRockerAxleSuspension,
) -> None:
    """Render the shared ARB as one continuous series."""
    links = rocker_axle.get_visualization_links()
    arb_links = [link for link in links if link.label == "Anti-Roll Bar"]
    droplinks = [link for link in links if link.label.endswith("Droplink")]

    assert len(arb_links) == 1
    assert arb_links[0].points[0] == PointRef(Side.LEFT, PointID.DROPLINK_ARB)
    assert arb_links[0].points[-1] == PointRef(Side.RIGHT, PointID.DROPLINK_ARB)
    assert {link.label for link in droplinks} == {
        "Left Droplink",
        "Right Droplink",
    }
