"""Tests for renderer-facing display topology normalization."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.core.enums import PointID
from kinematics.core.point_ref import PointRef, Side
from kinematics.io import load_geometry
from kinematics.visualization.display import (
    AXIS_FOOT_SUFFIX,
    display_links,
    display_point_keys,
    display_positions,
    rocker_display_groups,
)


def test_corner_rocker_fan_becomes_axis_and_perpendicular_arms(
    test_data_dir: Path,
) -> None:
    """Normalize the rocker fan without exposing renderer-side geometry math."""
    suspension = load_geometry(test_data_dir / "corner_strut_rocker_geometry.yaml")
    groups = rocker_display_groups(suspension)
    positions = display_positions(
        suspension.initial_state().positions,
        display_point_keys(suspension),
        groups,
    )
    links = display_links(suspension)

    assert len(groups) == 1
    assert not any(link.label == "Rocker" for link in links)
    assert {link.label for link in links if "Rocker" in link.label} == {
        "Rocker Axis",
        "Rocker Pushrod Arm",
    }

    group = groups[0]
    axis_start = np.asarray(positions[group.axis_front])
    axis_end = np.asarray(positions[group.axis_rear])
    pickup_name = group.pickups[0]
    pickup = np.asarray(positions[pickup_name])
    foot = np.asarray(positions[f"{pickup_name}{AXIS_FOOT_SUFFIX}"])

    assert np.dot(axis_end - axis_start, pickup - foot) == pytest.approx(0.0)


def test_axle_display_includes_shared_arb_points_and_side_keyed_rockers(
    test_data_dir: Path,
) -> None:
    """Include fixed shared-link vertices and normalize both axle rockers."""
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    point_keys = display_point_keys(suspension)
    groups = rocker_display_groups(suspension)
    positions = display_positions(
        suspension.initial_state().positions,
        point_keys,
        groups,
    )

    assert PointRef(Side.CENTER, PointID.ARB_AXIS_A) in point_keys
    assert PointRef(Side.CENTER, PointID.ARB_AXIS_B) in point_keys
    assert {group.label_prefix for group in groups} == {"Left ", "Right "}
    for group in groups:
        assert len(group.pickups) == 2
        for pickup in group.pickups:
            assert f"{pickup}{AXIS_FOOT_SUFFIX}" in positions
