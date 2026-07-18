"""Tests for the transport-neutral decoded-input boundary."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from kinematics.core.enums import Axis, PointID
from kinematics.core.input import (
    build_suspension,
    build_sweep,
    parse_geometry_spec,
    parse_sweep_spec,
)
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.targeting import PointTargetAxis


def _read_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Fixture must contain a mapping: {path}")
    return data


def test_core_builds_corner_from_decoded_mapping(test_data_dir: Path) -> None:
    data = _read_mapping(test_data_dir / "geometry.yaml")

    suspension = build_suspension(data)

    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.initial_state().positions[PointID.AXLE_OUTBOARD].data.shape == (
        3,
    )


def test_core_builds_nested_axle_from_decoded_mapping(test_data_dir: Path) -> None:
    data = _read_mapping(test_data_dir / "axle_geometry_rocker.yaml")

    suspension = build_suspension(data)

    assert isinstance(suspension, AxleSuspension)
    assert set(suspension.corners) == {Side.LEFT, Side.RIGHT}
    suspension.initial_state()


def test_core_builds_sweep_from_decoded_mapping(test_data_dir: Path) -> None:
    suspension = build_suspension(_read_mapping(test_data_dir / "axle_geometry.yaml"))
    data = _read_mapping(test_data_dir / "axle_sweep.yaml")

    spec = parse_sweep_spec(data)
    sweep = build_sweep(data, suspension)

    assert spec.targets[0].point is PointID.WHEEL_CENTER
    assert spec.targets[0].side is Side.LEFT
    assert spec.targets[0].direction.axis is Axis.Z
    target = sweep.target_sweeps[0][0]
    assert target.point_id == PointRef(Side.LEFT, PointID.WHEEL_CENTER)
    assert isinstance(target.direction, PointTargetAxis)
    assert target.direction.axis is Axis.Z


def test_missing_coordinate_is_a_field_located_value_error(
    test_data_dir: Path,
) -> None:
    data = _read_mapping(test_data_dir / "geometry.yaml")
    del data["hardpoints"]["axle_outboard"]["z"]

    with pytest.raises(
        ValueError,
        match=r"(?s)axle_outboard.*missing coordinate\(s\): z",
    ):
        parse_geometry_spec(data)


def test_unknown_coordinate_is_rejected(test_data_dir: Path) -> None:
    data = _read_mapping(test_data_dir / "geometry.yaml")
    data["hardpoints"]["axle_outboard"]["w"] = 1.0

    with pytest.raises(ValueError, match=r"unknown coordinate\(s\): w"):
        parse_geometry_spec(data)
