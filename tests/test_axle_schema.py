"""Focused validation tests for axle geometry and side-qualified sweeps."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import _read_yaml_mapping
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.geometry import (
    AxleHardpointsSpec,
    DoubleWishboneAxleGeometrySpec,
    parse_geometry_spec,
)
from kinematics.core.schema.sweep import SweepSpec, build_sweep_config
from kinematics.core.targeting import PointTargetAxis


def test_mirrored_axle_geometry_parses_without_top_level_side(
    test_data_dir: Path,
) -> None:
    spec = parse_geometry_spec(
        _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    )

    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)
    assert spec.hardpoints.side is Side.LEFT
    assert not spec.hardpoints.is_explicit


def test_explicit_axle_geometry_requires_both_sides() -> None:
    with pytest.raises(ValueError, match="require both 'left' and 'right'"):
        AxleHardpointsSpec.model_validate({"left": {}})


def test_mirror_flag_is_not_part_of_axle_schema() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AxleHardpointsSpec.model_validate({"points": {}, "mirror": True})


def test_mirror_source_must_be_a_physical_side() -> None:
    with pytest.raises(ValueError, match="Mirror source side"):
        AxleHardpointsSpec.model_validate({"points": {}, "side": "center"})


def test_mirror_source_side_is_required() -> None:
    with pytest.raises(ValueError, match="require 'side'"):
        AxleHardpointsSpec.model_validate({"points": {}})


def test_side_target_requires_suspension_context() -> None:
    spec = SweepSpec.model_validate(
        {
            "version": 1,
            "targets": [
                {
                    "point": "wheel_center",
                    "side": "left",
                    "direction": {"axis": "z"},
                    "values": [0.0],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="requires a suspension context"):
        build_sweep_config(spec)


def test_x_axis_remains_an_axis_target() -> None:
    spec = SweepSpec.model_validate(
        {
            "targets": [
                {
                    "point": "wheel_center",
                    "direction": {"axis": "x"},
                    "values": [0.0],
                }
            ]
        }
    )

    config = build_sweep_config(spec)
    target = config.target_sweeps[0][0]
    assert target.point_id is PointID.WHEEL_CENTER
    assert isinstance(target.direction, PointTargetAxis)
    assert target.direction.axis is Axis.X


def test_sweep_spec_reports_expanded_step_count() -> None:
    spec = SweepSpec.model_validate(
        {
            "steps": 7,
            "targets": [
                {
                    "point": "wheel_center",
                    "direction": {"axis": "z"},
                    "start": -10.0,
                    "stop": 10.0,
                }
            ],
        }
    )

    assert spec.n_steps == 7


def test_sweep_spec_step_count_validates_target_lengths() -> None:
    spec = SweepSpec.model_validate(
        {
            "targets": [
                {
                    "point": "wheel_center",
                    "direction": {"axis": "z"},
                    "values": [-10.0, 0.0, 10.0],
                },
                {
                    "point": "trackrod_inboard",
                    "direction": {"axis": "y"},
                    "values": [0.0, 1.0],
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="same length"):
        _ = spec.n_steps
