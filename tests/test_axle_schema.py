"""Focused validation tests for axle geometry and side-qualified sweeps."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import _read_yaml_mapping
from kinematics.cli.io.schema_parser import parse_enum, parse_geometry_data
from kinematics.core.enums import (
    ActuationType,
    ArbType,
    Axis,
    CornerSpringType,
    HeaveLinkType,
    PointID,
    Scope,
    SuspensionType,
    TargetPositionMode,
)
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.geometry import (
    AxleHardpointsSpec,
    DoubleWishboneAxleGeometrySpec,
    HeaveLinkSpec,
    parse_geometry_spec,
)
from kinematics.core.schema.sweep import SweepSpec, TargetSpec, build_sweep_config
from kinematics.core.targeting import PointTargetAxis


def test_mirrored_axle_geometry_parses_without_top_level_side(
    test_data_dir: Path,
) -> None:
    spec = parse_geometry_spec(
        parse_geometry_data(
            _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
        )
    )

    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)
    assert spec.hardpoints.side is Side.LEFT
    assert not spec.hardpoints.is_explicit


def test_geometry_selectors_parse_as_enums_and_serialize_as_strings(
    test_data_dir: Path,
) -> None:
    spec = parse_geometry_spec(
        parse_geometry_data(
            _read_yaml_mapping(test_data_dir / "axle_geometry_rocker.yaml", "Geometry")
        )
    )
    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)

    assert spec.type is SuspensionType.DOUBLE_WISHBONE
    assert spec.scope is Scope.AXLE
    assert spec.actuation.type is ActuationType.PUSHROD_ROCKER
    assert spec.spring.type is CornerSpringType.TORSION_BAR
    assert spec.anti_roll.type is ArbType.U_BAR
    assert spec.heave_link.type is HeaveLinkType.NONE

    assert spec.model_dump(mode="json", include={"type", "scope"}) == {
        "type": "double_wishbone",
        "scope": "axle",
    }
    assert spec.actuation.model_dump(mode="json") == {
        "type": "pushrod_rocker",
        "mount": "upright",
    }
    assert spec.spring.model_dump(mode="json") == {"type": "torsion_bar"}
    assert spec.anti_roll.model_dump(mode="json") == {"type": "u_bar"}
    assert spec.heave_link.model_dump(mode="json") == {"type": "none"}


def test_t_bar_selector_round_trips_without_heave_link(test_data_dir: Path) -> None:
    spec = parse_geometry_spec(
        parse_geometry_data(
            _read_yaml_mapping(
                test_data_dir / "axle_geometry_t_bar.yaml",
                "Geometry",
            )
        )
    )
    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)

    assert spec.anti_roll.type is ArbType.T_BAR
    assert spec.heave_link.type is HeaveLinkType.NONE
    assert spec.anti_roll.model_dump(mode="json") == {"type": "t_bar"}


def test_t_bar_requires_pushrod_rocker_actuation(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(
        test_data_dir / "axle_geometry_t_bar.yaml",
        "Geometry",
    )
    data["actuation"]["type"] = "direct"
    data["spring"]["type"] = "none"

    with pytest.raises(ValueError, match="requires pushrod-rocker actuation"):
        parse_geometry_spec(parse_geometry_data(data))


def test_rocker_to_rocker_heave_link_selector_round_trips() -> None:
    spec = HeaveLinkSpec.model_validate({"type": "rocker_to_rocker"})

    assert spec.type is HeaveLinkType.ROCKER_TO_ROCKER
    assert spec.model_dump(mode="json") == {"type": "rocker_to_rocker"}


def test_core_schema_accepts_enum_objects() -> None:
    spec = TargetSpec.model_validate(
        {
            "point": PointID.WHEEL_CENTER,
            "direction": {"axis": Axis.Z},
            "side": Side.LEFT,
            "mode": TargetPositionMode.ABSOLUTE,
            "values": [0.0],
        }
    )

    assert spec.point is PointID.WHEEL_CENTER
    assert spec.direction.axis is Axis.Z
    assert spec.side is Side.LEFT
    assert spec.mode is TargetPositionMode.ABSOLUTE


def test_cli_enum_parser_is_case_sensitive() -> None:
    assert parse_enum(PointID, "wheel_center") is PointID.WHEEL_CENTER

    with pytest.raises(ValueError, match="Invalid PointID"):
        parse_enum(PointID, "WHEEL_CENTER")


def test_core_schema_does_not_parse_serialized_enum_names() -> None:
    with pytest.raises(ValueError):
        TargetSpec.model_validate(
            {
                "point": "wheel_center",
                "direction": {"axis": "z"},
                "values": [0.0],
            }
        )


def test_explicit_axle_geometry_requires_both_sides() -> None:
    with pytest.raises(ValueError, match="require both 'left' and 'right'"):
        AxleHardpointsSpec.model_validate({"left": {}})


def test_mirror_flag_is_not_part_of_axle_schema() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AxleHardpointsSpec.model_validate({"points": {}, "mirror": True})


def test_mirror_source_must_be_a_physical_side() -> None:
    with pytest.raises(ValueError, match="Mirror source side"):
        AxleHardpointsSpec.model_validate({"points": {}, "side": Side.CENTER})


def test_mirror_source_side_is_required() -> None:
    with pytest.raises(ValueError, match="require 'side'"):
        AxleHardpointsSpec.model_validate({"points": {}})


def test_side_target_requires_suspension_context() -> None:
    spec = SweepSpec.model_validate(
        {
            "version": 1,
            "targets": [
                {
                    "point": PointID.WHEEL_CENTER,
                    "side": Side.LEFT,
                    "direction": {"axis": Axis.Z},
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
                    "point": PointID.WHEEL_CENTER,
                    "direction": {"axis": Axis.X},
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
                    "point": PointID.WHEEL_CENTER,
                    "direction": {"axis": Axis.Z},
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
                    "point": PointID.WHEEL_CENTER,
                    "direction": {"axis": Axis.Z},
                    "values": [-10.0, 0.0, 10.0],
                },
                {
                    "point": PointID.TRACKROD_INBOARD,
                    "direction": {"axis": Axis.Y},
                    "values": [0.0, 1.0],
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="same length"):
        _ = spec.n_steps
