"""Focused validation tests for axle geometry and side-qualified sweeps."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from kinematics.core.enums import (
    ActuationType,
    ArbType,
    Axis,
    AxlePosition,
    CornerSpringType,
    HeaveLinkType,
    PointID,
    Scope,
    SteeringType,
    SuspensionType,
    TargetPositionMode,
)
from kinematics.core.input import build_suspension, parse_geometry_spec
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.config import HeaveLinkConfig
from kinematics.core.schema.decoding import parse_enum
from kinematics.core.schema.geometry import (
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneGeometrySpec,
)
from kinematics.core.schema.sweep import SweepSpec, TargetSpec, build_sweep_config
from kinematics.core.targeting import PointTargetAxis


def _read_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    """Read one test fixture as an ordinary decoded mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{kind} fixture must contain a mapping")
    return data


def test_axle_geometry_uses_left_corner_as_mirror_source(
    test_data_dir: Path,
) -> None:
    spec = parse_geometry_spec(
        _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    )

    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)
    assert spec.hardpoints.right is None
    assert spec.hardpoints.left[PointID.AXLE_OUTBOARD][Axis.Y] > 0.0
    assert spec.vehicle_config.wheelbase == pytest.approx(2500.0)
    assert spec.axle_config.axle_position is AxlePosition.FRONT
    assert spec.axle_config.steering.type is SteeringType.RACK
    assert spec.axle_config.wheel.tire.section_width == pytest.approx(270.0)
    assert spec.axle_config.left_setup.camber_shim is None


def test_geometry_selectors_parse_as_enums_and_serialize_as_strings(
    test_data_dir: Path,
) -> None:
    spec = parse_geometry_spec(
        _read_yaml_mapping(test_data_dir / "axle_geometry_rocker.yaml", "Geometry")
    )
    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)

    assert spec.type is SuspensionType.DOUBLE_WISHBONE
    assert spec.scope is Scope.AXLE
    assert spec.axle_config.actuation.type is ActuationType.PUSHROD_ROCKER
    assert spec.axle_config.spring.type is CornerSpringType.TORSION_BAR
    assert spec.axle_config.anti_roll.type is ArbType.U_BAR
    assert spec.axle_config.heave_link.type is HeaveLinkType.NONE
    assert spec.axle_config.steering.type is SteeringType.RACK

    assert spec.model_dump(mode="json", include={"type", "scope"}) == {
        "type": "double_wishbone",
        "scope": "axle",
    }
    assert spec.axle_config.model_dump(mode="json")["steering"] == {"type": "rack"}
    assert spec.axle_config.actuation.model_dump(mode="json") == {
        "type": "pushrod_rocker",
        "mount": "upright",
    }
    assert spec.axle_config.spring.model_dump(mode="json") == {"type": "torsion_bar"}
    assert spec.axle_config.anti_roll.model_dump(mode="json") == {"type": "u_bar"}
    assert spec.axle_config.heave_link.model_dump(mode="json") == {"type": "none"}


def test_axle_rejects_boolean_steering_flag(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["axle_config"].pop("steering")
    data["axle_config"]["steered"] = True

    with pytest.raises(ValueError, match="axle_config.steering"):
        parse_geometry_spec(data)


def test_axle_rejects_unknown_steering_type(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["axle_config"]["steering"] = {"type": "rear_steer_magic"}

    with pytest.raises(ValueError, match="steering.type"):
        parse_geometry_spec(data)


def test_t_bar_selector_round_trips_without_heave_link(test_data_dir: Path) -> None:
    spec = parse_geometry_spec(
        _read_yaml_mapping(
            test_data_dir / "axle_geometry_t_bar.yaml",
            "Geometry",
        )
    )
    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)

    assert spec.axle_config.anti_roll.type is ArbType.T_BAR
    assert spec.axle_config.heave_link.type is HeaveLinkType.NONE
    assert spec.axle_config.anti_roll.model_dump(mode="json") == {"type": "t_bar"}


def test_t_bar_requires_pushrod_rocker_actuation(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(
        test_data_dir / "axle_geometry_t_bar.yaml",
        "Geometry",
    )
    data["axle_config"]["actuation"]["type"] = "direct"
    data["axle_config"]["spring"]["type"] = "none"

    with pytest.raises(ValueError, match="requires pushrod-rocker actuation"):
        parse_geometry_spec(data)


def test_rocker_to_rocker_heave_link_selector_round_trips() -> None:
    spec = HeaveLinkConfig.model_validate({"type": "rocker_to_rocker"})

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


def test_core_enum_parser_is_case_sensitive() -> None:
    assert parse_enum(PointID, "wheel_center") is PointID.WHEEL_CENTER

    with pytest.raises(ValueError, match="Invalid PointID"):
        parse_enum(PointID, "WHEEL_CENTER")


def test_core_schema_parses_serialized_enum_names() -> None:
    spec = TargetSpec.model_validate(
        {
            "point": "wheel_center",
            "side": "left",
            "mode": "absolute",
            "direction": {"axis": "z"},
            "values": [0.0],
        }
    )

    assert spec.point is PointID.WHEEL_CENTER
    assert spec.side is Side.LEFT
    assert spec.mode is TargetPositionMode.ABSOLUTE
    assert spec.direction.axis is Axis.Z


def test_axle_geometry_requires_left_corner(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["hardpoints"].pop("left")

    with pytest.raises(ValueError, match="left"):
        parse_geometry_spec(data)


def test_axle_geometry_requires_front_or_rear_position(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["axle_config"].pop("axle_position")

    with pytest.raises(ValueError, match="axle_position"):
        parse_geometry_spec(data)


@pytest.mark.parametrize("axle_position", ["front", "rear"])
def test_axle_position_accepts_front_or_rear(
    test_data_dir: Path,
    axle_position: str,
) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["axle_config"]["axle_position"] = axle_position

    spec = parse_geometry_spec(data)

    assert isinstance(spec, DoubleWishboneAxleGeometrySpec)
    assert spec.axle_config.axle_position is AxlePosition(axle_position)
    suspension = build_suspension(data)
    assert suspension.config is not None
    assert suspension.config.axle_position is AxlePosition(axle_position)


def test_right_setup_requires_explicit_right_hardpoints(
    test_data_dir: Path,
) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["axle_config"]["right_setup"] = {}

    with pytest.raises(ValueError, match="right_setup requires explicit"):
        parse_geometry_spec(data)


def test_axle_config_rejects_per_side_mechanisms(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    axle_config = data["axle_config"]
    axle_config["left"] = {
        "actuation": axle_config.pop("actuation"),
        "spring": axle_config.pop("spring"),
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        parse_geometry_spec(data)


def test_standalone_corner_defaults_to_left(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "geometry.yaml", "Geometry")
    data.pop("side")

    spec = parse_geometry_spec(data)

    assert isinstance(spec, DoubleWishboneGeometrySpec)
    assert spec.side is Side.LEFT


def test_axle_left_corner_rejects_right_handed_geometry(test_data_dir: Path) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["hardpoints"]["left"]["axle_outboard"]["y"] = -950.0
    with pytest.raises(ValueError, match="Side 'left' requires AXLE_OUTBOARD Y > 0"):
        build_suspension(data)


@pytest.mark.parametrize("field", ["steering", "wheel"])
def test_axle_scoped_configuration_is_not_vehicle_configuration(
    test_data_dir: Path,
    field: str,
) -> None:
    data = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml", "Geometry")
    data["vehicle_config"][field] = data["axle_config"].pop(field)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        parse_geometry_spec(data)


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
