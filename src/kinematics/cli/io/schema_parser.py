"""Parse YAML values into objects accepted by the core schemas."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, TypeVar, cast

import numpy as np

from kinematics.core.enums import Axis, PointID, TargetPositionMode, Units
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import Side

E = TypeVar("E", bound=Enum)


def parse_enum(enum_type: type[E], value: object) -> E:
    """Parse one canonical, case-sensitive serialized enum value."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        for member in enum_type:
            serialized = (
                member.value if isinstance(member.value, str) else member.name.lower()
            )
            if value == serialized:
                return member

    valid = ", ".join(
        str(member.value) if isinstance(member.value, str) else member.name.lower()
        for member in enum_type
    )
    raise ValueError(
        f"Invalid {enum_type.__name__}: {value!r}. Expected one of: {valid}"
    )


def parse_point3(value: object) -> Point3:
    """Build a point from a three-component YAML sequence or mapping."""
    if isinstance(value, Point3):
        return value
    if isinstance(value, dict):
        coordinates = cast("dict[str, object]", value)
        array = np.array(
            [coordinates["x"], coordinates["y"], coordinates["z"]],
            dtype=np.float64,
        )
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"Point3 must have 3 components, got shape {array.shape}")
    return Point3(array)


def parse_direction3(value: object) -> Direction3:
    """Build a direction from a three-component YAML sequence or mapping."""
    if isinstance(value, Direction3):
        return value
    point = parse_point3(value)
    return Direction3(point.data)


def _parse_hardpoint_map(points: object) -> object:
    if not isinstance(points, dict):
        return points
    return {
        parse_enum(PointID, point): parse_point3(position)
        for point, position in points.items()
    }


def _parse_geometry_config(config: object) -> None:
    if not isinstance(config, dict):
        return
    config_data = cast("dict[str, object]", config)
    if "cg_position" in config_data:
        config_data["cg_position"] = parse_point3(config_data["cg_position"])
    mounted_points = config_data.get("upright_mounted_points")
    if isinstance(mounted_points, list):
        config_data["upright_mounted_points"] = [
            parse_enum(PointID, point) for point in mounted_points
        ]

    camber_shim = config_data.get("camber_shim")
    if not isinstance(camber_shim, dict):
        return
    camber_shim_data = cast("dict[str, object]", camber_shim)
    for field in ("shim_face_point_a", "shim_face_point_b"):
        if field in camber_shim_data:
            camber_shim_data[field] = parse_point3(camber_shim_data[field])
    if "shim_face_normal" in camber_shim_data:
        camber_shim_data["shim_face_normal"] = parse_direction3(
            camber_shim_data["shim_face_normal"]
        )


def parse_geometry_data(data: dict[str, Any]) -> dict[str, Any]:
    """Build domain objects from canonical values in geometry YAML data."""
    parsed = deepcopy(data)

    if "units" in parsed:
        parsed["units"] = parse_enum(Units, parsed["units"])
    if "side" in parsed:
        parsed["side"] = parse_enum(Side, parsed["side"])
    _parse_geometry_config(parsed.get("config"))

    hardpoints = parsed.get("hardpoints")
    if not isinstance(hardpoints, dict):
        return parsed

    is_axle_layout = any(
        field in hardpoints for field in ("points", "left", "right", "center")
    )
    if not is_axle_layout:
        parsed["hardpoints"] = _parse_hardpoint_map(hardpoints)
        return parsed

    if "side" in hardpoints:
        hardpoints["side"] = parse_enum(Side, hardpoints["side"])
    for field in ("points", "left", "right", "center"):
        if field in hardpoints:
            hardpoints[field] = _parse_hardpoint_map(hardpoints[field])
    return parsed


def parse_sweep_data(data: dict[str, Any]) -> dict[str, Any]:
    """Build domain objects from canonical values in sweep YAML data."""
    parsed = deepcopy(data)
    targets = parsed.get("targets")
    if not isinstance(targets, list):
        return parsed

    for target in targets:
        if not isinstance(target, dict):
            continue
        if "point" in target:
            target["point"] = parse_enum(PointID, target["point"])
        if "side" in target:
            target["side"] = parse_enum(Side, target["side"])
        if "mode" in target:
            target["mode"] = parse_enum(TargetPositionMode, target["mode"])

        direction = target.get("direction")
        if isinstance(direction, dict) and "axis" in direction:
            direction["axis"] = parse_enum(Axis, direction["axis"])
    return parsed
