"""Transport-neutral decoding helpers for schema field values."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, TypeVar, cast

import numpy as np
from pydantic import BeforeValidator

from kinematics.core.enums import Axis, PointID, TargetPositionMode
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
    """Build a point from a three-component decoded sequence or mapping."""
    if isinstance(value, Point3):
        return value
    if isinstance(value, Mapping):
        coordinates = cast("Mapping[str, object]", value)
        required = {"x", "y", "z"}
        missing = required.difference(coordinates)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Point3 mapping is missing coordinate(s): {names}")
        extra = set(coordinates).difference(required)
        if extra:
            names = ", ".join(sorted(str(name) for name in extra))
            raise ValueError(f"Point3 mapping has unknown coordinate(s): {names}")
        value = [coordinates["x"], coordinates["y"], coordinates["z"]]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Point3 components must be numeric") from error
    if array.shape != (3,):
        raise ValueError(f"Point3 must have 3 components, got shape {array.shape}")
    return Point3(array)


def parse_direction3(value: object) -> Direction3:
    """Build a direction from a three-component decoded sequence or mapping."""
    if isinstance(value, Direction3):
        return value
    return Direction3(parse_point3(value).data)


Point3Value = Annotated[Point3, BeforeValidator(parse_point3)]
Direction3Value = Annotated[Direction3, BeforeValidator(parse_direction3)]
PointIDValue = Annotated[
    PointID, BeforeValidator(lambda value: parse_enum(PointID, value))
]
SideValue = Annotated[Side, BeforeValidator(lambda value: parse_enum(Side, value))]
AxisValue = Annotated[Axis, BeforeValidator(lambda value: parse_enum(Axis, value))]
TargetPositionModeValue = Annotated[
    TargetPositionMode,
    BeforeValidator(lambda value: parse_enum(TargetPositionMode, value)),
]
