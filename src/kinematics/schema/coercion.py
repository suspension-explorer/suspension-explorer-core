"""Shared Pydantic coercion utilities for external schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypeVar

import numpy as np
from pydantic import BeforeValidator

from kinematics.core.enums import Axis, PointID, TargetPositionMode, Units
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import Side

E = TypeVar("E", bound=Enum)

Point3Like = Point3 | dict[str, float] | list[float] | tuple[float, float, float]
Direction3Like = (
    Direction3 | Point3 | dict[str, float] | list[float] | tuple[float, float, float]
)


def coerce_enum(enum_cls: type[E], value: str | int | E) -> E:
    """Coerce an enum case-insensitively from its name, value, or instance."""
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        for name, member in enum_cls.__members__.items():
            if name.lower() == value.lower():
                return member
        for member in enum_cls:
            if isinstance(member.value, str) and member.value.lower() == value.lower():
                return member

    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        pass

    valid = ", ".join(enum_cls.__members__.keys())
    raise ValueError(f"Invalid {enum_cls.__name__}: {value!r}. Valid: {valid}")


CIAxis = Annotated[Axis, BeforeValidator(lambda v: coerce_enum(Axis, v))]
CIPointID = Annotated[PointID, BeforeValidator(lambda v: coerce_enum(PointID, v))]
CIUnits = Annotated[Units, BeforeValidator(lambda v: coerce_enum(Units, v))]
CISide = Annotated[Side, BeforeValidator(lambda v: coerce_enum(Side, v))]
CITargetPositionMode = Annotated[
    TargetPositionMode, BeforeValidator(lambda v: coerce_enum(TargetPositionMode, v))
]


def coerce_point3(value: Any) -> Point3:
    """Coerce a three-component sequence, mapping, or array to ``Point3``."""
    if isinstance(value, Point3):
        return value

    if isinstance(value, np.ndarray):
        arr = value.astype(np.float64)
    elif isinstance(value, dict):
        arr = np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    else:
        arr = np.array(value, dtype=np.float64)

    if arr.shape != (3,):
        raise ValueError(f"Point3 must have 3 components, got shape {arr.shape}")
    return Point3(arr)


PydanticPoint3 = Annotated[Point3, BeforeValidator(coerce_point3)]


def coerce_direction3(value: Any) -> Direction3:
    """Coerce a three-component value to a normalized ``Direction3``."""
    if isinstance(value, Direction3):
        return value
    if isinstance(value, Point3):
        return Direction3(value.data)

    if isinstance(value, np.ndarray):
        arr = value.astype(np.float64)
    elif isinstance(value, dict):
        arr = np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    else:
        arr = np.array(value, dtype=np.float64)

    if arr.shape != (3,):
        raise ValueError(f"Direction3 must have 3 components, got shape {arr.shape}")
    return Direction3(arr)


PydanticDirection3 = Annotated[Direction3, BeforeValidator(coerce_direction3)]
