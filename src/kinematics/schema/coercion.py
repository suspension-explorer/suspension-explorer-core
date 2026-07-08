"""
Coercion utilities for structured input.

These helpers turn loosely-typed structured data (parsed YAML, JSON request
bodies, plain Python dicts) into the package's core value types. They are the
single tolerant boundary of the package: everything inboard of the schema layer
works with strict core types (enums, Point3, Direction3) only.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypeVar

import numpy as np
from pydantic import BeforeValidator

from kinematics.core.enums import Axis, PointID, TargetPositionMode, Units
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import Side

E = TypeVar("E", bound=Enum)

# Input types that can be coerced to Point3.
Point3Like = Point3 | dict[str, float] | list[float] | tuple[float, float, float]

# Input types that can be coerced to Direction3.
Direction3Like = (
    Direction3 | Point3 | dict[str, float] | list[float] | tuple[float, float, float]
)


def coerce_enum(enum_cls: type[E], value: str | int | E) -> E:
    """Case-insensitive enum coercion from name, value, or instance."""
    if isinstance(value, enum_cls):
        return value

    # Try by name (case-insensitive).
    if isinstance(value, str):
        for name, member in enum_cls.__members__.items():
            if name.lower() == value.lower():
                return member
        # Try by value for string-valued enums.
        for member in enum_cls:
            if isinstance(member.value, str) and member.value.lower() == value.lower():
                return member

    # Try by value directly (for IntEnum).
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        pass

    valid = ", ".join(enum_cls.__members__.keys())
    raise ValueError(f"Invalid {enum_cls.__name__}: {value!r}. Valid: {valid}")


# Case-insensitive enum type aliases.
CIAxis = Annotated[Axis, BeforeValidator(lambda v: coerce_enum(Axis, v))]
CIPointID = Annotated[PointID, BeforeValidator(lambda v: coerce_enum(PointID, v))]
CISide = Annotated[Side, BeforeValidator(lambda v: coerce_enum(Side, v))]
CIUnits = Annotated[Units, BeforeValidator(lambda v: coerce_enum(Units, v))]
CITargetPositionMode = Annotated[
    TargetPositionMode, BeforeValidator(lambda v: coerce_enum(TargetPositionMode, v))
]


def coerce_point3(value: Any) -> Point3:
    """
    Coerce various input formats to a Point3.

    Accepts:
        - [x, y, z] list/tuple
        - {x: ..., y: ..., z: ...} dict
        - numpy array
        - Point3
    """
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


# Pydantic field type for Point3-valued fields.
# Static type is Point3 (so attribute access works without casts), but the
# BeforeValidator coerces wider inputs (lists, tuples, dicts, ndarrays) at
# runtime. Direct model construction with raw inputs requires explicit
# Point3(...) wrapping to satisfy the type checker; dict-driven
# `Model.model_validate(...)` is unaffected.
PydanticPoint3 = Annotated[Point3, BeforeValidator(coerce_point3)]


def coerce_direction3(value: Any) -> Direction3:
    """
    Coerce various input formats to a Direction3 (auto-normalised).

    Accepts:
        - [x, y, z] list/tuple
        - {x: ..., y: ..., z: ...} dict
        - numpy array
        - Point3 (treated as a position vector)
        - Direction3

    Raises:
        ValueError: If the input has zero length (cannot be normalized).
    """
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


# Pydantic field type for Direction3-valued fields.
# Static type is Direction3 (so attribute access works without casts). The
# BeforeValidator coerces wider inputs at runtime and normalizes to unit
# length; zero-length inputs raise ValueError.
PydanticDirection3 = Annotated[Direction3, BeforeValidator(coerce_direction3)]
