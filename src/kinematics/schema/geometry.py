"""Structured geometry specifications for explicit corner models."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from kinematics.core.enums import Units
from kinematics.schema.coercion import CIPointID, CISide, CIUnits, PydanticPoint3
from kinematics.schema.config import SuspensionConfig

HardpointMap = dict[CIPointID, PydanticPoint3]


class GeometrySpecBase(BaseModel):
    """Fields shared by every geometry specification."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str = "unnamed"
    version: str = "0.0.0"
    units: CIUnits = Units.MILLIMETERS
    side: CISide
    type: str
    config: SuspensionConfig


class DoubleWishboneGeometrySpec(GeometrySpecBase):
    """A basic double-wishbone corner."""

    type: Literal["double_wishbone"] = "double_wishbone"
    hardpoints: HardpointMap


class DoubleWishboneCoiloverGeometrySpec(GeometrySpecBase):
    """A double-wishbone corner with a lower-wishbone-mounted coilover."""

    type: Literal["double_wishbone_coilover"] = "double_wishbone_coilover"
    hardpoints: HardpointMap


GeometrySpec = DoubleWishboneGeometrySpec | DoubleWishboneCoiloverGeometrySpec


def parse_geometry_spec(data: Mapping[str, Any]) -> GeometrySpecBase:
    """Validate a raw mapping as the explicitly selected corner type."""
    if "type" not in data:
        raise ValueError("Geometry type not specified")

    normalized = dict(data)
    type_key = str(normalized["type"]).lower()
    normalized["type"] = type_key
    from kinematics.suspensions.registry import get_suspension_definition

    definition = get_suspension_definition(type_key)
    if definition is None:
        raise ValueError(f"Unsupported geometry type: '{type_key}'")
    normalized["type"] = definition.type_key
    try:
        return definition.spec_type.model_validate(normalized)
    except Exception as error:
        raise ValueError(f"Invalid geometry specification: {error}") from error


__all__ = [
    "DoubleWishboneCoiloverGeometrySpec",
    "DoubleWishboneGeometrySpec",
    "GeometrySpec",
    "GeometrySpecBase",
    "HardpointMap",
    "parse_geometry_spec",
]
