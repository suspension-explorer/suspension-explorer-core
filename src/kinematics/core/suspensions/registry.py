"""Single catalogue of supported suspension geometry types."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from kinematics.core.schema.geometry import (
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
)
from kinematics.core.suspensions.axle import DoubleWishboneAxleSuspension
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.build import (
    build_double_wishbone,
    build_double_wishbone_axle,
)
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.enums import SuspensionType

SuspensionBuilder = Callable[[GeometrySpecBase], Suspension]
SuspensionClass = type[Suspension]


@dataclass(frozen=True)
class SuspensionDefinition:
    """Schema, builder, and runtime class belonging to one public type key."""

    type_key: SuspensionType
    spec_type: type[GeometrySpecBase]
    build: SuspensionBuilder
    suspension_type: SuspensionClass
    aliases: frozenset[str] = frozenset()


SUSPENSION_DEFINITIONS = (
    SuspensionDefinition(
        SuspensionType.DOUBLE_WISHBONE,
        DoubleWishboneGeometrySpec,
        build_double_wishbone,
        DoubleWishboneSuspension,
        DoubleWishboneSuspension.ALIASES,
    ),
    SuspensionDefinition(
        SuspensionType.DOUBLE_WISHBONE_AXLE,
        DoubleWishboneAxleGeometrySpec,
        build_double_wishbone_axle,
        DoubleWishboneAxleSuspension,
    ),
)

_definitions_by_key: dict[str, SuspensionDefinition] = {}
for definition in SUSPENSION_DEFINITIONS:
    _definitions_by_key[definition.type_key.value] = definition
    for alias in definition.aliases:
        _definitions_by_key[alias] = definition

SUSPENSION_REGISTRY = MappingProxyType(_definitions_by_key)


def get_suspension_definition(
    type_key: str | SuspensionType,
) -> SuspensionDefinition | None:
    """Return the complete definition for a type key."""
    return SUSPENSION_REGISTRY.get(type_key)


def get_suspension_class(type_key: str | SuspensionType) -> SuspensionClass | None:
    """Return the runtime class registered for a type key."""
    definition = get_suspension_definition(type_key)
    return None if definition is None else definition.suspension_type


def list_supported_types() -> list[str]:
    """Return every supported public type key in sorted order."""
    return sorted(SUSPENSION_REGISTRY)
