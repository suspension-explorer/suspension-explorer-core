"""Single catalogue of supported suspension geometry types."""

from dataclasses import dataclass
from typing import Callable

from kinematics.schema.geometry import (
    DoubleWishboneCoiloverGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
)
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.build import (
    build_double_wishbone,
    build_double_wishbone_coilover,
)
from kinematics.suspensions.corner import (
    DoubleWishboneCoiloverSuspension,
    DoubleWishboneSuspension,
)

SuspensionBuilder = Callable[[GeometrySpecBase], Suspension]
SuspensionClass = type[Suspension]


@dataclass(frozen=True)
class SuspensionDefinition:
    """Schema, builder, and runtime class belonging to one public type key."""

    type_key: str
    spec_type: type[GeometrySpecBase]
    build: SuspensionBuilder
    suspension_type: SuspensionClass
    aliases: frozenset[str] = frozenset()


SUSPENSION_DEFINITIONS = (
    SuspensionDefinition(
        "double_wishbone",
        DoubleWishboneGeometrySpec,
        build_double_wishbone,
        DoubleWishboneSuspension,
        DoubleWishboneSuspension.ALIASES,
    ),
    SuspensionDefinition(
        "double_wishbone_coilover",
        DoubleWishboneCoiloverGeometrySpec,
        build_double_wishbone_coilover,
        DoubleWishboneCoiloverSuspension,
    ),
)

SUSPENSION_REGISTRY: dict[str, SuspensionDefinition] = {}
for definition in SUSPENSION_DEFINITIONS:
    SUSPENSION_REGISTRY[definition.type_key] = definition
    for alias in definition.aliases:
        SUSPENSION_REGISTRY[alias] = definition


def get_suspension_definition(type_key: str) -> SuspensionDefinition | None:
    """Return the complete definition for a type key."""
    return SUSPENSION_REGISTRY.get(type_key.lower())


def get_suspension_class(type_key: str) -> SuspensionClass | None:
    """Return the runtime class registered for a type key."""
    definition = get_suspension_definition(type_key)
    return None if definition is None else definition.suspension_type


def list_supported_types() -> list[str]:
    """Return every supported public type key in sorted order."""
    return sorted(SUSPENSION_REGISTRY)


__all__ = [
    "SUSPENSION_DEFINITIONS",
    "SUSPENSION_REGISTRY",
    "SuspensionBuilder",
    "SuspensionClass",
    "SuspensionDefinition",
    "get_suspension_class",
    "get_suspension_definition",
    "list_supported_types",
]
