"""Single catalogue of supported suspension geometry types."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from kinematics.core.enums import Scope, SuspensionType
from kinematics.core.schema.geometry import (
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
    MacPhersonAxleGeometrySpec,
    MacPhersonGeometrySpec,
)
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.build import (
    build_double_wishbone,
    build_double_wishbone_axle,
    build_macpherson,
    build_macpherson_axle,
)
from kinematics.core.suspensions.corner import (
    DoubleWishboneSuspension,
    MacPhersonSuspension,
)

SuspensionBuilder = Callable[[GeometrySpecBase], Suspension]
SuspensionClass = type[Suspension]


@dataclass(frozen=True)
class SuspensionDefinition:
    """Schema, builder, and runtime class for one architecture and scope."""

    type_key: SuspensionType
    scope: Scope
    spec_type: type[GeometrySpecBase]
    build: SuspensionBuilder
    suspension_type: SuspensionClass


SUSPENSION_DEFINITIONS = (
    SuspensionDefinition(
        SuspensionType.DOUBLE_WISHBONE,
        Scope.CORNER,
        DoubleWishboneGeometrySpec,
        build_double_wishbone,
        DoubleWishboneSuspension,
    ),
    SuspensionDefinition(
        SuspensionType.DOUBLE_WISHBONE,
        Scope.AXLE,
        DoubleWishboneAxleGeometrySpec,
        build_double_wishbone_axle,
        AxleSuspension,
    ),
    SuspensionDefinition(
        SuspensionType.MACPHERSON,
        Scope.CORNER,
        MacPhersonGeometrySpec,
        build_macpherson,
        MacPhersonSuspension,
    ),
    SuspensionDefinition(
        SuspensionType.MACPHERSON,
        Scope.AXLE,
        MacPhersonAxleGeometrySpec,
        build_macpherson_axle,
        AxleSuspension,
    ),
)

_definitions_by_key = {
    (definition.type_key.value, definition.scope): definition
    for definition in SUSPENSION_DEFINITIONS
}

SUSPENSION_REGISTRY = MappingProxyType(_definitions_by_key)


def get_suspension_definition(
    type_key: str | SuspensionType,
    scope: Scope = Scope.CORNER,
) -> SuspensionDefinition | None:
    """Return the complete definition for an architecture and scope."""
    return SUSPENSION_REGISTRY.get((str(type_key), scope))


def get_suspension_class(
    type_key: str | SuspensionType,
    scope: Scope = Scope.CORNER,
) -> SuspensionClass | None:
    """Return the runtime class registered for an architecture and scope."""
    definition = get_suspension_definition(type_key, scope)
    return None if definition is None else definition.suspension_type


def list_supported_types() -> list[str]:
    """Return every supported public architecture key in sorted order."""
    return sorted({type_key for type_key, _ in SUSPENSION_REGISTRY})
