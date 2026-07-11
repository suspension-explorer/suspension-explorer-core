"""Build concrete suspensions from validated geometry specifications."""

from typing import cast

from kinematics.core.enums import PointID, ShimType
from kinematics.core.geometry import Point3
from kinematics.schema.config import SuspensionConfig
from kinematics.schema.geometry import (
    DoubleWishboneCoiloverGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
)
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.corner import (
    DoubleWishboneCoiloverSuspension,
    DoubleWishboneSuspension,
)


def build_suspension(spec: GeometrySpecBase) -> Suspension:
    """Construct a suspension using the registered type definition."""
    from kinematics.suspensions.registry import get_suspension_definition

    definition = get_suspension_definition(spec.type)
    if definition is None:
        raise TypeError(f"Unsupported geometry spec type: {spec.type}")
    if not isinstance(spec, definition.spec_type):
        raise TypeError(
            f"Type '{spec.type}' requires {definition.spec_type.__name__}, "
            f"got {type(spec).__name__}."
        )
    return definition.build(spec)


def build_double_wishbone(spec: GeometrySpecBase) -> Suspension:
    """Build the basic double-wishbone corner."""
    typed = cast(DoubleWishboneGeometrySpec, spec)
    return _build_corner(typed, DoubleWishboneSuspension)


def build_double_wishbone_coilover(spec: GeometrySpecBase) -> Suspension:
    """Build the lower-wishbone-mounted coilover corner."""
    typed = cast(DoubleWishboneCoiloverGeometrySpec, spec)
    return _build_corner(typed, DoubleWishboneCoiloverSuspension)


def _build_corner(
    spec: DoubleWishboneGeometrySpec | DoubleWishboneCoiloverGeometrySpec,
    cls: type[DoubleWishboneSuspension],
) -> DoubleWishboneSuspension:
    """Build one concrete corner after exact point validation."""
    _check_valid_points(spec.hardpoints, cls)
    _check_shim_support(spec.config, cls)
    return cls(
        name=spec.name,
        version=spec.version,
        units=spec.units,
        hardpoints={
            point: position.copy() for point, position in spec.hardpoints.items()
        },
        config=spec.config,
    )


def _check_valid_points(points: dict[PointID, Point3], cls: type[Suspension]) -> None:
    """Reject points the concrete suspension class does not define."""
    unknown = set(points) - set(cls.all_valid_points())
    if unknown:
        names = ", ".join(sorted(point.name for point in unknown))
        raise ValueError(f"Invalid hardpoints for {cls.TYPE_KEY}: {names}")


def _check_shim_support(config: SuspensionConfig, cls: type[Suspension]) -> None:
    """Reject a camber shim config on a type that does not support shims."""
    if (
        config.camber_shim is not None
        and ShimType.OUTBOARD_CAMBER not in cls.SUPPORTED_SHIMS
    ):
        raise ValueError(
            f"Suspension type '{cls.TYPE_KEY}' does not support outboard camber shims"
        )
