"""Build concrete suspensions from validated geometry specifications."""

from typing import cast

from kinematics.core.enums import PointID, ShimType
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import Side
from kinematics.schema.config import SuspensionConfig
from kinematics.schema.geometry import (
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneCoiloverGeometrySpec,
    DoubleWishboneGeometrySpec,
    DoubleWishbonePushrodRockerArbGeometrySpec,
    DoubleWishbonePushrodRockerAxleGeometrySpec,
    DoubleWishbonePushrodRockerGeometrySpec,
    GeometrySpecBase,
)
from kinematics.suspensions.axle import (
    DoubleWishboneAxleSuspension,
    DoubleWishbonePushrodRockerAxleSuspension,
)
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.corner import (
    DoubleWishboneCoiloverSuspension,
    DoubleWishbonePushrodRockerArbSuspension,
    DoubleWishbonePushrodRockerSuspension,
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


def build_double_wishbone_pushrod_rocker(spec: GeometrySpecBase) -> Suspension:
    """Build an explicit pushrod-rocker corner."""
    typed = cast(DoubleWishbonePushrodRockerGeometrySpec, spec)
    return _build_corner(
        typed,
        DoubleWishbonePushrodRockerSuspension,
        spring_type=typed.spring.type,
    )


def build_double_wishbone_pushrod_rocker_arb(spec: GeometrySpecBase) -> Suspension:
    """Build a pushrod-rocker corner with a rocker-side ARB pickup."""
    typed = cast(DoubleWishbonePushrodRockerArbGeometrySpec, spec)
    return _build_corner(
        typed,
        DoubleWishbonePushrodRockerArbSuspension,
        spring_type=typed.spring.type,
    )


def build_double_wishbone_axle(spec: GeometrySpecBase) -> Suspension:
    """Build a basic two-corner double-wishbone axle."""
    typed = cast(DoubleWishboneAxleGeometrySpec, spec)
    _check_shim_support(typed.config, DoubleWishboneSuspension)
    side_points = _build_axle_side_points(typed.hardpoints)

    for points in side_points.values():
        _check_valid_points(points, DoubleWishboneSuspension)

    side_configs = {
        Side.LEFT: typed.config,
        Side.RIGHT: _mirror_config(typed.config),
    }
    corners = {
        side: DoubleWishboneSuspension(
            name=f"{typed.name}_{side.name.lower()}",
            version=typed.version,
            units=typed.units,
            side=side,
            hardpoints=points,
            config=side_configs[side],
        )
        for side, points in side_points.items()
    }
    return DoubleWishboneAxleSuspension(
        name=typed.name,
        version=typed.version,
        units=typed.units,
        side=Side.CENTER,
        hardpoints={},
        config=typed.config,
        corners=corners,
    )


def build_double_wishbone_pushrod_rocker_axle(
    spec: GeometrySpecBase,
) -> Suspension:
    """Build two ARB-ready rocker corners and their shared anti-roll bar."""
    typed = cast(DoubleWishbonePushrodRockerAxleGeometrySpec, spec)
    corner_type = DoubleWishbonePushrodRockerArbSuspension
    _check_shim_support(typed.config, corner_type)
    side_points = _build_axle_side_points(typed.hardpoints)
    droplink_arb_points: dict[Side, Point3] = {}
    for side, points in side_points.items():
        try:
            droplink_arb_points[side] = points.pop(PointID.DROPLINK_ARB)
        except KeyError as error:
            raise ValueError(
                f"{side.name} rocker axle requires DROPLINK_ARB"
            ) from error
        _check_valid_points(points, corner_type)

    center_points = _copy_points(typed.hardpoints.center)
    expected_center = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B}
    if set(center_points) != expected_center:
        names = ", ".join(sorted(point.name for point in expected_center))
        raise ValueError(f"Rocker axle center points must be exactly: {names}")

    side_configs = {
        Side.LEFT: typed.config,
        Side.RIGHT: _mirror_config(typed.config),
    }
    corners = {
        side: corner_type(
            name=f"{typed.name}_{side.name.lower()}",
            version=typed.version,
            units=typed.units,
            side=side,
            hardpoints=points,
            config=side_configs[side],
            spring_type=typed.spring.type,
        )
        for side, points in side_points.items()
    }
    return DoubleWishbonePushrodRockerAxleSuspension(
        name=typed.name,
        version=typed.version,
        units=typed.units,
        side=Side.CENTER,
        hardpoints={},
        config=typed.config,
        corners=corners,
        center_points=center_points,
        droplink_arb_points=droplink_arb_points,
    )


def _build_axle_side_points(blocks) -> dict[Side, dict[PointID, Point3]]:
    """Copy explicit axle corners or mirror one explicitly sided source."""
    if blocks.is_explicit:
        assert blocks.left is not None and blocks.right is not None
        side_points = {
            Side.LEFT: _copy_points(blocks.left),
            Side.RIGHT: _copy_points(blocks.right),
        }
    else:
        assert blocks.points is not None
        source_side = blocks.side
        assert source_side is not None
        source_points = _copy_points(blocks.points)
        other_side = Side.RIGHT if source_side is Side.LEFT else Side.LEFT
        side_points = {
            source_side: source_points,
            other_side: _mirror_hardpoints(source_points),
        }
    for side, points in side_points.items():
        _validate_side_signs(points, side)
    return side_points


def _build_corner(
    spec: DoubleWishboneGeometrySpec
    | DoubleWishboneCoiloverGeometrySpec
    | DoubleWishbonePushrodRockerGeometrySpec
    | DoubleWishbonePushrodRockerArbGeometrySpec,
    cls: type[DoubleWishboneSuspension],
    **kwargs: object,
) -> DoubleWishboneSuspension:
    """Build one concrete corner after exact point validation."""
    _check_valid_points(spec.hardpoints, cls)
    _validate_side_signs(spec.hardpoints, spec.side)
    _check_shim_support(spec.config, cls)
    return cls(
        name=spec.name,
        version=spec.version,
        units=spec.units,
        side=spec.side,
        hardpoints={
            point: position.copy() for point, position in spec.hardpoints.items()
        },
        config=spec.config,
        **kwargs,
    )


def _copy_points(points: dict[PointID, Point3]) -> dict[PointID, Point3]:
    """Copy a point map so built suspensions do not alias schema data."""
    return {point: position.copy() for point, position in points.items()}


def _validate_side_signs(points: dict[PointID, Point3], side: Side) -> None:
    """Require the axle-outboard Y sign to match the declared side."""
    axle_outboard = points.get(PointID.AXLE_OUTBOARD)
    if axle_outboard is None:
        return
    lateral_position = float(axle_outboard.data[1])
    if side is Side.LEFT and lateral_position <= 0.0:
        raise ValueError(
            "Side 'left' requires AXLE_OUTBOARD Y > 0 "
            f"(got {lateral_position}); check the hardpoint handedness."
        )
    if side is Side.RIGHT and lateral_position >= 0.0:
        raise ValueError(
            "Side 'right' requires AXLE_OUTBOARD Y < 0 "
            f"(got {lateral_position}); check the hardpoint handedness."
        )


def _mirror_point(point: Point3) -> Point3:
    """Reflect a point through the vehicle XZ plane."""
    x, y, z = point.data
    return Point3([float(x), -float(y), float(z)])


def _mirror_hardpoints(
    hardpoints: dict[PointID, Point3],
) -> dict[PointID, Point3]:
    """Reflect every hardpoint through the vehicle XZ plane."""
    return {point: _mirror_point(position) for point, position in hardpoints.items()}


def _mirror_config(config: SuspensionConfig) -> SuspensionConfig:
    """Mirror side-dependent camber-shim geometry for the right corner."""
    if config.camber_shim is None:
        return config
    shim = config.camber_shim
    normal = shim.shim_face_normal.data
    mirrored_shim = shim.model_copy(
        update={
            "shim_face_point_a": _mirror_point(shim.shim_face_point_a),
            "shim_face_point_b": _mirror_point(shim.shim_face_point_b),
            "shim_face_normal": Direction3(
                [float(normal[0]), -float(normal[1]), float(normal[2])]
            ),
        }
    )
    return config.model_copy(update={"camber_shim": mirrored_shim})


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
