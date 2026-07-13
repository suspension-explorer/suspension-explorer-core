"""Build composed suspensions from validated geometry specifications."""

from typing import cast

from kinematics.core.elements import RockerPickup, RockerPickupType
from kinematics.core.enums import (
    ActuationType,
    ArbType,
    CornerSpringType,
    HeaveLinkType,
    PointID,
    ShimType,
)
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.schema.geometry import (
    ActuationSpec,
    AxleHardpointsSpec,
    CornerSpringSpec,
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
)
from kinematics.core.suspensions.axle import DoubleWishboneAxleSuspension
from kinematics.core.suspensions.axle.mechanisms import (
    ArbNone,
    ArbTBar,
    ArbUBar,
    AxleArb,
    AxleHeaveLink,
    HeaveLinkNone,
    HeaveLinkRockerToRocker,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.corner.mechanisms import (
    Actuation,
    ActuationDirect,
    ActuationPushrodRocker,
    CornerSpring,
    CornerSpringCoilover,
    CornerSpringNone,
    CornerSpringTorsionBar,
)


def build_suspension(spec: GeometrySpecBase) -> Suspension:
    """Construct a suspension using the registered architecture definition."""
    from kinematics.core.suspensions.registry import get_suspension_definition

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
    """Build one double-wishbone corner with composed mechanisms."""
    typed = cast(DoubleWishboneGeometrySpec, spec)
    actuation = build_actuation(typed.actuation)
    spring = build_corner_spring(typed.spring)
    return _build_corner(typed, actuation, spring)


def build_double_wishbone_axle(spec: GeometrySpecBase) -> Suspension:
    """Build two composed corners and their shared axle mechanisms."""
    typed = cast(DoubleWishboneAxleGeometrySpec, spec)
    side_points = _build_axle_side_points(typed.hardpoints)

    external_pickups: list[RockerPickup] = []
    anti_roll_droplink_points: dict[Side, Point3] = {}
    if typed.anti_roll.type in (ArbType.U_BAR, ArbType.T_BAR):
        external_pickups.append(
            RockerPickup(PointID.DROPLINK_ROCKER, RockerPickupType.DROPLINK)
        )
        droplink_point_id = (
            PointID.DROPLINK_U_BAR
            if typed.anti_roll.type is ArbType.U_BAR
            else PointID.DROPLINK_T_BAR
        )
        for side, points in side_points.items():
            try:
                anti_roll_droplink_points[side] = points.pop(droplink_point_id)
            except KeyError as error:
                mechanism_name = typed.anti_roll.type.value.replace("_", "-")
                raise ValueError(
                    f"{side.name} {mechanism_name} requires {droplink_point_id.name}"
                ) from error

    if typed.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER:
        external_pickups.append(
            RockerPickup(
                PointID.HEAVE_LINK_ROCKER,
                RockerPickupType.HEAVE_LINK,
            )
        )

    actuation = build_actuation(
        typed.corner.actuation,
        external_pickups=tuple(external_pickups),
    )
    spring = build_corner_spring(typed.corner.spring)
    _check_shim_support(typed.config, DoubleWishboneSuspension)

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
            hardpoints=cast("dict[PointKey, Point3]", points),
            config=side_configs[side],
            actuation=actuation,
            spring=spring,
        )
        for side, points in side_points.items()
    }

    anti_roll = build_anti_roll(
        typed,
        anti_roll_droplink_points,
    )
    heave_link = build_heave_link(typed)
    return DoubleWishboneAxleSuspension(
        name=typed.name,
        version=typed.version,
        units=typed.units,
        side=Side.CENTER,
        hardpoints={},
        config=typed.config,
        corners=corners,
        anti_roll=anti_roll,
        heave_link=heave_link,
    )


def build_actuation(
    spec: ActuationSpec,
    *,
    external_pickups: tuple[RockerPickup, ...] = (),
) -> Actuation:
    """Build one typed corner actuation mechanism."""
    if spec.type is ActuationType.DIRECT:
        if external_pickups:
            raise ValueError("Direct actuation does not accept rocker pickups")
        return ActuationDirect()
    if spec.type is ActuationType.PUSHROD_ROCKER:
        return ActuationPushrodRocker(external_pickups=external_pickups)
    raise TypeError(f"Unsupported actuation type: {spec.type}")


def build_corner_spring(spec: CornerSpringSpec) -> CornerSpring:
    """Build one typed corner spring mechanism."""
    if spec.type is CornerSpringType.NONE:
        return CornerSpringNone()
    if spec.type is CornerSpringType.COILOVER:
        return CornerSpringCoilover()
    if spec.type is CornerSpringType.TORSION_BAR:
        return CornerSpringTorsionBar()
    raise TypeError(f"Unsupported corner spring type: {spec.type}")


def build_anti_roll(
    spec: DoubleWishboneAxleGeometrySpec,
    droplink_points: dict[Side, Point3],
) -> AxleArb:
    """Build the selected shared anti-roll mechanism."""
    center_points = _copy_points(spec.hardpoints.center)
    if spec.anti_roll.type is ArbType.NONE:
        if center_points:
            raise ValueError(
                "Axle without anti-roll hardware does not accept center points"
            )
        return ArbNone()
    if spec.anti_roll.type is ArbType.U_BAR:
        return ArbUBar(
            center_points=center_points,
            droplink_points=droplink_points,
        )
    if spec.anti_roll.type is ArbType.T_BAR:
        return ArbTBar(
            center_points=center_points,
            droplink_points=droplink_points,
        )
    raise TypeError(f"Unsupported anti-roll type: {spec.anti_roll.type}")


def build_heave_link(spec: DoubleWishboneAxleGeometrySpec) -> AxleHeaveLink:
    """Build the selected shared heave mechanism."""
    if spec.heave_link.type is HeaveLinkType.NONE:
        return HeaveLinkNone()
    if spec.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER:
        return HeaveLinkRockerToRocker()
    raise TypeError(f"Unsupported heave-link type: {spec.heave_link.type}")


def _build_corner(
    spec: DoubleWishboneGeometrySpec,
    actuation: Actuation,
    spring: CornerSpring,
) -> DoubleWishboneSuspension:
    """Build one corner after mechanism-aware point validation."""
    _validate_side_signs(spec.hardpoints, spec.side)
    _check_shim_support(spec.config, DoubleWishboneSuspension)
    return DoubleWishboneSuspension(
        name=spec.name,
        version=spec.version,
        units=spec.units,
        side=spec.side,
        hardpoints={
            point: position.copy() for point, position in spec.hardpoints.items()
        },
        config=spec.config,
        actuation=actuation,
        spring=spring,
    )


def _build_axle_side_points(
    blocks: AxleHardpointsSpec,
) -> dict[Side, dict[PointID, Point3]]:
    """Copy explicit axle corners or mirror one explicitly sided source."""
    if blocks.is_explicit:
        if blocks.left is None or blocks.right is None:
            raise ValueError(
                "Explicit axle hardpoints require both left and right point maps."
            )
        side_points = {
            Side.LEFT: _copy_points(blocks.left),
            Side.RIGHT: _copy_points(blocks.right),
        }
    else:
        if blocks.points is None:
            raise ValueError("Mirrored axle hardpoints require a source point map.")
        source_side = blocks.side
        if source_side not in (Side.LEFT, Side.RIGHT):
            raise ValueError(
                "Mirrored axle hardpoints require a left or right source side."
            )
        source_points = _copy_points(blocks.points)
        other_side = Side.RIGHT if source_side is Side.LEFT else Side.LEFT
        side_points = {
            source_side: source_points,
            other_side: _mirror_hardpoints(source_points),
        }
    for side, points in side_points.items():
        _validate_side_signs(points, side)
    return side_points


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


def _check_shim_support(
    config: SuspensionConfig,
    cls: type[Suspension],
) -> None:
    """Reject a camber shim config on an architecture without shim support."""
    if (
        config.camber_shim is not None
        and ShimType.OUTBOARD_CAMBER not in cls.SUPPORTED_SHIMS
    ):
        raise ValueError(
            f"Suspension type '{cls.TYPE_KEY}' does not support outboard camber shims"
        )
