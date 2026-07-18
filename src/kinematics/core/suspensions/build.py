"""Build composed suspensions from validated geometry specifications."""

from collections.abc import Mapping
from typing import TypeVar, cast

from kinematics.core.elements import RockerPickup, RockerPickupType
from kinematics.core.enums import (
    ActuationType,
    ArbType,
    CornerSpringType,
    HeaveLinkType,
    MountBody,
    PointID,
    ShimType,
)
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.config import CornerConfig, SuspensionConfig
from kinematics.core.schema.geometry import (
    ActuationSpec,
    AxleGeometrySpecBase,
    AxleHardpointsSpec,
    CornerSpringSpec,
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneGeometrySpec,
    GeometrySpecBase,
    MacPhersonAxleGeometrySpec,
    MacPhersonGeometrySpec,
)
from kinematics.core.suspensions.axle import AxleSuspension
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
from kinematics.core.suspensions.corner import (
    CornerSuspension,
    DoubleWishboneSuspension,
    MacPhersonSuspension,
)
from kinematics.core.suspensions.corner.mechanisms import (
    Actuation,
    ActuationDirect,
    ActuationPushrodRocker,
    CornerSpring,
    CornerSpringCoilover,
    CornerSpringNone,
    CornerSpringTorsionBar,
)


def build_double_wishbone(spec: GeometrySpecBase) -> Suspension:
    """Build one double-wishbone corner with composed mechanisms."""
    typed = cast(DoubleWishboneGeometrySpec, spec)
    return _build_double_wishbone_corner(typed)


def _build_double_wishbone_corner(
    spec: DoubleWishboneGeometrySpec,
    external_pickups: tuple[RockerPickup, ...] = (),
) -> DoubleWishboneSuspension:
    """Build one double-wishbone corner with optional axle attachments."""
    actuation = build_actuation(
        spec.actuation,
        mount_bodies=DoubleWishboneSuspension.MOUNT_BODIES,
        external_pickups=external_pickups,
    )
    spring = build_corner_spring(spec.spring)
    return _build_corner(spec, actuation, spring)


def build_macpherson(spec: GeometrySpecBase) -> Suspension:
    """Build one MacPherson strut corner."""
    typed = cast(MacPhersonGeometrySpec, spec)
    _validate_side_signs(typed.hardpoints, typed.side)
    _check_shim_support(typed.config, MacPhersonSuspension)
    return MacPhersonSuspension(
        name=typed.name,
        version=typed.version,
        units=typed.units,
        side=typed.side,
        hardpoints={
            point: position.copy() for point, position in typed.hardpoints.items()
        },
        config=typed.config,
    )


def build_double_wishbone_axle(spec: GeometrySpecBase) -> Suspension:
    """Build a double-wishbone axle with composed shared hardware."""
    typed = cast(DoubleWishboneAxleGeometrySpec, spec)
    corner_setups = _double_wishbone_axle_corner_setups(typed)
    side_points = _build_axle_side_points(typed.hardpoints)
    external_pickups, droplink_points = _extract_axle_pickups(typed, side_points)

    corners: dict[Side, CornerSuspension] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner_setup = corner_setups[side]
        corner_geometry = DoubleWishboneGeometrySpec(
            name=f"{typed.name}_{side.name.lower()}",
            version=typed.version,
            units=typed.units,
            side=side,
            config=SuspensionConfig.from_parts(
                typed.vehicle_config,
                typed.axle_config,
                corner_setup,
            ),
            actuation=typed.axle_config.actuation,
            spring=typed.axle_config.spring,
            hardpoints=side_points[side],
        )
        corners[side] = _build_double_wishbone_corner(
            corner_geometry,
            external_pickups,
        )
    return _assemble_axle(typed, corners, droplink_points)


def build_macpherson_axle(spec: GeometrySpecBase) -> Suspension:
    """Build a MacPherson axle from a left and optional explicit right corner."""
    typed = cast(MacPhersonAxleGeometrySpec, spec)
    side_points = _build_axle_side_points(typed.hardpoints)
    corners: dict[Side, CornerSuspension] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner_geometry = MacPhersonGeometrySpec(
            name=f"{typed.name}_{side.name.lower()}",
            version=typed.version,
            units=typed.units,
            side=side,
            config=SuspensionConfig.from_parts(
                typed.vehicle_config,
                typed.axle_config,
                CornerConfig(),
            ),
            hardpoints=side_points[side],
        )
        corners[side] = cast(CornerSuspension, build_macpherson(corner_geometry))
    return _assemble_axle(typed, corners, {})


def _extract_axle_pickups(
    spec: AxleGeometrySpecBase,
    side_points: dict[Side, dict[PointID, Point3]],
) -> tuple[tuple[RockerPickup, ...], dict[Side, Point3]]:
    """Collect rocker pickups and droplink points for shared hardware."""
    external_pickups: list[RockerPickup] = []
    droplink_points: dict[Side, Point3] = {}
    if spec.axle_config.anti_roll.type in (ArbType.U_BAR, ArbType.T_BAR):
        external_pickups.append(
            RockerPickup(PointID.DROPLINK_ROCKER, RockerPickupType.DROPLINK)
        )
        droplink_point_id = (
            PointID.DROPLINK_U_BAR
            if spec.axle_config.anti_roll.type is ArbType.U_BAR
            else PointID.DROPLINK_T_BAR
        )
        for side, points in side_points.items():
            try:
                droplink_points[side] = points.pop(droplink_point_id)
            except KeyError as error:
                mechanism_name = spec.axle_config.anti_roll.type.value.replace("_", "-")
                raise ValueError(
                    f"{side.name} {mechanism_name} requires {droplink_point_id.name}"
                ) from error

    if spec.axle_config.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER:
        external_pickups.append(
            RockerPickup(
                PointID.HEAVE_LINK_ROCKER,
                RockerPickupType.HEAVE_LINK,
            )
        )
    return tuple(external_pickups), droplink_points


def _assemble_axle(
    spec: AxleGeometrySpecBase,
    corners: dict[Side, CornerSuspension],
    droplink_points: dict[Side, Point3],
) -> AxleSuspension:
    """Compose built corners with shared axle mechanisms."""
    return AxleSuspension(
        type_key=spec.type,
        name=spec.name,
        version=spec.version,
        units=spec.units,
        side=Side.CENTER,
        hardpoints={},
        config=corners[Side.LEFT].config,
        corners=corners,
        anti_roll=build_anti_roll(spec, droplink_points),
        heave_link=build_heave_link(spec),
    )


def build_actuation(
    spec: ActuationSpec,
    *,
    mount_bodies: Mapping[MountBody, tuple[PointID, ...]],
    external_pickups: tuple[RockerPickup, ...] = (),
) -> Actuation:
    """
    Build one typed corner actuation mechanism.

    The attachment bodies come from the locating architecture; the geometry
    spec always selects which body carries the moving pickup.
    """
    if spec.mount not in mount_bodies:
        raise ValueError(
            f"Architecture does not provide the '{spec.mount}' mounting body"
        )
    if spec.type is ActuationType.DIRECT:
        if external_pickups:
            raise ValueError("Direct actuation does not accept rocker pickups")
        return ActuationDirect(spring_pickup_body=mount_bodies[spec.mount])
    if spec.type is ActuationType.PUSHROD_ROCKER:
        return ActuationPushrodRocker(
            pushrod_outboard_body=mount_bodies[spec.mount],
            external_pickups=external_pickups,
        )
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
    spec: AxleGeometrySpecBase,
    droplink_points: dict[Side, Point3],
) -> AxleArb:
    """Build the selected shared anti-roll mechanism."""
    center_points = _copy_points(spec.hardpoints.center)
    if spec.axle_config.anti_roll.type is ArbType.NONE:
        if center_points:
            raise ValueError(
                "Axle without anti-roll hardware does not accept center points"
            )
        return ArbNone()
    if spec.axle_config.anti_roll.type is ArbType.U_BAR:
        return ArbUBar(
            center_points=center_points,
            droplink_points=droplink_points,
        )
    if spec.axle_config.anti_roll.type is ArbType.T_BAR:
        return ArbTBar(
            center_points=center_points,
            droplink_points=droplink_points,
        )
    raise TypeError(f"Unsupported anti-roll type: {spec.axle_config.anti_roll.type}")


def build_heave_link(spec: AxleGeometrySpecBase) -> AxleHeaveLink:
    """Build the selected shared heave mechanism."""
    if spec.axle_config.heave_link.type is HeaveLinkType.NONE:
        return HeaveLinkNone()
    if spec.axle_config.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER:
        return HeaveLinkRockerToRocker()
    raise TypeError(f"Unsupported heave-link type: {spec.axle_config.heave_link.type}")


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
    hardpoints: AxleHardpointsSpec,
) -> dict[Side, dict[PointID, Point3]]:
    """Return the authored left hardpoints and explicit or mirrored right map."""
    left = _copy_points(hardpoints.left)
    right = hardpoints.right
    if right is None:
        right_points = _mirror_hardpoints(left)
    else:
        right_points = _copy_points(right)
    return {Side.LEFT: left, Side.RIGHT: right_points}


def _double_wishbone_axle_corner_setups(
    spec: DoubleWishboneAxleGeometrySpec,
) -> dict[Side, CornerConfig]:
    """Return the authored left setup and explicit or mirrored right setup."""
    left = spec.axle_config.left_setup
    right = spec.axle_config.right_setup
    if right is None:
        right = _mirror_corner_config(left)
    return {Side.LEFT: left, Side.RIGHT: right}


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


C = TypeVar("C", bound=CornerConfig)


def _mirror_corner_config(config: C) -> C:
    """Mirror side-dependent setup geometry for the right corner."""
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
    cls: type[CornerSuspension],
) -> None:
    """Reject a camber shim config on an architecture without shim support."""
    if (
        config.camber_shim is not None
        and ShimType.OUTBOARD_CAMBER not in cls.SUPPORTED_SHIMS
    ):
        raise ValueError(
            f"Suspension type '{cls.TYPE_KEY}' does not support outboard camber shims"
        )
