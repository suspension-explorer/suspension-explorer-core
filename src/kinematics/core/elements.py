"""
Physical suspension element declarations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from kinematics.core.primitives.enums import Axis
from kinematics.core.primitives.point_ref import PointKey


class ElementType(StrEnum):
    """
    Physical kind of suspension element geometry.
    """

    WISHBONE = "wishbone"
    UPRIGHT = "upright"
    TRACK_ROD = "track_rod"
    RACK = "rack"
    AXLE = "axle"
    CONTACT_PATCH = "contact_patch"
    PUSHROD = "pushrod"
    ROCKER = "rocker"
    SPRING_DAMPER = "spring_damper"
    ANTI_ROLL_BAR = "anti_roll_bar"
    TORSION_BAR = "torsion_bar"
    DROPLINK = "droplink"


class RockerPickupType(StrEnum):
    """
    Physical connection made at a rocker pickup.
    """

    PUSHROD = "pushrod"
    DROPLINK = "droplink"


@dataclass(frozen=True)
class RockerPickup:
    """
    One typed attachment point on a rocker.
    """

    point: PointKey
    type: RockerPickupType


@dataclass(frozen=True)
class AxisProjection:
    """
    Projection of one point onto a physical rotation axis.
    """

    point: PointKey
    rotation_axis: tuple[PointKey, PointKey]


type ElementPathPoint = PointKey | AxisProjection


@dataclass(frozen=True)
class ElementPath:
    """
    Ordered physical geometry for one part of a suspension element.
    """

    points: tuple[ElementPathPoint, ...]
    type: ElementType
    label: str

    def __post_init__(self) -> None:
        """
        Require at least one point in every element path.
        """
        if not self.points:
            raise ValueError("Element paths must contain at least one point")


@dataclass(frozen=True)
class SuspensionElement(ABC):
    """
    Base type for one physical element in a suspension assembly.
    """

    label: str

    @property
    @abstractmethod
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return every point referenced by this element.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RigidLinkElement(SuspensionElement):
    """
    A two-point link that preserves its design length.
    """

    type: ElementType
    point_a: PointKey
    point_b: PointKey

    def __post_init__(self) -> None:
        """
        Require a fixed-length link type.
        """
        valid_types = {
            ElementType.WISHBONE,
            ElementType.TRACK_ROD,
            ElementType.AXLE,
            ElementType.PUSHROD,
            ElementType.DROPLINK,
        }
        if self.type not in valid_types:
            raise ValueError(f"Invalid rigid-link element type: {self.type.value}")

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return both link endpoints.
        """
        return self.point_a, self.point_b


@dataclass(frozen=True)
class VariableLengthLinkElement(SuspensionElement):
    """
    A two-point link whose length is free to change.
    """

    type: ElementType
    point_a: PointKey
    point_b: PointKey

    def __post_init__(self) -> None:
        """
        Require a variable-length link type.
        """
        if self.type is not ElementType.SPRING_DAMPER:
            raise ValueError(f"Invalid variable-length element type: {self.type.value}")

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return both link endpoints.
        """
        return self.point_a, self.point_b


@dataclass(frozen=True)
class RackElement(SuspensionElement):
    """
    A steering rack connecting left and right inner trackrod joints.
    """

    left_inner: PointKey
    right_inner: PointKey
    translation_axis: Axis

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return the rack joint points in left-to-right order.
        """
        return self.left_inner, self.right_inner


@dataclass(frozen=True)
class UprightElement(SuspensionElement):
    """
    Rigid upright hardpoints, attachments, and visible body segments.
    """

    hardpoints: tuple[PointKey, ...]
    attachments: tuple[PointKey, ...]
    segments: tuple[tuple[PointKey, PointKey], ...]

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return every upright point in declaration order without duplicates.
        """
        segment_points = tuple(point for segment in self.segments for point in segment)
        return tuple(dict.fromkeys(self.hardpoints + self.attachments + segment_points))


@dataclass(frozen=True)
class TorsionElement(SuspensionElement):
    """
    A torsion member with an arbitrary rotation axis and attachments.
    """

    type: ElementType
    rotation_axis: tuple[PointKey, PointKey]
    attachments: tuple[PointKey, ...]
    path: tuple[PointKey, ...]

    def __post_init__(self) -> None:
        """
        Require a torsion-element type.
        """
        valid_types = {ElementType.ANTI_ROLL_BAR, ElementType.TORSION_BAR}
        if self.type not in valid_types:
            raise ValueError(f"Invalid torsion element type: {self.type.value}")

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return axis, attachment, and path points.
        """
        return self.rotation_axis + self.attachments + self.path


@dataclass(frozen=True)
class RockerElement(SuspensionElement):
    """
    A rigid rocker rotating about an arbitrarily oriented axis.
    """

    rotation_axis: tuple[PointKey, PointKey]
    pickups: tuple[RockerPickup, ...]

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return the rocker axis and pickup points.
        """
        return self.rotation_axis + tuple(pickup.point for pickup in self.pickups)


@dataclass(frozen=True)
class WheelElement(SuspensionElement):
    """
    A wheel, hub axis, and contact patch.
    """

    center: PointKey
    inboard: PointKey
    outboard: PointKey
    axle_inboard: PointKey
    axle_outboard: PointKey
    contact_patch: PointKey

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """
        Return all wheel and hub reference points.
        """
        return (
            self.center,
            self.inboard,
            self.outboard,
            self.axle_inboard,
            self.axle_outboard,
            self.contact_patch,
        )


def element_paths(element: SuspensionElement) -> tuple[ElementPath, ...]:
    """
    Describe the renderer-neutral geometry of one physical element.
    """
    if isinstance(element, RigidLinkElement | VariableLengthLinkElement):
        return (
            ElementPath(
                points=(element.point_a, element.point_b),
                type=element.type,
                label=element.label,
            ),
        )
    if isinstance(element, RackElement):
        return (
            ElementPath(
                points=(element.left_inner, element.right_inner),
                type=ElementType.RACK,
                label=element.label,
            ),
        )
    if isinstance(element, UprightElement):
        return tuple(
            ElementPath(
                points=segment,
                type=ElementType.UPRIGHT,
                label=element.label,
            )
            for segment in element.segments
        )
    if isinstance(element, TorsionElement):
        return (
            ElementPath(
                points=element.path,
                type=element.type,
                label=element.label,
            ),
        )
    if isinstance(element, RockerElement):
        paths = [
            ElementPath(
                points=element.rotation_axis,
                type=ElementType.ROCKER,
                label=f"{element.label} Axis",
            )
        ]
        for pickup in element.pickups:
            pickup_name = pickup.type.value.replace("_", " ").title()
            paths.append(
                ElementPath(
                    points=(
                        pickup.point,
                        AxisProjection(pickup.point, element.rotation_axis),
                    ),
                    type=ElementType.ROCKER,
                    label=f"{element.label} {pickup_name} Arm",
                )
            )
        return tuple(paths)
    if isinstance(element, WheelElement):
        return (
            ElementPath(
                points=(element.contact_patch,),
                type=ElementType.CONTACT_PATCH,
                label=f"{element.label} Contact Patch",
            ),
        )
    raise TypeError(f"Unsupported suspension element: {type(element)!r}")


def map_element_points(
    element: SuspensionElement,
    transform: Callable[[PointKey], PointKey],
    *,
    label: str | None = None,
) -> SuspensionElement:
    """
    Map every point reference while preserving the concrete element type.
    """
    mapped_label = element.label if label is None else label
    if isinstance(element, RigidLinkElement | VariableLengthLinkElement):
        return replace(
            element,
            label=mapped_label,
            point_a=transform(element.point_a),
            point_b=transform(element.point_b),
        )
    if isinstance(element, RackElement):
        return replace(
            element,
            label=mapped_label,
            left_inner=transform(element.left_inner),
            right_inner=transform(element.right_inner),
        )
    if isinstance(element, UprightElement):
        return replace(
            element,
            label=mapped_label,
            hardpoints=tuple(transform(point) for point in element.hardpoints),
            attachments=tuple(transform(point) for point in element.attachments),
            segments=tuple(
                (transform(start), transform(end)) for start, end in element.segments
            ),
        )
    if isinstance(element, TorsionElement):
        return replace(
            element,
            label=mapped_label,
            rotation_axis=tuple(transform(point) for point in element.rotation_axis),
            attachments=tuple(transform(point) for point in element.attachments),
            path=tuple(transform(point) for point in element.path),
        )
    if isinstance(element, RockerElement):
        return replace(
            element,
            label=mapped_label,
            rotation_axis=tuple(transform(point) for point in element.rotation_axis),
            pickups=tuple(
                replace(pickup, point=transform(pickup.point))
                for pickup in element.pickups
            ),
        )
    if isinstance(element, WheelElement):
        return replace(
            element,
            label=mapped_label,
            center=transform(element.center),
            inboard=transform(element.inboard),
            outboard=transform(element.outboard),
            axle_inboard=transform(element.axle_inboard),
            axle_outboard=transform(element.axle_outboard),
            contact_patch=transform(element.contact_patch),
        )
    raise TypeError(f"Unsupported suspension element: {type(element)!r}")
