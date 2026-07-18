"""
Renderer-neutral, name-keyed geometry derived from suspension assemblies.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.elements import (
    ElementType,
    RackElement,
    RigidLinkElement,
    RockerElement,
    SuspensionElement,
    TBarElement,
    TorsionElement,
    UprightElement,
    VariableLengthLinkElement,
    WheelElement,
)
from kinematics.core.export import flatten_positions
from kinematics.core.primitives.point_ref import PointKey, point_key_name
from kinematics.core.schema.config import SuspensionConfig


@dataclass(frozen=True)
class AxisProjection:
    """
    Presentation point projected onto a physical rotation axis.
    """

    point: PointKey
    rotation_axis: tuple[PointKey, PointKey]


@dataclass(frozen=True)
class PointMidpoint:
    """
    Presentation midpoint of two physical element points.
    """

    point_a: PointKey
    point_b: PointKey


type ElementPathPoint = PointKey | AxisProjection | PointMidpoint


@dataclass(frozen=True)
class ElementPath:
    """
    Ordered renderer-neutral geometry for one part of an element.
    """

    points: tuple[ElementPathPoint, ...]
    type: ElementType
    label: str


@dataclass(frozen=True)
class NamedElementPath:
    """
    One element path resolved to stable public point names.
    """

    points: tuple[str, ...]
    type: ElementType
    label: str


@dataclass(frozen=True)
class WheelDimensions:
    """
    Physical tire and rim dimensions in mm.
    """

    radius: float
    width: float
    rim_radius: float


@dataclass(frozen=True)
class WheelReferences:
    """
    Public point names defining one wheel's position and orientation.
    """

    center: str
    inboard: str
    outboard: str
    axle_inboard: str
    axle_outboard: str
    contact_patch: str


def axis_projection_name(projection: AxisProjection) -> str:
    """
    Return a stable public name for a point projected onto an axis.
    """
    axis_names = sorted(point_key_name(point) for point in projection.rotation_axis)
    return (
        f"{point_key_name(projection.point)}_axis_projection_"
        f"{axis_names[0]}_{axis_names[1]}"
    )


def point_midpoint_name(midpoint: PointMidpoint) -> str:
    """
    Return a stable public name for the midpoint of two physical points.
    """
    point_names = sorted(
        (point_key_name(midpoint.point_a), point_key_name(midpoint.point_b))
    )
    return f"{point_names[0]}_{point_names[1]}_midpoint"


def _path_point_name(point: ElementPathPoint) -> str:
    """
    Resolve a physical or projected path point to its public name.
    """
    if isinstance(point, AxisProjection):
        return axis_projection_name(point)
    if isinstance(point, PointMidpoint):
        return point_midpoint_name(point)
    return point_key_name(point)


def _element_paths(
    element: SuspensionElement,
    torsion_bar_axes: set[tuple[PointKey, PointKey]],
) -> tuple[ElementPath, ...]:
    """
    Derive renderer-neutral geometry from one physical element.
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
            ElementPath(segment, ElementType.UPRIGHT, element.label)
            for segment in element.segments
        )
    if isinstance(element, TorsionElement):
        points = element.rotation_axis
        if element.type is ElementType.ANTI_ROLL_BAR:
            points = (
                element.attachments[0],
                *element.rotation_axis,
                element.attachments[1],
            )
        return (ElementPath(points, element.type, element.label),)
    if isinstance(element, TBarElement):
        midpoint = PointMidpoint(
            element.left_attachment,
            element.right_attachment,
        )
        return (
            ElementPath(
                (element.pivot, midpoint),
                ElementType.ANTI_ROLL_BAR,
                element.label,
            ),
            ElementPath(
                (element.left_attachment, midpoint, element.right_attachment),
                ElementType.ANTI_ROLL_BAR,
                element.label,
            ),
        )
    if isinstance(element, RockerElement):
        paths: list[ElementPath] = []
        if element.rotation_axis not in torsion_bar_axes:
            paths.append(
                ElementPath(
                    element.rotation_axis,
                    ElementType.ROCKER,
                    f"{element.label} Axis",
                )
            )
        for pickup in element.pickups:
            pickup_name = pickup.type.value.replace("_", " ").title()
            paths.append(
                ElementPath(
                    (
                        pickup.point,
                        AxisProjection(pickup.point, element.rotation_axis),
                    ),
                    ElementType.ROCKER,
                    f"{element.label} {pickup_name} Arm",
                )
            )
        return tuple(paths)
    if isinstance(element, WheelElement):
        return (
            ElementPath(
                (element.contact_patch,),
                ElementType.CONTACT_PATCH,
                f"{element.label} Contact Patch",
            ),
        )
    raise TypeError(f"Unsupported suspension element: {type(element)!r}")


def element_paths(assembly: SuspensionAssembly) -> tuple[ElementPath, ...]:
    """
    Derive renderer-neutral paths for an assembly's physical elements.
    """
    torsion_bar_axes = {
        axis
        for element in assembly.elements
        if isinstance(element, TorsionElement)
        and element.type is ElementType.TORSION_BAR
        for axis in (
            element.rotation_axis,
            (element.rotation_axis[1], element.rotation_axis[0]),
        )
    }
    return tuple(
        path
        for element in assembly.elements
        for path in _element_paths(element, torsion_bar_axes)
    )


def named_element_paths(assembly: SuspensionAssembly) -> list[NamedElementPath]:
    """
    Resolve assembly element paths to stable public point names.
    """
    return [
        NamedElementPath(
            points=tuple(_path_point_name(point) for point in path.points),
            type=path.type,
            label=path.label,
        )
        for path in element_paths(assembly)
    ]


def named_point_keys(assembly: SuspensionAssembly) -> list[str]:
    """
    Return every physical and projected position name in stable order.
    """
    names = [point_key_name(point) for point in assembly.referenced_point_keys]
    names.extend(
        axis_projection_name(projection) for projection in _axis_projections(assembly)
    )
    names.extend(
        point_midpoint_name(midpoint) for midpoint in _point_midpoints(assembly)
    )
    return names


def _axis_projections(assembly: SuspensionAssembly) -> tuple[AxisProjection, ...]:
    """
    Return unique projected points in assembly path order.
    """
    projections: list[AxisProjection] = []
    seen: set[AxisProjection] = set()
    for path in element_paths(assembly):
        for point in path.points:
            if isinstance(point, AxisProjection) and point not in seen:
                projections.append(point)
                seen.add(point)
    return tuple(projections)


def _point_midpoints(assembly: SuspensionAssembly) -> tuple[PointMidpoint, ...]:
    """
    Return unique midpoint references in assembly path order.
    """
    midpoints: list[PointMidpoint] = []
    seen: set[PointMidpoint] = set()
    for path in element_paths(assembly):
        for point in path.points:
            if isinstance(point, PointMidpoint) and point not in seen:
                midpoints.append(point)
                seen.add(point)
    return tuple(midpoints)


def resolve_positions(
    positions: Mapping[PointKey, object],
    assembly: SuspensionAssembly,
) -> dict[str, tuple[float, float, float]]:
    """
    Resolve one solver state to all named physical and projected positions.

    Raises:
        ValueError: If an element point is missing or a projection axis is
            degenerate.
    """
    missing = [
        point for point in assembly.referenced_point_keys if point not in positions
    ]
    if missing:
        raise ValueError(f"Cannot resolve missing assembly points: {missing!r}")

    named = flatten_positions(positions, assembly.referenced_point_keys)
    for projection in _axis_projections(assembly):
        point = np.asarray(named[point_key_name(projection.point)], dtype=np.float64)
        axis_start = np.asarray(
            named[point_key_name(projection.rotation_axis[0])],
            dtype=np.float64,
        )
        axis_end = np.asarray(
            named[point_key_name(projection.rotation_axis[1])],
            dtype=np.float64,
        )
        axis_direction = axis_end - axis_start
        axis_length_sq = float(np.dot(axis_direction, axis_direction))
        if axis_length_sq <= 0.0:
            raise ValueError(
                "Cannot project onto a zero-length rotation axis: "
                f"{projection.rotation_axis!r}"
            )

        point_from_axis = point - axis_start
        axis_parameter = float(np.dot(point_from_axis, axis_direction)) / axis_length_sq
        projected = axis_start + axis_parameter * axis_direction
        named[axis_projection_name(projection)] = (
            float(projected[0]),
            float(projected[1]),
            float(projected[2]),
        )
    for midpoint in _point_midpoints(assembly):
        point_a = np.asarray(named[point_key_name(midpoint.point_a)], dtype=np.float64)
        point_b = np.asarray(named[point_key_name(midpoint.point_b)], dtype=np.float64)
        position = point_a + (point_b - point_a) / 2.0
        named[point_midpoint_name(midpoint)] = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
    return named


def wheel_dimensions(config: SuspensionConfig | None) -> WheelDimensions | None:
    """
    Return physical tire dimensions, or None when no config is available.
    """
    if config is None:
        return None
    tire = config.wheel.tire
    return WheelDimensions(
        radius=float(tire.nominal_radius),
        width=float(tire.section_width),
        rim_radius=float(tire.rim_diameter_mm) / 2.0,
    )


def wheel_references(assembly: SuspensionAssembly) -> list[WheelReferences]:
    """
    Return public point names for every wheel in the assembly.
    """
    return [
        WheelReferences(
            center=point_key_name(wheel.center),
            inboard=point_key_name(wheel.inboard),
            outboard=point_key_name(wheel.outboard),
            axle_inboard=point_key_name(wheel.axle_inboard),
            axle_outboard=point_key_name(wheel.axle_outboard),
            contact_patch=point_key_name(wheel.contact_patch),
        )
        for wheel in assembly.wheels
    ]
