"""
Renderer-neutral, name-keyed geometry derived from suspension assemblies.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.elements import AxisProjection, ElementPathPoint, ElementType
from kinematics.core.export import flatten_positions
from kinematics.core.primitives.point_ref import PointKey, point_key_name
from kinematics.core.schema.config import SuspensionConfig


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


def _path_point_name(point: ElementPathPoint) -> str:
    """
    Resolve a physical or projected path point to its public name.
    """
    if isinstance(point, AxisProjection):
        return axis_projection_name(point)
    return point_key_name(point)


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
        for path in assembly.element_paths
    ]


def named_point_keys(assembly: SuspensionAssembly) -> list[str]:
    """
    Return every physical and projected position name in stable order.
    """
    names = [point_key_name(point) for point in assembly.referenced_point_keys]
    names.extend(
        axis_projection_name(projection) for projection in _axis_projections(assembly)
    )
    return names


def _axis_projections(assembly: SuspensionAssembly) -> tuple[AxisProjection, ...]:
    """
    Return unique projected points in assembly path order.
    """
    projections: list[AxisProjection] = []
    seen: set[AxisProjection] = set()
    for path in assembly.element_paths:
        for point in path.points:
            if isinstance(point, AxisProjection) and point not in seen:
                projections.append(point)
                seen.add(point)
    return tuple(projections)


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
