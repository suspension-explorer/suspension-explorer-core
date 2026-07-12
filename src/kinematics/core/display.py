"""
Name-keyed analysis views derived from renderer-neutral suspension topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey, point_key_name
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.topology import LinkRole, SuspensionTopology

AXIS_FOOT_SUFFIX = "_axis_foot"


@dataclass(frozen=True)
class DisplayLink:
    """One name-keyed polyline in the display topology."""

    points: tuple[str, ...]
    role: LinkRole
    label: str


@dataclass(frozen=True)
class RockerDisplayGroup:
    """Display description for one rocker-equipped corner."""

    axis_front: str
    axis_rear: str
    pickups: tuple[str, ...]
    label_prefix: str


@dataclass(frozen=True)
class WheelDisplayDimensions:
    """Static tire dimensions in mm for drawing a wheel."""

    radius: float
    width: float
    rim_radius: float


@dataclass(frozen=True)
class WheelAnchorNames:
    """Name-keyed positions anchoring one displayed wheel."""

    center: str
    inboard: str
    outboard: str
    axle_inboard: str
    axle_outboard: str


def rocker_display_groups(topology: SuspensionTopology) -> list[RockerDisplayGroup]:
    """Convert typed rocker topology to public point names."""
    return [
        RockerDisplayGroup(
            axis_front=point_key_name(rocker.axis_front),
            axis_rear=point_key_name(rocker.axis_rear),
            pickups=tuple(point_key_name(point) for point in rocker.pickups),
            label_prefix=rocker.label_prefix,
        )
        for rocker in topology.rockers
    ]


def display_point_keys(topology: SuspensionTopology) -> tuple[PointKey, ...]:
    """Return all point keys required to resolve the display topology."""
    points = list(topology.output_points)
    seen = set(points)
    for link in topology.links:
        for key in link.points:
            if key not in seen:
                points.append(key)
                seen.add(key)
    for rocker in topology.rockers:
        for key in (rocker.axis_front, rocker.axis_rear, *rocker.pickups):
            if key not in seen:
                points.append(key)
                seen.add(key)
    return tuple(points)


def display_positions(
    positions: Mapping[PointKey, object],
    point_keys: tuple[PointKey, ...],
    rocker_groups: list[RockerDisplayGroup],
) -> dict[str, tuple[float, float, float]]:
    """Flatten positions to name keys and append synthetic rocker-axis feet."""
    named: dict[str, tuple[float, float, float]] = {}
    for key in point_keys:
        position = positions.get(key)
        if position is None:
            continue
        raw = extract_array(position)
        named[point_key_name(key)] = (float(raw[0]), float(raw[1]), float(raw[2]))

    for group in rocker_groups:
        _append_axis_feet(named, group)
    return named


def _append_axis_feet(
    named_positions: dict[str, tuple[float, float, float]],
    group: RockerDisplayGroup,
) -> None:
    """Append the perpendicular projection of each pickup onto its rocker axis."""
    axis_a = named_positions.get(group.axis_front)
    axis_b = named_positions.get(group.axis_rear)
    if axis_a is None or axis_b is None:
        return

    axis_origin = np.asarray(axis_a, dtype=np.float64)
    axis_direction = np.asarray(axis_b, dtype=np.float64) - axis_origin
    norm_sq = float(np.dot(axis_direction, axis_direction))
    if norm_sq <= 0.0:
        return

    for pickup in group.pickups:
        position = named_positions.get(pickup)
        if position is None:
            continue
        radius = np.asarray(position, dtype=np.float64) - axis_origin
        parameter = float(np.dot(radius, axis_direction)) / norm_sq
        foot = axis_origin + parameter * axis_direction
        named_positions[f"{pickup}{AXIS_FOOT_SUFFIX}"] = (
            float(foot[0]),
            float(foot[1]),
            float(foot[2]),
        )


def display_links(topology: SuspensionTopology) -> list[DisplayLink]:
    """Return name-keyed physical links and normalized rocker lever arms."""
    links = [
        DisplayLink(
            points=tuple(point_key_name(key) for key in link.points),
            role=link.role,
            label=link.label,
        )
        for link in topology.links
    ]

    for group in rocker_display_groups(topology):
        links.append(
            DisplayLink(
                points=(group.axis_front, group.axis_rear),
                role=LinkRole.ROCKER,
                label=f"{group.label_prefix}Rocker Axis",
            )
        )
        for pickup in group.pickups:
            arm = (
                "Droplink Arm"
                if pickup.endswith(PointID.DROPLINK_ROCKER.name.lower())
                else "Pushrod Arm"
            )
            links.append(
                DisplayLink(
                    points=(pickup, f"{pickup}{AXIS_FOOT_SUFFIX}"),
                    role=LinkRole.ROCKER,
                    label=f"{group.label_prefix}Rocker {arm}",
                )
            )
    return links


def wheel_display_dimensions(
    config: SuspensionConfig | None,
) -> WheelDisplayDimensions | None:
    """Return static tire dimensions, or None when no config is available."""
    if config is None:
        return None
    tire = config.wheel.tire
    return WheelDisplayDimensions(
        radius=float(tire.nominal_radius),
        width=float(tire.section_width),
        rim_radius=float(tire.rim_diameter_mm) / 2.0,
    )


def wheel_anchor_names(topology: SuspensionTopology) -> list[WheelAnchorNames]:
    """Return name-keyed drawing anchors for every wheel."""
    return [
        WheelAnchorNames(
            center=point_key_name(anchors.center),
            inboard=point_key_name(anchors.inboard),
            outboard=point_key_name(anchors.outboard),
            axle_inboard=point_key_name(anchors.axle_inboard),
            axle_outboard=point_key_name(anchors.axle_outboard),
        )
        for anchors in topology.wheels
    ]
