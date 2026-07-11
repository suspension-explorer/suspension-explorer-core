"""
Display topology normalized for interactive front-ends.

Solver visualization links use point keys and represent a rocker as a fan.
Interactive renderers instead receive string-keyed positions and links, with
each rocker represented by its axis and true perpendicular lever arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.geometry import extract_array
from kinematics.core.point_ref import PointKey, point_key_name
from kinematics.schema.config import SuspensionConfig
from kinematics.suspensions.base import Suspension

AXIS_FOOT_SUFFIX = "_axis_foot"
ROCKER_DISPLAY_COLOR = "mediumvioletred"


@dataclass(frozen=True)
class DisplayLink:
    """One name-keyed polyline in the display topology."""

    points: tuple[str, ...]
    color: str
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


def rocker_display_groups(suspension: Suspension) -> list[RockerDisplayGroup]:
    """Describe the axis and rigid pickups of every rocker in a suspension."""
    corners = getattr(suspension, "corners", None)
    if corners is None:
        rocker_corners = (
            [("", "", suspension)] if getattr(suspension, "has_rocker", False) else []
        )
    else:
        rocker_corners = [
            (f"{side.name.lower()}_", f"{side.name.title()} ", corner)
            for side, corner in corners.items()
            if corner.has_rocker
        ]

    groups: list[RockerDisplayGroup] = []
    for key_prefix, label_prefix, corner in rocker_corners:
        pickups = [f"{key_prefix}{PointID.PUSHROD_INBOARD.name.lower()}"]
        if PointID.DROPLINK_ROCKER in corner.hardpoints:
            pickups.append(f"{key_prefix}{PointID.DROPLINK_ROCKER.name.lower()}")
        groups.append(
            RockerDisplayGroup(
                axis_front=f"{key_prefix}{PointID.ROCKER_AXIS_FRONT.name.lower()}",
                axis_rear=f"{key_prefix}{PointID.ROCKER_AXIS_REAR.name.lower()}",
                pickups=tuple(pickups),
                label_prefix=label_prefix,
            )
        )
    return groups


def display_point_keys(suspension: Suspension) -> tuple[PointKey, ...]:
    """Return all point keys required to resolve the display topology."""
    points = list(suspension.output_points())
    seen = set(points)
    for link in suspension.get_visualization_links():
        for key in link.points:
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


def display_links(suspension: Suspension) -> list[DisplayLink]:
    """Return name-keyed links with rocker fans normalized to true lever arms."""
    links = [
        DisplayLink(
            points=tuple(point_key_name(key) for key in link.points),
            color=link.color,
            label=link.label,
        )
        for link in suspension.get_visualization_links()
        if not link.label.endswith("Rocker")
    ]

    for group in rocker_display_groups(suspension):
        links.append(
            DisplayLink(
                points=(group.axis_front, group.axis_rear),
                color=ROCKER_DISPLAY_COLOR,
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
                    color=ROCKER_DISPLAY_COLOR,
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


def wheel_anchor_names(suspension: Suspension) -> list[WheelAnchorNames]:
    """Return name-keyed drawing anchors for every wheel."""
    return [
        WheelAnchorNames(
            center=point_key_name(anchors.center),
            inboard=point_key_name(anchors.inboard),
            outboard=point_key_name(anchors.outboard),
            axle_inboard=point_key_name(anchors.axle_inboard),
            axle_outboard=point_key_name(anchors.axle_outboard),
        )
        for anchors in suspension.wheel_visualization_anchors()
    ]
