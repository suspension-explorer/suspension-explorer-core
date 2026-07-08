"""
Display topology for interactive front-ends.

The solver-facing visualization types (kinematics.visualization.main) key
links on PointKey and include the triangular rocker "fan" used by the
matplotlib animation. Interactive viewers (e.g. a web 3D view) want a
flattened, name-keyed topology instead:

- every link vertex addressed by the same string key as the exported
  positions (``PointID.name`` / ``PointRef.name``), so a renderer joins
  positions to polylines with plain lookups and needs no per-suspension-type
  knowledge; and
- the rocker fan replaced by a rocker-axis line plus one true lever arm per
  pickup, drawn perpendicular from the pickup onto the axis. The arm's inner
  vertex is a synthetic "axis foot" position that is not a solver point, so
  it is published under a derived ``<pickup>_AXIS_FOOT`` key.

This module owns that mapping, including the perpendicular-foot geometry, so
every front-end renders the same topology from the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.geometry import extract_array
from kinematics.core.point_ref import PointKey
from kinematics.schema.config import SuspensionConfig
from kinematics.suspensions.base import Suspension

# Suffix for the synthetic perpendicular-foot positions published alongside
# the solver points (e.g. "LEFT_PUSHROD_INBOARD_AXIS_FOOT").
AXIS_FOOT_SUFFIX = "_AXIS_FOOT"

# Display color for the rocker axis and lever arms; matches the fan color
# used by the matplotlib visualization.
ROCKER_DISPLAY_COLOR = "mediumvioletred"


@dataclass(frozen=True)
class DisplayLink:
    """
    One polyline of the display topology, addressed by position-name keys.

    Attributes:
        points: Ordered vertex keys (position-map keys, possibly synthetic).
        color: Display color name.
        label: Human-readable link label (e.g. "Left Rocker Axis").
    """

    points: tuple[str, ...]
    color: str
    label: str


@dataclass(frozen=True)
class RockerDisplayGroup:
    """
    The display description of one rocker-equipped corner.

    Attributes:
        axis_front: Position key of the front rocker-axis point.
        axis_rear: Position key of the rear rocker-axis point.
        pickups: Position keys of every pickup rigidly fixed to the rocker
            (pushrod inboard and, when present, the droplink pickup).
        label_prefix: "Left " / "Right " for an axle corner, "" for a single
            corner.
    """

    axis_front: str
    axis_rear: str
    pickups: tuple[str, ...]
    label_prefix: str


@dataclass(frozen=True)
class WheelDisplayDimensions:
    """
    Static tire dimensions (mm) for drawing a wheel.
    """

    radius: float
    width: float
    rim_radius: float


@dataclass(frozen=True)
class WheelAnchorNames:
    """
    Position keys anchoring one drawn wheel, as position-map string keys.
    """

    center: str
    inboard: str
    outboard: str
    axle_inboard: str
    axle_outboard: str


def rocker_display_groups(suspension: Suspension) -> list[RockerDisplayGroup]:
    """
    Describe each rocker group's axis and rigidly-attached pickup points.

    An axle contributes one group per rocker-equipped corner with side-prefixed
    keys and labels; a single corner contributes at most one un-prefixed group.
    """
    # The axle composes per-side corners; duck-typing on `corners` avoids a
    # circular import of the axle class here.
    corners = getattr(suspension, "corners", None)
    if corners is None:
        rocker_corners = [("", "", suspension)] if suspension.has_rocker else []
    else:
        rocker_corners = [
            (f"{side.name}_", f"{side.name.title()} ", corner)
            for side, corner in corners.items()
            if corner.has_rocker
        ]

    groups: list[RockerDisplayGroup] = []
    for key_prefix, label_prefix, corner in rocker_corners:
        pickups = [f"{key_prefix}{PointID.PUSHROD_INBOARD.name}"]
        if PointID.DROPLINK_ROCKER in corner.hardpoints:
            pickups.append(f"{key_prefix}{PointID.DROPLINK_ROCKER.name}")
        groups.append(
            RockerDisplayGroup(
                axis_front=f"{key_prefix}{PointID.ROCKER_AXIS_FRONT.name}",
                axis_rear=f"{key_prefix}{PointID.ROCKER_AXIS_REAR.name}",
                pickups=tuple(pickups),
                label_prefix=label_prefix,
            )
        )
    return groups


def display_point_keys(suspension: Suspension) -> tuple[PointKey, ...]:
    """
    Point keys a renderer needs, in stable column order.

    The suspension's own output set, extended with any extra points its
    visualization links reference (e.g. the axle's fixed CENTER ARB-axis
    points), so every polyline of the display topology can resolve its
    vertices from the exported positions.
    """
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
    """
    Flatten a state's positions into a name-keyed map with synthetic feet.

    Args:
        positions: The state's point positions (PointKey -> Point3/ndarray).
        point_keys: Which points to export (see display_point_keys).
        rocker_groups: Rocker display groups whose lever-arm feet to append.

    Returns:
        Mapping of position key to (x, y, z), including one
        ``<pickup>_AXIS_FOOT`` entry per rocker pickup.
    """
    named: dict[str, tuple[float, float, float]] = {}
    for key in point_keys:
        position = positions.get(key)
        if position is None:
            continue
        raw = extract_array(position)
        named[key.name] = (float(raw[0]), float(raw[1]), float(raw[2]))

    for group in rocker_groups:
        _append_axis_feet(named, group)
    return named


def _append_axis_feet(
    named_positions: dict[str, tuple[float, float, float]],
    group: RockerDisplayGroup,
) -> None:
    """
    Add each rocker pickup's perpendicular foot on the rocker axis.

    For a pickup p and an axis through point a with direction d, the closest
    point on the axis is:

        foot = a + (dot(p - a, d) / dot(d, d)) * d

    The feet let a renderer draw the true lever arms (perpendicular to the
    axis) rather than lines to the axis endpoints.
    """
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
    """
    The name-keyed link topology for interactive rendering.

    The solver's triangular rocker fan (axis end -> pickups -> axis end) is
    replaced by a rocker-axis line plus one perpendicular lever arm per
    pickup, drawn to the pickup's ``_AXIS_FOOT`` synthetic position.
    """
    links = [
        DisplayLink(
            points=tuple(key.name for key in link.points),
            color=link.color,
            label=link.label,
        )
        for link in suspension.get_visualization_links()
        # The fan is labelled "Rocker" ("Left Rocker"/"Right Rocker" on axles).
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
                if pickup.endswith(PointID.DROPLINK_ROCKER.name)
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
    """
    Static tire dimensions for drawing, or None without a wheel config.
    """
    if config is None:
        return None
    tire = config.wheel.tire
    return WheelDisplayDimensions(
        radius=float(tire.nominal_radius),
        width=float(tire.section_width),
        rim_radius=float(tire.rim_diameter_mm) / 2.0,
    )


def wheel_anchor_names(suspension: Suspension) -> list[WheelAnchorNames]:
    """
    Anchor position keys for each wheel to draw (one per corner for an axle).
    """
    return [
        WheelAnchorNames(
            center=anchors.center.name,
            inboard=anchors.inboard.name,
            outboard=anchors.outboard.name,
            axle_inboard=anchors.axle_inboard.name,
            axle_outboard=anchors.axle_outboard.name,
        )
        for anchors in suspension.wheel_visualization_anchors()
    ]
