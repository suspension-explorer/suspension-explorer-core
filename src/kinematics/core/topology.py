"""Renderer-neutral topology for suspension consumers."""

from dataclasses import dataclass
from enum import StrEnum

from kinematics.core.primitives.point_ref import PointKey


class LinkRole(StrEnum):
    """Physical role of a link or point in the suspension topology."""

    WISHBONE = "wishbone"
    UPRIGHT = "upright"
    TRACK_ROD = "track_rod"
    TRACK_ROD_COUPLING = "track_rod_coupling"
    AXLE = "axle"
    CONTACT_PATCH = "contact_patch"
    PUSHROD = "pushrod"
    ROCKER = "rocker"
    SPRING_DAMPER = "spring_damper"
    ANTI_ROLL_BAR = "anti_roll_bar"
    DROPLINK = "droplink"


@dataclass(frozen=True)
class LinkTopology:
    """One physical polyline or point in a suspension topology."""

    points: tuple[PointKey, ...]
    role: LinkRole
    label: str


@dataclass(frozen=True)
class RockerTopology:
    """A rocker axis and its rigid pickup points."""

    axis_front: PointKey
    axis_rear: PointKey
    pickups: tuple[PointKey, ...]
    label_prefix: str = ""


@dataclass(frozen=True)
class WheelTopology:
    """Point keys anchoring one wheel."""

    center: PointKey
    inboard: PointKey
    outboard: PointKey
    axle_inboard: PointKey
    axle_outboard: PointKey


@dataclass(frozen=True)
class SuspensionTopology:
    """Complete renderer-neutral topology for one suspension model."""

    output_points: tuple[PointKey, ...]
    links: tuple[LinkTopology, ...]
    rockers: tuple[RockerTopology, ...]
    wheels: tuple[WheelTopology, ...]
