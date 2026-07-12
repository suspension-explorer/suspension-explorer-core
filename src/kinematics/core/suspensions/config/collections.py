"""
Common suspension component point collections.

This module defines dataclasses for point collections which are shared across suspension
architectures.
"""

from dataclasses import dataclass


@dataclass
class LowerWishbonePoints:
    """
    Points defining the lower wishbone geometry.

    Attributes:
        inboard_front: Front inboard mounting point coordinates.
        inboard_rear: Rear inboard mounting point coordinates.
        outboard: Outboard mounting point coordinates.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class UpperWishbonePoints:
    """
    Points defining the upper wishbone geometry.

    Attributes:
        inboard_front: Front inboard mounting point coordinates.
        inboard_rear: Rear inboard mounting point coordinates.
        outboard: Outboard mounting point coordinates.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class WheelAxlePoints:
    """
    Points defining the wheel axle geometry.

    Attributes:
        inner: Inner axle point coordinates.
        outer: Outer axle point coordinates.
    """

    inner: dict[str, float]
    outer: dict[str, float]


@dataclass
class TrackRodPoints:
    """
    Points defining the track rod/tie rod geometry.

    Attributes:
        inner: Inner track rod mounting point coordinates.
        outer: Outer track rod mounting point coordinates.
    """

    inner: dict[str, float]
    outer: dict[str, float]


# Upright component definitions.


@dataclass
class UprightMounts:
    """
    References to hardpoints that define the upright's kinematic frame.

    These are string names that map to PointID enum values. The upright does not
    own these points - it references them from the central hardpoints registry.

    Attributes:
        upper_ball_joint: PointID name for upper wishbone connection.
        lower_ball_joint: PointID name for lower wishbone connection.
        steering_pickup: PointID name for track rod connection.
    """

    upper_ball_joint: str
    lower_ball_joint: str
    steering_pickup: str


@dataclass
class UprightAttachments:
    """
    Attachments owned by the upright that move rigidly with it.

    These are defined as global coordinates at design position. They will be
    converted to local offsets when the upright is constructed, and transformed
    back to global coordinates based on the current hardpoint state.

    Attributes:
        axle_inboard: Inboard axle point {x, y, z} (spindle/bearing side)
        axle_outboard: Outboard axle point {x, y, z} (hub face/wheel side)
    """

    axle_inboard: dict[str, float]
    axle_outboard: dict[str, float]


@dataclass
class UprightComponentConfig:
    """
    Complete upright component configuration combining mounts and attachments.

    Attributes:
        mounts: References to hardpoints defining the kinematic frame
        attachments: Points owned by the upright that move with it
    """

    mounts: UprightMounts
    attachments: UprightAttachments


@dataclass
class ComponentsConfig:
    """
    Configuration for all rigid body components in the suspension.

    Attributes:
        upright: Upright component configuration
    """

    upright: UprightComponentConfig
