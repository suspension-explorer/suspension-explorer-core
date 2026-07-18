"""Cross-cutting enumeration types for suspension kinematics."""

from enum import IntEnum, StrEnum


class Axis(IntEnum):
    """Principal axes in three-dimensional space."""

    X = 0
    Y = 1
    Z = 2


class TargetPositionMode(StrEnum):
    """Interpretation of a target position value."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class Units(StrEnum):
    """Units used by geometric inputs and outputs."""

    MILLIMETERS = "millimeters"
    DEGREES = "degrees"

    @property
    def symbol(self) -> str:
        """Return the abbreviated unit symbol."""
        return {Units.MILLIMETERS: "mm", Units.DEGREES: "deg"}[self]


class PointID(IntEnum):
    """Identifiers for authored and derived suspension points."""

    NOT_ASSIGNED = 0

    LOWER_WISHBONE_INBOARD_FRONT = 1
    LOWER_WISHBONE_INBOARD_REAR = 2
    LOWER_WISHBONE_OUTBOARD = 3

    UPPER_WISHBONE_INBOARD_FRONT = 4
    UPPER_WISHBONE_INBOARD_REAR = 5
    UPPER_WISHBONE_OUTBOARD = 6

    PUSHROD_INBOARD = 7
    PUSHROD_OUTBOARD = 8

    TRACKROD_INBOARD = 9
    TRACKROD_OUTBOARD = 10
    TOE_LINK_INBOARD = 11
    TOE_LINK_OUTBOARD = 12

    AXLE_INBOARD = 13
    AXLE_OUTBOARD = 14
    AXLE_MIDPOINT = 15

    STRUT_TOP = 16
    STRUT_BOTTOM = 17

    WHEEL_CENTER = 18
    WHEEL_INBOARD = 19
    WHEEL_OUTBOARD = 20

    CONTACT_PATCH_CENTER = 21

    # Outboard camber shim geometry. Datum points A and B lie on the design
    # mid-thickness plane; the face normal is perpendicular to that plane.
    CAMBER_SHIM_FACE_POINT_A = 22
    CAMBER_SHIM_FACE_POINT_B = 23
    CAMBER_SHIM_FACE_NORMAL = 24

    ROCKER_AXIS_A = 25
    ROCKER_AXIS_B = 26
    DROPLINK_ROCKER = 27
    DROPLINK_U_BAR = 28
    ARB_U_BAR_AXIS_A = 29
    ARB_U_BAR_AXIS_B = 30
    HEAVE_LINK_ROCKER = 31
    ARB_T_BAR_PIVOT = 32
    DROPLINK_T_BAR = 33


class ShimType(StrEnum):
    """Supported suspension shim adjustments."""

    OUTBOARD_CAMBER = "outboard_camber"


class SuspensionType(StrEnum):
    """Supported suspension architecture carriers."""

    DOUBLE_WISHBONE = "double_wishbone"
    MACPHERSON = "macpherson"


class Scope(StrEnum):
    """Whether a model or metric covers one corner or a composed axle."""

    CORNER = "corner"
    AXLE = "axle"


class AxlePosition(StrEnum):
    """Which end of the vehicle an axle or corner belongs to."""

    FRONT = "front"
    REAR = "rear"


class ActuationType(StrEnum):
    """Supported corner actuation mechanisms."""

    DIRECT = "direct"
    PUSHROD_ROCKER = "pushrod_rocker"


class MountBody(StrEnum):
    """Rigid corner bodies that a moving mechanism pickup can be fixed to."""

    LOWER_WISHBONE = "lower_wishbone"
    UPRIGHT = "upright"


class CornerSpringType(StrEnum):
    """Supported corner spring mechanisms."""

    NONE = "none"
    COILOVER = "coilover"
    TORSION_BAR = "torsion_bar"


class ArbType(StrEnum):
    """Supported axle anti-roll mechanisms."""

    NONE = "none"
    U_BAR = "u_bar"
    T_BAR = "t_bar"


class HeaveLinkType(StrEnum):
    """Supported axle heave-link layouts."""

    NONE = "none"
    ROCKER_TO_ROCKER = "rocker_to_rocker"


class SteeringType(StrEnum):
    """Supported axle steering actuators."""

    NONE = "none"
    RACK = "rack"
