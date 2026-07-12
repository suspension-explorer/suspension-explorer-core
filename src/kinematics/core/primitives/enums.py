"""
Enumeration types for suspension kinematics.
"""

from enum import Enum, IntEnum


class PointID(IntEnum):
    """
    Enumeration of all point identifiers used in the suspension system.
    """

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

    AXLE_INBOARD = 11
    AXLE_OUTBOARD = 12
    AXLE_MIDPOINT = 13

    STRUT_TOP = 14
    STRUT_BOTTOM = 15

    WHEEL_CENTER = 16
    WHEEL_INBOARD = 17
    WHEEL_OUTBOARD = 18

    CONTACT_PATCH_CENTER = 19

    # Outboard camber shim geometry. Datum points A and B lie on the design
    # mid-thickness plane; the face normal is perpendicular to that plane.
    CAMBER_SHIM_FACE_POINT_A = 21
    CAMBER_SHIM_FACE_POINT_B = 22
    CAMBER_SHIM_FACE_NORMAL = 23

    ROCKER_AXIS_FRONT = 24
    ROCKER_AXIS_REAR = 25
    DROPLINK_ROCKER = 26
    ARB_AXIS_A = 27
    ARB_AXIS_B = 28
    DROPLINK_ARB = 29


class Axis(IntEnum):
    """
    Enumeration of the three principal axes in 3D space.
    """

    X = 0
    Y = 1
    Z = 2


class TargetPositionMode(Enum):
    """
    Specifies how a target value should be interpreted.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class Units(Enum):
    """
    Units of measurement for geometric parameters.
    """

    MILLIMETERS = "millimeters"
    DEGREES = "degrees"

    @property
    def symbol(self) -> str:
        """
        Short display symbol for the unit, e.g. "mm".
        """
        return {Units.MILLIMETERS: "mm", Units.DEGREES: "deg"}[self]


class ShimType(Enum):
    """
    Types of shim adjustments supported by suspension templates.
    """

    OUTBOARD_CAMBER = "outboard_camber"
