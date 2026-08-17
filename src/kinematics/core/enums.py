"""Cross-cutting enumeration types for suspension kinematics."""

from enum import IntEnum, StrEnum


class Axis(IntEnum):
    """Principal axes in three-dimensional space."""

    X = 0
    Y = 1
    Z = 2


class TargetValueMode(StrEnum):
    """Interpretation of an authored scalar target value."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class CoordinateSidePolicy(StrEnum):
    """Static side ownership of a globally known sweep coordinate."""

    CORNER = "corner"
    SHARED = "shared"


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

    # Geometric support point where the wheel plane is tangent to the axle's
    # shared, zero-grade ground plane.
    WHEEL_CONTACT_CENTRE = 21

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

    # Unsteered semi-trailing-arm locating geometry. A/B are the fixed chassis
    # mounts defining the oblique arm pivot. Torsion-bar springing uses a
    # separate transverse axis through pivot A.
    TRAILING_ARM_PIVOT_A = 34
    TRAILING_ARM_PIVOT_B = 35
    TRAILING_ARM_OUTBOARD = 36
    TORSION_BAR_AXIS_A = 37
    TORSION_BAR_AXIS_B = 38

    # Independent linear damper endpoints. The chassis pickup is fixed while
    # the rocker pickup is carried by a pushrod/rocker actuation mechanism.
    DAMPER_CHASSIS = 39
    DAMPER_ROCKER = 40

    @property
    def sweep_side_policy(self) -> CoordinateSidePolicy:
        """Return static side ownership for this point coordinate."""
        if self in (
            PointID.ARB_U_BAR_AXIS_A,
            PointID.ARB_U_BAR_AXIS_B,
            PointID.ARB_T_BAR_PIVOT,
        ):
            return CoordinateSidePolicy.SHARED
        return CoordinateSidePolicy.CORNER

    @property
    def output_only_target_guidance(self) -> str | None:
        """Return point-specific guidance when an output cannot be driven."""
        if self is PointID.WHEEL_CONTACT_CENTRE:
            return (
                "Target 'wheel_center' along Z as the available heave input; "
                "wheel orientation can still move the wheel contact centre, so read "
                "ride height from the 'ride_height_change' metric of the solved "
                "output."
            )
        return None


class ShimType(StrEnum):
    """Supported suspension shim adjustments."""

    OUTBOARD_CAMBER = "outboard_camber"


class SuspensionType(StrEnum):
    """Supported suspension architecture carriers."""

    DOUBLE_WISHBONE = "double_wishbone"
    MACPHERSON = "macpherson"
    TRAILING_ARM = "trailing_arm"


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


class CornerDamperType(StrEnum):
    """Supported independent corner damper mechanisms."""

    NONE = "none"
    LINEAR = "linear"


class ActuatorPositionCoordinateID(StrEnum):
    """Stable wire IDs for globally known actuator-position coordinates."""

    RACK = "rack"

    @property
    def label(self) -> str:
        """Return the stable user-facing coordinate label."""
        return "Rack Position"

    @property
    def unit(self) -> str:
        """Return the stable scalar unit symbol."""
        return Units.MILLIMETERS.symbol

    @property
    def side_policy(self) -> CoordinateSidePolicy:
        """Return static side ownership for this coordinate."""
        return CoordinateSidePolicy.SHARED


class ElementLengthCoordinateID(StrEnum):
    """Stable wire IDs for globally known element-length coordinates."""

    DAMPER = "damper"
    HEAVE_LINK = "heave_link"

    @property
    def label(self) -> str:
        """Return the stable user-facing coordinate label."""
        return {
            ElementLengthCoordinateID.DAMPER: "Damper Length",
            ElementLengthCoordinateID.HEAVE_LINK: "Heave Link Length",
        }[self]

    @property
    def unit(self) -> str:
        """Return the stable scalar unit symbol."""
        return Units.MILLIMETERS.symbol

    @property
    def side_policy(self) -> CoordinateSidePolicy:
        """Return static side ownership for this coordinate."""
        return {
            ElementLengthCoordinateID.DAMPER: CoordinateSidePolicy.CORNER,
            ElementLengthCoordinateID.HEAVE_LINK: CoordinateSidePolicy.SHARED,
        }[self]


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
