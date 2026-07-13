"""Serialized selectors for suspension architectures and mechanisms."""

from enum import StrEnum


class SuspensionType(StrEnum):
    """Supported suspension architecture carriers."""

    DOUBLE_WISHBONE = "double_wishbone"
    DOUBLE_WISHBONE_AXLE = "double_wishbone_axle"


class ActuationType(StrEnum):
    """Supported corner actuation mechanisms."""

    DIRECT = "direct"
    PUSHROD_ROCKER = "pushrod_rocker"


class CornerSpringType(StrEnum):
    """Supported corner spring mechanisms."""

    NONE = "none"
    COILOVER = "coilover"
    TORSION_BAR = "torsion_bar"


class ArbType(StrEnum):
    """Supported axle anti-roll mechanisms."""

    NONE = "none"
    U_BAR = "u_bar"


class HeaveLinkType(StrEnum):
    """Supported axle heave-link layouts."""

    NONE = "none"
    ROCKER_TO_ROCKER = "rocker_to_rocker"
