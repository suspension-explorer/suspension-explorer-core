"""Concrete single-corner suspension models."""

from kinematics.suspensions.corner.double_wishbone import DoubleWishboneSuspension
from kinematics.suspensions.corner.double_wishbone_coilover import (
    DoubleWishboneCoiloverSuspension,
)

__all__ = [
    "DoubleWishboneCoiloverSuspension",
    "DoubleWishboneSuspension",
]
