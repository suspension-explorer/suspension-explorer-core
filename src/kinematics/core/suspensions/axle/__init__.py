"""Full-axle suspension models composed from corner models."""

from kinematics.core.suspensions.axle.double_wishbone import (
    DoubleWishboneAxleSuspension,
)
from kinematics.core.suspensions.axle.mechanisms import (
    ArbNone,
    ArbTBar,
    ArbUBar,
    AxleArb,
    AxleHeaveLink,
    HeaveLinkNone,
    HeaveLinkRockerToRocker,
)
