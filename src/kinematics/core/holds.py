"""Generic fixed-coordinate declarations.

A hold names scalar coordinates but deliberately stores no numerical values.
Materialising it against a reference state produces absolute targets equal to
the coordinates' values in that state.  The same operation supports two
different callers:

* a nonlinear sweep materialises once at its reference state and enforces the
  resulting values at every step; and
* a local response materialises independently at each solved state, where the
  resulting target rows are held at zero rate.

Those callers remain separate because they answer different questions.  This
module unifies only the meaning of *what is held* and the mechanics used to
capture it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kinematics.core.coordinates import CoordinateTarget, ScalarCoordinate
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey


@dataclass(frozen=True)
class CoordinateHold:
    """An ordered, value-free set of scalar coordinates to hold constant."""

    coordinates: tuple[ScalarCoordinate, ...] = ()

    def __post_init__(self) -> None:
        identities = tuple(
            coordinate.coordinate_identity for coordinate in self.coordinates
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Coordinate hold has duplicate held coordinates")

    def materialize(
        self,
        positions: Mapping[PointKey, Point3],
    ) -> tuple[CoordinateTarget, ...]:
        """Capture every coordinate as an absolute target at ``positions``."""
        return tuple(
            coordinate.current_value_target(positions)
            for coordinate in self.coordinates
        )

    def __bool__(self) -> bool:
        return bool(self.coordinates)
