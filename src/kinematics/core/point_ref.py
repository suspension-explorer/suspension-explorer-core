"""
Side-qualified point references for multi-corner suspension models.

The single-corner machinery keys every position, constraint, and derived point on
the global :class:`~kinematics.core.enums.PointID`. To model a full axle (two
corners solved together), the same points must exist twice -- once per side --
without colliding. :class:`PointRef` pairs a :class:`Side` with a ``PointID`` to
give each corner its own namespace while keeping the runtime machinery
key-agnostic.

The :data:`PointKey` alias (``PointID | PointRef``) is the generalised key type:
single-corner code keeps building plain ``PointID`` keyed dicts, while axle code
builds ``PointRef`` keyed dicts. Nothing in the solver, state, or constraint
runtime depends on which concrete key type is used -- only on hashability,
ordering, and a ``.name`` for output columns.
"""

from enum import IntEnum
from typing import NamedTuple

from kinematics.core.enums import PointID


class Side(IntEnum):
    """
    Which corner of the axle a point belongs to.

    Handedness follows the repo-wide ISO 8855 convention (X forward, Y left,
    Z up, right-handed):

    - ``LEFT`` is the +Y side.
    - ``RIGHT`` is the -Y side.
    - ``CENTER`` is for chassis elements shared between the two corners
      (e.g. the steering rack, the anti-roll-bar axis) that are not mirrored.

    Values are ordered ``LEFT < RIGHT < CENTER`` so that :class:`PointRef`
    tuples sort deterministically by side first.
    """

    LEFT = 0
    RIGHT = 1
    CENTER = 2


class PointRef(NamedTuple):
    """
    A side-qualified point reference: ``(side, point)``.

    Being a :class:`~typing.NamedTuple`, a ``PointRef`` is hashable and compares
    and sorts as the plain tuple ``(side, point)`` -- i.e. by side first, then by
    ``PointID``. This gives deterministic ordering when
    :class:`~kinematics.state.SuspensionState` sorts its free points.

    Note:
        A positions dict must use homogeneous key types -- either all
        ``PointID`` or all ``PointRef`` -- because Python cannot sort a mix of
        the two (``PointID`` and ``PointRef`` are not mutually orderable), and
        the solver relies on sorting the free-point keys for a stable variable
        ordering. Single-corner models use ``PointID``; axle models use
        ``PointRef`` throughout.
    """

    side: Side
    point: PointID

    @property
    def name(self) -> str:
        """
        Column-friendly name, e.g. ``"LEFT_LOWER_WISHBONE_OUTBOARD"``.

        Matches how single-corner output columns are built from ``pid.name``,
        with the side prefixed so left/right columns do not collide.
        """
        return f"{self.side.name}_{self.point.name}"


# Generalised point-key type. Single-corner code uses PointID; axle code uses
# PointRef. All core machinery (state, constraints, solver, derived points) is
# annotated over this alias so it works with either concrete key type.
PointKey = PointID | PointRef
