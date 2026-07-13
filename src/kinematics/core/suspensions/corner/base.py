"""Abstract base for single-corner suspension architectures."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from kinematics.core.enums import PointID
from kinematics.core.metrics.main import compute_metrics_for_state
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.sensitivity import TangentField


@dataclass
class CornerSuspension(Suspension):
    """
    One vehicle corner.

    Owns the point-role vocabulary that shared metrics consume: which points
    define the wheel spin axis, the steering axis, and the rack attachment.
    Roles name PointID values resolved through the solved state, so a role may
    refer to a derived point (for example a virtual steering-axis pivot on a
    multilink corner). The steering-axis pivots need not be free points: a
    MacPherson corner returns its fixed strut top as the upper pivot.
    """

    @abstractmethod
    def free_points(self) -> Sequence[PointID]:
        """
        Corner free points are always bare PointID values.

        The axle composer relies on this to side-qualify them as PointRef
        keys without ambiguity.
        """
        ...

    def wheel_axis_points(self) -> tuple[PointID, PointID]:
        """
        Wheel spin axis as (inboard, outboard).

        The inboard-to-outboard direction convention is load-bearing for
        camber and toe signs. Every supported corner names its spin axis with
        these points; override only for an architecture that does not.
        """
        return (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)

    @abstractmethod
    def steering_axis_points(self) -> tuple[PointID, PointID]:
        """
        Steering (kingpin) axis pivots as (lower, upper).

        The lower-to-upper direction convention is load-bearing for caster
        and KPI signs.
        """
        ...

    @abstractmethod
    def rack_attachment_point(self) -> PointID | None:
        """
        Point that translates with the steering rack, or None for an
        unsteered corner.

        Its offset from the design position along the rack axis is the
        exported rack displacement.
        """
        ...

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "MetricRow":
        """Compute one corner metric row, including derivatives when tangents exist."""
        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_state(state, self, self.config, tangents)
