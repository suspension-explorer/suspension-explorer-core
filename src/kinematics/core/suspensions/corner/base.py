"""Abstract base for single-corner suspension architectures."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.core.coordinates import PhysicalCoordinate
from kinematics.core.enums import (
    ActuatorPositionCoordinateID,
    Axis,
    PointID,
    Scope,
    SuspensionType,
)
from kinematics.core.metrics.main import compute_metrics_for_state
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.targeting import PointTargetAxis, TargetKind

if TYPE_CHECKING:
    from kinematics.core.coordinates import ArmAngleCoordinate
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.rigid_motion import UprightScrewAxisResult
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

    TYPE_KEY: ClassVar[SuspensionType]

    def reported_type_key(self) -> SuspensionType:
        """Return the corner architecture identity."""
        return self.TYPE_KEY

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
    def steering_axis_points(self) -> tuple[PointID, PointID] | None:
        """
        Physical steering (kingpin) axis pivots as (lower, upper).

        The lower-to-upper direction convention is load-bearing for caster
        and KPI signs. Return None for an architecture whose steering axis is
        purely motion-derived (e.g. a multi-link with separated ball joints);
        the physical steering metric family is then omitted and only the
        virtual screw-axis family reports.
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

    def required_actuator_coordinates(self) -> tuple[PhysicalCoordinate, ...]:
        """Require the rack translation coordinate for a steered corner."""
        steering = self.steering_actuator_coordinate()
        return (steering,) if steering is not None else ()

    def steering_actuator_coordinate(self) -> PhysicalCoordinate | None:
        """Return the rack translation coordinate for a steered corner."""
        return next(
            (
                coordinate
                for coordinate in self.drive_coordinates()
                if coordinate.kind is TargetKind.ACTUATOR_POSITION
                and coordinate.id == ActuatorPositionCoordinateID.RACK
            ),
            None,
        )

    def drive_coordinates(self) -> tuple[PhysicalCoordinate, ...]:
        """Expose a named rack position plus installed element coordinates."""
        coordinates = super().drive_coordinates()
        rack_point = self.rack_attachment_point()
        if rack_point is None:
            return coordinates
        return (
            PhysicalCoordinate(
                id=ActuatorPositionCoordinateID.RACK,
                kind=TargetKind.ACTUATOR_POSITION,
                label=ActuatorPositionCoordinateID.RACK.label,
                unit=ActuatorPositionCoordinateID.RACK.unit,
                point_keys=(rack_point,),
                scope=Scope.CORNER,
                direction=PointTargetAxis(Axis.Y),
            ),
            *coordinates,
        )

    def _installed_damper_coordinate(self) -> PhysicalCoordinate | None:
        """Return the installed true damper/strut length coordinate, if any."""
        from kinematics.core.enums import ElementLengthCoordinateID

        return next(
            (
                coordinate
                for coordinate in self.drive_coordinates()
                if coordinate.kind is TargetKind.ELEMENT_LENGTH
                and coordinate.id == ElementLengthCoordinateID.DAMPER
            ),
            None,
        )

    def _arm_angle_coordinate(
        self,
        *,
        coordinate_id: str,
        label: str,
        hinge_point_a: PointID,
        hinge_point_b: PointID,
        carried_point: PointID,
    ) -> "ArmAngleCoordinate":
        """Build one fixed-axis signed arm angle from the design state."""
        from kinematics.core.coordinates import ArmAngleCoordinate

        return ArmAngleCoordinate.from_positions(
            id=coordinate_id,
            label=label,
            hinge_point_a=hinge_point_a,
            hinge_point_b=hinge_point_b,
            carried_point=carried_point,
            positions=self.initial_state().positions,
            scope=Scope.CORNER,
            side=self.side,
        )

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
        steering_response_axes: "Sequence[UprightScrewAxisResult] | None" = None,
    ) -> "MetricRow":
        """Compute one corner metric row, including derivatives when tangents exist."""
        if self.config is None:
            raise ValueError("Suspension has no configuration")
        axis_results = tuple(steering_response_axes or ())
        return compute_metrics_for_state(
            state,
            self,
            self.config,
            tangents,
            steering_response_axis=(
                axis_results[0] if len(axis_results) == 1 else None
            ),
        )
