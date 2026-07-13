"""Double-wishbone corner suspension with an outboard coilover."""

from dataclasses import dataclass
from typing import ClassVar, Sequence

from kinematics.core.constraints import Constraint
from kinematics.core.elements import (
    ElementType,
    SuspensionElement,
    VariableLengthLinkElement,
)
from kinematics.core.metrics.derivatives import (
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.suspensions.corner.attachments import (
    chiral_rigid_point_constraints,
)
from kinematics.core.suspensions.corner.double_wishbone import DoubleWishboneSuspension


@dataclass
class DoubleWishboneCoiloverSuspension(DoubleWishboneSuspension):
    """A double-wishbone corner with a lower-wishbone-mounted coilover."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_coilover"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = (
        DoubleWishboneSuspension.REQUIRED_POINTS
        | {PointID.STRUT_TOP, PointID.STRUT_BOTTOM}
    )
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        *DoubleWishboneSuspension.OUTPUT_POINTS,
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
    )

    @property
    def has_strut(self) -> bool:
        """This topology always includes a coilover."""
        return True

    def free_points(self) -> Sequence[PointID]:
        """Base corner variables plus the moving coilover pickup."""
        return (*super().free_points(), PointID.STRUT_BOTTOM)

    def constraints(self) -> list[Constraint]:
        """Base corner constraints plus the lower-wishbone attachment."""
        constraints = super().constraints()
        constraints.extend(
            chiral_rigid_point_constraints(
                self.initial_state(),
                PointID.STRUT_BOTTOM,
                (
                    PointID.LOWER_WISHBONE_INBOARD_FRONT,
                    PointID.LOWER_WISHBONE_INBOARD_REAR,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                ),
            )
        )
        return constraints

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare damper length relative to hub vertical travel."""
        return (
            DerivativeMetricDefinition(
                response=PointDistanceResponse(
                    PointID.STRUT_TOP,
                    PointID.STRUT_BOTTOM,
                    name="damper_length",
                    unit=MetricUnit.MM,
                ),
                driver=PointCoordinateResponse.from_world_axis(
                    PointID.WHEEL_CENTER,
                    Axis.Z,
                    name="hub_z",
                    unit=MetricUnit.MM,
                ),
            ),
        )

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return base corner elements plus the coilover."""
        return (
            *super().elements(),
            VariableLengthLinkElement(
                label="Spring/Damper",
                type=ElementType.SPRING_DAMPER,
                point_a=PointID.STRUT_TOP,
                point_b=PointID.STRUT_BOTTOM,
            ),
        )
