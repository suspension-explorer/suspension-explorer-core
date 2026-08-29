"""Fixed toe-link component for non-steered corner suspensions."""

from dataclasses import dataclass
from typing import ClassVar

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import ElementType, RigidLinkElement, SuspensionElement
from kinematics.core.enums import PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    anchored_rigid_point_constraints,
    validate_rigid_anchor_points,
)


@dataclass(frozen=True)
class ToeLink:
    """Locate wheel heading from a fixed chassis pickup."""

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {PointID.TOE_LINK_INBOARD, PointID.TOE_LINK_OUTBOARD}
    )
    OUTPUT_POINTS: ClassVar[tuple[PointID, PointID]] = (
        PointID.TOE_LINK_INBOARD,
        PointID.TOE_LINK_OUTBOARD,
    )

    upright_anchors: tuple[PointID, ...]
    preserve_attachment_handedness: bool = True
    length_adjustment: float = 0.0

    @property
    def inboard_point(self) -> PointID:
        """Return the fixed chassis pickup."""
        return PointID.TOE_LINK_INBOARD

    @property
    def outboard_point(self) -> PointID:
        """Return the upright pickup."""
        return PointID.TOE_LINK_OUTBOARD

    def validate(self, hardpoints: dict[PointKey, Point3]) -> None:
        """Validate that the outboard pickup can be fixed to the upright."""
        validate_rigid_anchor_points(hardpoints, self.upright_anchors, "Toe link")

    @property
    def free_points(self) -> tuple[PointID, ...]:
        """Return the moving outboard pickup."""
        return (PointID.TOE_LINK_OUTBOARD,)

    def constraints(self, initial_state: SuspensionState) -> list[Constraint]:
        """Hold shim-adjusted link length and attach it to the upright."""
        positions = initial_state.positions
        if self.preserve_attachment_handedness:
            attachment_constraints = anchored_rigid_point_constraints(
                initial_state,
                PointID.TOE_LINK_OUTBOARD,
                self.upright_anchors,
            )
        else:
            attachment_constraints = [
                DistanceConstraint(
                    PointID.TOE_LINK_OUTBOARD,
                    anchor,
                    compute_point_point_distance(
                        positions[PointID.TOE_LINK_OUTBOARD],
                        positions[anchor],
                    ),
                )
                for anchor in self.upright_anchors
            ]

        design_length = compute_point_point_distance(
            positions[PointID.TOE_LINK_INBOARD],
            positions[PointID.TOE_LINK_OUTBOARD],
        )
        setup_length = design_length + self.length_adjustment
        if setup_length <= EPS_GEOMETRIC:
            raise ValueError(
                "Toe shim produces a non-positive setup toe-link length: "
                f"{setup_length:g} mm"
            )

        return [
            DistanceConstraint(
                PointID.TOE_LINK_INBOARD,
                PointID.TOE_LINK_OUTBOARD,
                setup_length,
            ),
            *attachment_constraints,
        ]

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the physical toe-link element."""
        return (
            RigidLinkElement(
                label="Toe Link",
                type=ElementType.TOE_LINK,
                point_a=PointID.TOE_LINK_INBOARD,
                point_b=PointID.TOE_LINK_OUTBOARD,
            ),
        )
