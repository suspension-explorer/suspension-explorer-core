"""Rack-driven track-rod component for steered corner suspensions."""

from dataclasses import dataclass
from typing import ClassVar

from kinematics.core.constraints import (
    Constraint,
    DistanceConstraint,
    point_on_line_constraints,
)
from kinematics.core.coordinates import ChassisAxisSystem
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
class TrackRod:
    """Steer the wheel from a rack-driven inboard pickup."""

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD}
    )
    OUTPUT_POINTS: ClassVar[tuple[PointID, PointID]] = (
        PointID.TRACKROD_INBOARD,
        PointID.TRACKROD_OUTBOARD,
    )

    upright_anchors: tuple[PointID, ...]
    preserve_attachment_handedness: bool = True
    length_adjustment: float = 0.0

    @property
    def inboard_point(self) -> PointID:
        """Return the rack pickup."""
        return PointID.TRACKROD_INBOARD

    @property
    def outboard_point(self) -> PointID:
        """Return the upright pickup."""
        return PointID.TRACKROD_OUTBOARD

    def validate(self, hardpoints: dict[PointKey, Point3]) -> None:
        """Validate that the outboard pickup can be fixed to the upright."""
        validate_rigid_anchor_points(hardpoints, self.upright_anchors, "Track rod")

    @property
    def free_points(self) -> tuple[PointID, ...]:
        """Return the moving rack and upright pickups."""
        return (PointID.TRACKROD_OUTBOARD, PointID.TRACKROD_INBOARD)

    def constraints(self, initial_state: SuspensionState) -> list[Constraint]:
        """Hold shim-adjusted length, attach upright, and guide rack motion."""
        positions = initial_state.positions
        if self.preserve_attachment_handedness:
            attachment_constraints = anchored_rigid_point_constraints(
                initial_state,
                PointID.TRACKROD_OUTBOARD,
                self.upright_anchors,
            )
        else:
            attachment_constraints = [
                DistanceConstraint(
                    PointID.TRACKROD_OUTBOARD,
                    anchor,
                    compute_point_point_distance(
                        positions[PointID.TRACKROD_OUTBOARD],
                        positions[anchor],
                    ),
                )
                for anchor in self.upright_anchors
            ]

        design_length = compute_point_point_distance(
            positions[PointID.TRACKROD_INBOARD],
            positions[PointID.TRACKROD_OUTBOARD],
        )
        setup_length = design_length + self.length_adjustment
        if setup_length <= EPS_GEOMETRIC:
            raise ValueError(
                "Toe shim produces a non-positive setup track-rod length: "
                f"{setup_length:g} mm"
            )

        return [
            DistanceConstraint(
                PointID.TRACKROD_INBOARD,
                PointID.TRACKROD_OUTBOARD,
                setup_length,
            ),
            *attachment_constraints,
            *point_on_line_constraints(
                point_id=PointID.TRACKROD_INBOARD,
                line_point=positions[PointID.TRACKROD_INBOARD],
                line_direction=ChassisAxisSystem.Y,
            ),
        ]

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the physical track-rod element."""
        return (
            RigidLinkElement(
                label="Track Rod",
                type=ElementType.TRACK_ROD,
                point_a=PointID.TRACKROD_INBOARD,
                point_b=PointID.TRACKROD_OUTBOARD,
            ),
        )
