"""Constraint helpers for points rigidly attached to moving corner bodies."""

from collections.abc import Mapping
from typing import cast

from kinematics.core.constraints import (
    Constraint,
    DistanceConstraint,
    ScalarTripleProductConstraint,
)
from kinematics.core.enums import PointID
from kinematics.core.primitives.constants import EPS_GEOMETRIC, MIN_CHIRALITY_VOLUME
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    compute_scalar_triple_product,
)
from kinematics.core.state import SuspensionState


def rigid_point_constraints(
    initial_state: SuspensionState,
    point: PointID,
    references: tuple[PointID, PointID, PointID],
) -> list[Constraint]:
    """
    Hold a point rigidly to a body defined by three reference points.

    Three design-length distance constraints locate the point relative to the
    body. These distances do not distinguish reflected assembly branches.
    """
    positions = initial_state.positions
    return [
        DistanceConstraint(
            point,
            reference,
            compute_point_point_distance(positions[point], positions[reference]),
        )
        for reference in references
    ]


def chiral_rigid_point_constraints(
    initial_state: SuspensionState,
    point: PointID,
    references: tuple[PointID, PointID, PointID],
) -> list[Constraint]:
    """Hold a rigid pickup to three references and preserve authored handedness."""
    constraints = rigid_point_constraints(initial_state, point, references)
    positions = initial_state.positions
    reference_a, reference_b, reference_c = references
    authored_volume = compute_scalar_triple_product(
        positions[reference_b] - positions[reference_a],
        positions[reference_c] - positions[reference_a],
        positions[point] - positions[reference_a],
    )
    if abs(authored_volume) < MIN_CHIRALITY_VOLUME:
        raise ValueError(
            f"{point.name} and its rigid-body references do not define reliable "
            "handedness"
        )
    constraints.append(
        ScalarTripleProductConstraint(
            reference_a,
            reference_b,
            reference_c,
            point,
            target_volume=authored_volume,
            scale=abs(authored_volume),
        )
    )
    return constraints


def validate_rigid_anchor_points(
    hardpoints: Mapping[PointKey, Point3],
    anchors: tuple[PointID, ...],
    label: str,
) -> None:
    """Validate that the anchors can rigidly fix a moving pickup to a body."""
    # Three non-collinear anchors are the minimum to fix a point rigidly
    # to a body; collinear anchors leave rotation about their line free.
    if len(anchors) < 3:
        raise ValueError(f"{label} requires at least three mounting body anchors")
    anchor_a, anchor_b, anchor_c = (hardpoints[point] for point in anchors[:3])
    if compute_point_point_distance(anchor_a, anchor_b) <= EPS_GEOMETRIC:
        raise ValueError(f"{label} mounting body anchors must be distinct")
    anchor_line = (anchor_b - anchor_a).normalize()
    if compute_point_to_line_distance(anchor_c, anchor_a, anchor_line) <= EPS_GEOMETRIC:
        raise ValueError(
            f"The first three {label} mounting body anchors must not be collinear"
        )


def anchored_rigid_point_constraints(
    initial_state: SuspensionState,
    point: PointID,
    anchors: tuple[PointID, ...],
) -> list[Constraint]:
    """Hold a pickup rigidly to a body defined by three or more anchors."""
    # The first three anchors hold the pickup rigidly with authored handedness;
    # further anchors add plain redundant distances.
    primary_anchors = cast(
        "tuple[PointID, PointID, PointID]",
        anchors[:3],
    )
    positions = initial_state.positions
    constraints: list[Constraint] = list(
        chiral_rigid_point_constraints(initial_state, point, primary_anchors)
    )
    constraints.extend(
        DistanceConstraint(
            point,
            anchor,
            compute_point_point_distance(positions[point], positions[anchor]),
        )
        for anchor in anchors[3:]
    )
    return constraints
