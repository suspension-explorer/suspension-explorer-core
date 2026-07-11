"""Constraint helpers for points rigidly attached to moving corner bodies."""

from kinematics.constraints import (
    Constraint,
    DistanceConstraint,
    ScalarTripleProductConstraint,
)
from kinematics.core.constants import MIN_CHIRALITY_VOLUME
from kinematics.core.enums import PointID
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_scalar_triple_product,
)
from kinematics.state import SuspensionState


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
