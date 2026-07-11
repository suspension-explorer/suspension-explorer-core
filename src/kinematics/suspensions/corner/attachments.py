"""Constraint helpers for points rigidly attached to moving corner bodies."""

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core.enums import PointID
from kinematics.core.vector_utils.geometric import compute_point_point_distance
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
