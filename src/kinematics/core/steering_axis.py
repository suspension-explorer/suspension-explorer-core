"""Instantaneous steering axes extracted from analytical tangent fields."""

from collections.abc import Sequence

from kinematics.core.elements import UprightElement
from kinematics.core.rigid_motion import (
    UprightScrewAxisResult,
    compute_upright_screw_axis,
)
from kinematics.core.sensitivity import TangentField, TangentSolveInfo
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension


def compute_instantaneous_steering_axes(
    suspension: Suspension,
    state: SuspensionState,
    tangent_fields: Sequence[TangentField] | None,
    tangent_info: TangentSolveInfo | None,
) -> tuple[UprightScrewAxisResult, ...]:
    """Return one rack-partial screw-axis result for every upright.

    The rack tangent is selected semantically through the suspension's steering
    actuator declaration. Suspensions without a steering actuator have no
    instantaneous steering axes and return an empty tuple.
    """
    steering_actuator = suspension.steering_actuator_dof()
    if steering_actuator is None:
        return ()

    matching_tangents = [
        tangent
        for tangent in tangent_fields or ()
        if steering_actuator.matches(tangent.target)
    ]
    steering_tangent = matching_tangents[0] if len(matching_tangents) == 1 else None
    tangent_rank_deficient = (
        tangent_info.rank_deficient if tangent_info is not None else False
    )

    uprights = (
        element
        for element in suspension.assembly().elements
        if isinstance(element, UprightElement)
    )
    return tuple(
        compute_upright_screw_axis(
            upright_label=upright.label,
            positions=state.positions,
            tangent=steering_tangent,
            point_keys=upright.point_keys,
            tangent_rank_deficient=tangent_rank_deficient,
        )
        for upright in uprights
    )
