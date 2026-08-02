"""Calculate topology-defined, motion-derived steering-response axes.

A physical steering axis is established directly from mechanism geometry, for
example the line through a double wishbone's lower and upper outer ball joints.
A virtual steering axis is instead inferred from the instantaneous rigid-body
motion of the upright.  That inference is meaningful only after the velocity
field has been given an explicit steering boundary condition: generic
screw-axis mathematics cannot separate steering from bump, roll, or heave.

At a solved state ``q_k``, the suspension topology supplies a steering-response
definition containing a steering coordinate ``s`` and a minimal independent
set of travel coordinates ``l``.  This module constructs a local analytical
*steering probe* and solves

``C_q(q_k) q_s = 0``
``grad(s)(q_k) q_s = 1``
``L_q(q_k) q_s = 0``

where ``C`` contains the permanent mechanism constraints and ``L_q`` is the
stacked Jacobian of the held travel coordinates.  Every target is measured at
the current solved state and expressed as an absolute coordinate.  The values
therefore describe a counterfactual infinitesimal direction through the actual
configuration; they do not move the suspension back to design condition.

This probe is intentionally independent of the authored sweep target basis.  A
wheel-centre-height target is not a suspension-travel lock: steering about an
inclined kingpin would normally move the wheel centre vertically, so holding
that height forces a compensating jounce response.  Fitting the resulting
absolute upright motion yields the correct screw axis of combined steering and
jacking, but not an isolated steering axis.  Replacing the authored target
basis with the topology-owned probe removes that accidental dependency.

For an ideal double wishbone whose declared wishbone hinge angle fixes travel,
the probe makes both outer ball-joint velocities zero to first order.
The fitted upright line should consequently recover the physical kingpin axis
at every bumped or rolled state.  More complex topologies may define a
different complete hold basis.  The result is unique only relative to that
semantic declaration; an absent, rank-deficient, or inconsistent basis is
reported as unavailable rather than resolved through a numerical minimum-norm
choice.

The output here is a steering-*response* axis.  A sweep-path screw axis answers
a different question by combining all authored target rates along the actual
path and must remain a separately named result.  Both may reuse the generic
rigid-motion fitter, but they must never share their motion semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kinematics.core.constraints import Constraint
from kinematics.core.elements import UprightElement
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.rigid_motion import (
    ScrewAxisStatus,
    UprightScrewAxisResult,
    compute_upright_screw_axis,
    unavailable_upright_screw_axis,
)
from kinematics.core.sensitivity import (
    TangentField,
    TangentSolveInfo,
    compute_state_tangents,
)
from kinematics.core.state import SuspensionState
from kinematics.core.steering_response import (
    SteeringResponseDefinition,
    SteeringResponseProbe,
    materialize_steering_response_probe,
)
from kinematics.core.suspensions.base import Suspension


@dataclass(frozen=True)
class SteeringResponseTangent:
    """One state's isolated steering tangent and its analytical diagnostics."""

    probe: SteeringResponseProbe | None
    tangent: TangentField | None
    solve_info: TangentSolveInfo | None
    status: ScrewAxisStatus
    message: str | None = None

    @property
    def valid(self) -> bool:
        """Whether the probe established one unique, consistent tangent."""
        return self.status is ScrewAxisStatus.VALID and self.tangent is not None


def compute_steering_response_tangent(
    suspension: Suspension,
    state: SuspensionState,
    *,
    constraints: list[Constraint] | None = None,
    derived_manager: DerivedPointsManager | None = None,
    definition: SteeringResponseDefinition | None = None,
    requested_option_id: str | None = None,
) -> SteeringResponseTangent:
    """Return the isolated unit-rack tangent at one solved state.

    Optional prebuilt constraints and derived manager let sweep orchestration
    reuse immutable topology work across frames.  Failures are returned as
    local diagnostic results so one bad configuration does not abort a sweep.
    """
    steering_actuator = suspension.steering_actuator_dof()
    if steering_actuator is None:
        return SteeringResponseTangent(
            probe=None,
            tangent=None,
            solve_info=None,
            status=ScrewAxisStatus.NO_STEERING_ACTUATOR,
            message="The suspension topology has no steering actuator.",
        )

    definition = definition or suspension.resolve_steering_probe(requested_option_id)
    probe = materialize_steering_response_probe(definition, state)
    if probe is None:
        return SteeringResponseTangent(
            probe=None,
            tangent=None,
            solve_info=None,
            status=ScrewAxisStatus.NO_ISOLATION_DEFINITION,
            message=(
                "The steered suspension topology does not declare a complete "
                "independent steering-isolation basis."
            ),
        )

    active_constraints = (
        suspension.constraints() if constraints is None else constraints
    )
    active_derived_manager = (
        DerivedPointsManager(suspension.derived_spec())
        if derived_manager is None
        else derived_manager
    )
    try:
        tangents, solve_info = compute_state_tangents(
            state,
            active_constraints,
            active_derived_manager,
            probe.targets,
            post_derived_update=suspension.apply_ground_closure,
        )
    except Exception as error:  # noqa: BLE001 - one frame degrades explicitly
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=None,
            status=ScrewAxisStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Steering probe '{probe.definition.provenance}' failed: "
                f"{type(error).__name__}: {error}."
            ),
        )

    if len(tangents) != len(probe.targets):
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Steering probe '{probe.definition.provenance}' returned "
                f"{len(tangents)} tangent fields for {len(probe.targets)} targets."
            ),
        )

    steering_tangent = tangents[0]
    if not probe.definition.steering_actuator.matches(steering_tangent.target):
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Steering probe '{probe.definition.provenance}' did not return "
                "its declared steering response at target index zero."
            ),
        )

    try:
        response_info = solve_info.response_for_target(steering_tangent.target_index)
    except KeyError as error:
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.TANGENT_UNAVAILABLE,
            message=f"Steering response diagnostics are unavailable: {error}",
        )

    diagnostic_prefix = _diagnostic_prefix(probe)
    if not solve_info.full_column_rank:
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.RANK_DEFICIENT,
            message=(
                f"{diagnostic_prefix} is underconstrained: rank "
                f"{solve_info.rank}/{solve_info.n_variables}, nullity "
                f"{solve_info.nullity}, smallest singular value "
                f"{solve_info.smallest_singular_value:.6g}, condition number "
                f"{solve_info.condition_number:.6g}."
            ),
        )
    if not response_info.finite:
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.NON_FINITE,
            message=f"{diagnostic_prefix} produced non-finite tangent values.",
        )
    if not response_info.rate_consistent:
        return SteeringResponseTangent(
            probe=probe,
            tangent=None,
            solve_info=solve_info,
            status=ScrewAxisStatus.INCONSISTENT_TANGENT,
            message=(
                f"{diagnostic_prefix} is rate-inconsistent: maximum constraint "
                f"residual {response_info.max_constraint_rate_residual:.6g}, "
                f"selected steering-rate residual "
                f"{response_info.selected_target_rate_residual:.6g}, maximum held "
                f"coordinate residual "
                f"{response_info.max_other_target_rate_residual:.6g}, tolerance "
                f"{response_info.consistency_tolerance:.6g}, condition number "
                f"{solve_info.condition_number:.6g}."
            ),
        )

    return SteeringResponseTangent(
        probe=probe,
        tangent=steering_tangent,
        solve_info=solve_info,
        status=ScrewAxisStatus.VALID,
    )


def compute_steering_response_axes(
    suspension: Suspension,
    state: SuspensionState,
    *,
    constraints: list[Constraint] | None = None,
    derived_manager: DerivedPointsManager | None = None,
    definition: SteeringResponseDefinition | None = None,
    requested_option_id: str | None = None,
) -> tuple[UprightScrewAxisResult, ...]:
    """Return one isolated steering-response axis result per upright."""
    uprights = tuple(
        element
        for element in suspension.assembly().elements
        if isinstance(element, UprightElement)
    )
    response = compute_steering_response_tangent(
        suspension,
        state,
        constraints=constraints,
        derived_manager=derived_manager,
        definition=definition,
        requested_option_id=requested_option_id,
    )
    if response.status is ScrewAxisStatus.NO_STEERING_ACTUATOR:
        return ()
    if not response.valid:
        message = response.message or "The isolated steering response is unavailable."
        return tuple(
            unavailable_upright_screw_axis(
                upright.label,
                upright.point_keys,
                response.status,
                message,
            )
            for upright in uprights
        )

    return tuple(
        compute_upright_screw_axis(
            upright_label=upright.label,
            positions=state.positions,
            tangent=response.tangent,
            point_keys=upright.point_keys,
        )
        for upright in uprights
    )


def compute_steering_response_axes_for_states(
    suspension: Suspension,
    states: Sequence[SuspensionState],
    *,
    requested_option_id: str | None = None,
) -> tuple[tuple[UprightScrewAxisResult, ...], ...]:
    """Calculate aligned per-frame response axes with shared topology helpers."""
    constraints = suspension.constraints()
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    definition = suspension.resolve_steering_probe(requested_option_id)
    return tuple(
        compute_steering_response_axes(
            suspension,
            state,
            constraints=constraints,
            derived_manager=derived_manager,
            definition=definition,
        )
        for state in states
    )


def _diagnostic_prefix(probe: SteeringResponseProbe) -> str:
    """Describe a probe and its held coordinate identities compactly."""
    held = ", ".join(
        f"{coordinate_id}[{side.name.lower() if side is not None else 'shared'}]"
        for coordinate_id, side in zip(
            probe.definition.held_coordinate_ids,
            probe.definition.held_coordinate_sides,
            strict=True,
        )
    )
    return (
        f"Steering probe '{probe.definition.provenance}' "
        f"(actuator {probe.definition.steering_coordinate_id}, holds {held or 'none'})"
    )
