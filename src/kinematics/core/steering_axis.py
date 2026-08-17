"""Calculate topology-defined, motion-derived steering-response axes.

A physical steering axis is established directly from mechanism geometry, for
example the line through a double wishbone's lower and upper outer ball joints.
A virtual steering axis is instead inferred from the instantaneous rigid-body
motion of the upright.  That inference is meaningful only after the tangent
field has been given an explicit steering boundary condition: generic
screw-axis mathematics cannot separate steering from bump, roll, or heave.

At a solved state ``q_k``, the suspension topology supplies a steering-response
definition containing a steering coordinate ``s`` and zero or more travel
coordinates ``l``.  This module constructs the local response derivative by
solving

``C_q(q_k) q_s = 0``
``grad(s)(q_k) q_s = 1``
``L_q(q_k) q_s = 0``

where ``C`` contains the permanent mechanism constraints and ``L_q`` is the
stacked Jacobian of the held travel coordinates.  Every target is measured at
the current solved state and expressed as an absolute coordinate.  The values
therefore describe a counterfactual infinitesimal direction through the actual
configuration; they do not move the suspension back to design condition.

This target basis is intentionally independent of the authored sweep target basis. A
wheel-centre-height target is not a suspension-travel lock: steering about an
inclined kingpin would normally move the wheel centre vertically, so holding
that height forces a compensating jounce response.  Fitting the resulting
absolute upright motion yields the correct screw axis of combined steering and
jacking, but not an isolated steering axis.  Replacing the authored target
basis with the topology-owned suspension hold removes that accidental dependency.

For an ideal double wishbone whose declared wishbone angle fixes travel,
the response makes both outer ball-joint rates zero to first order.
The fitted upright line should consequently recover the physical kingpin axis
at every bumped or rolled state.  More complex topologies may define a
different hold basis.  The sensitivity solver obtains mechanism mobility from
``ker(C_q)`` and reports how completely the steering coordinate and holds span
that tangent space.  The result is unique only relative to that semantic
declaration; an absent, rank-deficient, or inconsistent basis is reported as
unavailable rather than resolved through an unnamed numerical choice.

The output here is a steering-*response* axis.  A sweep-path screw axis answers
a different question by combining all authored target rates along the actual
path and must remain a separately named result.  Both may reuse the generic
rigid-motion fitter, but they must never share their motion semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from kinematics.core.constraints import Constraint
from kinematics.core.coordinates import actuator_coordinate_matches
from kinematics.core.elements import UprightElement
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.screw_axis import (
    InstantaneousScrewAxis,
    RigidBodyTwist,
    ScrewAxisResult,
    ScrewAxisStatus,
    compute_screw_axis,
)
from kinematics.core.sensitivity import (
    TangentField,
    TangentSolveInfo,
    compute_state_tangents,
)
from kinematics.core.state import SuspensionState
from kinematics.core.steering_response import (
    SteeringResponseDefinition,
    SteeringResponseTargets,
    materialize_steering_response_targets,
)
from kinematics.core.suspensions.base import Suspension


class SteeringResponseStatus(StrEnum):
    """Outcome of steering-response orchestration and axis extraction."""

    VALID = "valid"
    NO_STEERING_ACTUATOR = "no_steering_actuator"
    NO_STEERING_RESPONSE_DEFINITION = "no_steering_response_definition"
    TANGENT_UNAVAILABLE = "tangent_unavailable"
    INCONSISTENT_TANGENT = "inconsistent_tangent"
    SCREW_AXIS_UNAVAILABLE = "screw_axis_unavailable"
    RANK_DEFICIENT = "rank_deficient"
    NON_FINITE = "non_finite"


@dataclass(frozen=True)
class SteeringResponseAxisResult:
    """Steering-owned result for one upright at one solved state."""

    upright_label: str
    point_keys: tuple[PointKey, ...]
    status: SteeringResponseStatus
    screw_axis: ScrewAxisResult | None = None
    message: str | None = None

    @property
    def axis(self) -> InstantaneousScrewAxis | None:
        """Return the extracted generic axis, if fitting succeeded."""
        return self.screw_axis.axis if self.screw_axis is not None else None

    @property
    def twist(self) -> RigidBodyTwist | None:
        """Return generic fit diagnostics, when fitting was attempted."""
        return self.screw_axis.twist if self.screw_axis is not None else None

    @property
    def point_count(self) -> int:
        """Return the number of points that participated in the generic fit."""
        return self.screw_axis.point_count if self.screw_axis is not None else 0

    @property
    def screw_axis_status(self) -> ScrewAxisStatus | None:
        """Return the generic fit/extraction outcome, when fitting was attempted."""
        return self.screw_axis.status if self.screw_axis is not None else None


@dataclass(frozen=True)
class SteeringResponseTangent:
    """One state's isolated steering tangent and its analytical diagnostics."""

    targets: SteeringResponseTargets | None
    tangent: TangentField | None
    solve_info: TangentSolveInfo | None
    status: SteeringResponseStatus
    message: str | None = None

    @property
    def valid(self) -> bool:
        """Whether the hold established one unique, consistent tangent."""
        return self.status is SteeringResponseStatus.VALID and self.tangent is not None


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
    steering_actuator = suspension.steering_actuator_coordinate()
    if steering_actuator is None:
        return SteeringResponseTangent(
            targets=None,
            tangent=None,
            solve_info=None,
            status=SteeringResponseStatus.NO_STEERING_ACTUATOR,
            message="The suspension topology has no steering actuator.",
        )

    definition = definition or suspension.resolve_suspension_hold(requested_option_id)
    response_targets = materialize_steering_response_targets(definition, state)
    if response_targets is None:
        return SteeringResponseTangent(
            targets=None,
            tangent=None,
            solve_info=None,
            status=SteeringResponseStatus.NO_STEERING_RESPONSE_DEFINITION,
            message=(
                "The steered suspension topology does not define a virtual "
                "steering response."
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
            response_targets.targets,
            post_derived_update=suspension.apply_ground_closure,
        )
    except Exception as error:  # noqa: BLE001 - one frame degrades explicitly
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=None,
            status=SteeringResponseStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Suspension hold '{response_targets.definition.provenance}' failed: "
                f"{type(error).__name__}: {error}."
            ),
        )

    if len(tangents) != len(response_targets.targets):
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Suspension hold '{response_targets.definition.provenance}' returned "
                f"{len(tangents)} tangent fields for "
                f"{len(response_targets.targets)} targets."
            ),
        )

    steering_tangent = tangents[0]
    if not actuator_coordinate_matches(
        response_targets.definition.steering_actuator,
        steering_tangent.target,
    ):
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.TANGENT_UNAVAILABLE,
            message=(
                f"Suspension hold '{response_targets.definition.provenance}' did not "
                "return "
                "its declared steering response at target index zero."
            ),
        )

    try:
        response_info = solve_info.response_for_target(steering_tangent.target_index)
    except KeyError as error:
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.TANGENT_UNAVAILABLE,
            message=f"Steering response diagnostics are unavailable: {error}",
        )

    diagnostic_prefix = _diagnostic_prefix(response_targets)
    if not solve_info.full_column_rank:
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.RANK_DEFICIENT,
            message=(
                f"{diagnostic_prefix} is underconstrained: rank "
                f"{solve_info.rank}/{solve_info.n_variables}, nullity "
                f"{solve_info.nullity}, mechanism mobility "
                f"{solve_info.mobility}, target rank "
                f"{solve_info.target_rank}/{solve_info.mobility}, smallest "
                f"target-space singular value "
                f"{solve_info.smallest_singular_value:.6g}, condition number "
                f"{solve_info.condition_number:.6g}, minimum target projection "
                f"ratio {solve_info.minimum_target_projection_ratio:.6g}."
            ),
        )
    if not response_info.finite:
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.NON_FINITE,
            message=f"{diagnostic_prefix} produced non-finite tangent values.",
        )
    if not response_info.rate_consistent:
        return SteeringResponseTangent(
            targets=response_targets,
            tangent=None,
            solve_info=solve_info,
            status=SteeringResponseStatus.INCONSISTENT_TANGENT,
            message=(
                f"{diagnostic_prefix} is rate-inconsistent: maximum constraint "
                f"residual {response_info.max_constraint_rate_residual:.6g}, "
                f"selected steering-rate residual "
                f"{response_info.selected_target_rate_residual:.6g}, maximum held "
                f"coordinate residual "
                f"{response_info.max_other_target_rate_residual:.6g}, tolerance "
                f"{response_info.consistency_tolerance:.6g}, condition number "
                f"{solve_info.condition_number:.6g}, mechanism mobility "
                f"{solve_info.mobility}, target rank "
                f"{solve_info.target_rank}/{solve_info.mobility}, minimum target "
                f"projection ratio "
                f"{solve_info.minimum_target_projection_ratio:.6g}."
            ),
        )

    return SteeringResponseTangent(
        targets=response_targets,
        tangent=steering_tangent,
        solve_info=solve_info,
        status=SteeringResponseStatus.VALID,
    )


def compute_steering_response_axes(
    suspension: Suspension,
    state: SuspensionState,
    *,
    constraints: list[Constraint] | None = None,
    derived_manager: DerivedPointsManager | None = None,
    definition: SteeringResponseDefinition | None = None,
    requested_option_id: str | None = None,
) -> tuple[SteeringResponseAxisResult, ...]:
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
    if response.status is SteeringResponseStatus.NO_STEERING_ACTUATOR:
        return ()
    if not response.valid:
        message = response.message or "The isolated steering response is unavailable."
        return tuple(
            SteeringResponseAxisResult(
                upright_label=upright.label,
                point_keys=tuple(dict.fromkeys(upright.point_keys)),
                status=response.status,
                message=message,
            )
            for upright in uprights
        )

    tangent = response.tangent
    assert tangent is not None
    return tuple(
        _steering_axis_result(
            upright.label,
            compute_screw_axis(
                positions=state.positions,
                tangent=tangent,
                point_keys=upright.point_keys,
            ),
        )
        for upright in uprights
    )


def compute_steering_response_axes_for_states(
    suspension: Suspension,
    states: Sequence[SuspensionState],
    *,
    requested_option_id: str | None = None,
) -> tuple[tuple[SteeringResponseAxisResult, ...], ...]:
    """Calculate aligned per-frame response axes with shared topology helpers."""
    constraints = suspension.constraints()
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    definition = suspension.resolve_suspension_hold(requested_option_id)
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


def _steering_axis_result(
    upright_label: str,
    screw_axis: ScrewAxisResult,
) -> SteeringResponseAxisResult:
    """Wrap one generic fit result in the steering-owned result boundary."""
    return SteeringResponseAxisResult(
        upright_label=upright_label,
        point_keys=screw_axis.point_keys,
        status=(
            SteeringResponseStatus.VALID
            if screw_axis.status is ScrewAxisStatus.VALID
            else SteeringResponseStatus.SCREW_AXIS_UNAVAILABLE
        ),
        screw_axis=screw_axis,
        message=screw_axis.message,
    )


def _diagnostic_prefix(targets: SteeringResponseTargets) -> str:
    """Describe a suspension hold and its coordinate identities compactly."""
    held = ", ".join(
        f"{coordinate_id}[{side.name.lower() if side is not None else 'shared'}]"
        for coordinate_id, side in zip(
            targets.definition.held_coordinate_ids,
            targets.definition.held_coordinate_sides,
            strict=True,
        )
    )
    return (
        f"Suspension hold '{targets.definition.provenance}' "
        f"(actuator {targets.definition.steering_coordinate_id}, "
        f"holds {held or 'none'})"
    )
