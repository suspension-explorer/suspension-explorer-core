"""Topology-owned, counterfactual steering-probe target construction.

The virtual steering axis is the screw axis of a *particular* velocity field;
the screw-axis fitter cannot decide which field is steering.  At a solved state
``q_k``, a topology declares one steering actuator ``s`` and a minimal,
independent collection of suspension-travel coordinates ``l``.  The eventual
analytical probe solves the permanent constraint rates together with

``grad(s) q_s = 1`` and ``L_q q_s = 0``.

This module owns only the domain declaration and the corresponding absolute
target basis.  It neither performs the tangent solve nor fits an axis.  In
particular, it never consults authored sweep targets: a wheel-centre-height
target and a damper-length target can reach the same state but define different
partials.  The held values are measured from the supplied solved state so that
the target set is a complete, internally consistent description of the local
counterfactual problem.  Materialisation is pure and does not alter the state
or any authored target collection.

An absent definition means the topology has not established a complete,
independent isolation basis.  Callers must report the steering response as
unavailable rather than manufacture a hold from an authored target or a
minimum-norm null-space direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kinematics.core.enums import TargetValueMode
from kinematics.core.isolation_coordinates import HingeAngleCoordinate
from kinematics.core.primitives.point_ref import Side
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import (
    ActuatorDOF,
    DriveCoordinate,
    ScalarCoordinateTarget,
)

type IsolationCoordinate = DriveCoordinate | HingeAngleCoordinate


class SteeringProbeOptionClass(StrEnum):
    """Meaning of one topology-declared steering isolation basis."""

    CANONICAL = "canonical"
    EQUIVALENT = "equivalent"
    DIAGNOSTIC = "diagnostic"


class SteeringProbeOptionAvailability(StrEnum):
    """Whether one option can be evaluated for the current configuration."""

    AVAILABLE = "available"
    AVAILABLE_WITH_WARNING = "available_with_warning"
    UNAVAILABLE = "unavailable"


class SteeringProbeSelectionSource(StrEnum):
    """How the resolved option was selected."""

    LAYOUT_DEFAULT = "layout_default"
    USER_OVERRIDE = "user_override"


@dataclass(frozen=True)
class SteeringProbeOption:
    """One semantic, topology-owned isolation choice."""

    id: str
    label: str
    description: str
    option_class: SteeringProbeOptionClass
    held_coordinates: tuple[IsolationCoordinate, ...]
    availability: SteeringProbeOptionAvailability = (
        SteeringProbeOptionAvailability.AVAILABLE
    )
    warning: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.description:
            raise ValueError(
                "Steering-probe option identity and copy must not be empty"
            )
        if not self.held_coordinates:
            raise ValueError(
                f"Steering-probe option '{self.id}' must hold at least one coordinate"
            )
        identities = tuple(
            coordinate.coordinate_identity for coordinate in self.held_coordinates
        )
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"Steering-probe option '{self.id}' has duplicate held coordinates"
            )
        if (
            self.availability is SteeringProbeOptionAvailability.UNAVAILABLE
            and not self.unavailable_reason
        ):
            raise ValueError(
                f"Unavailable steering-probe option '{self.id}' needs a reason"
            )
        if (
            self.availability is SteeringProbeOptionAvailability.AVAILABLE_WITH_WARNING
            and not self.warning
        ):
            raise ValueError(f"Steering-probe option '{self.id}' needs warning copy")


@dataclass(frozen=True)
class SteeringProbeCatalogue:
    """All steering-isolation semantics published by one topology."""

    default_option_id: str
    options: tuple[SteeringProbeOption, ...]

    def __post_init__(self) -> None:
        ids = tuple(option.id for option in self.options)
        if len(set(ids)) != len(ids):
            raise ValueError("Steering-probe option IDs must be unique")
        if self.default_option_id not in ids:
            raise ValueError("Steering-probe default must identify a published option")
        canonical = tuple(
            option
            for option in self.options
            if option.option_class is SteeringProbeOptionClass.CANONICAL
        )
        if len(canonical) != 1 or canonical[0].id != self.default_option_id:
            raise ValueError(
                "Steering-probe catalogue must have exactly one canonical default"
            )
        if canonical[0].availability is SteeringProbeOptionAvailability.UNAVAILABLE:
            raise ValueError("Canonical steering-probe option must be available")

    def option(self, option_id: str) -> SteeringProbeOption:
        """Return a published option or raise a user-facing validation error."""
        match = next(
            (option for option in self.options if option.id == option_id),
            None,
        )
        if match is None:
            available = ", ".join(option.id for option in self.options)
            raise ValueError(
                f"Unknown steering-probe isolation '{option_id}'. "
                f"Available options: {available}."
            )
        return match


@dataclass(frozen=True)
class SteeringResponseDefinition:
    """The topology's semantic definition of an isolated steering response.

    ``held_coordinates`` are deliberately explicit rather than inferred from
    ``drive_coordinates()``. They may be authored element lengths or internal
    analytical coordinates such as a signed chassis-hinge angle. A topology
    publishes one only when it forms part of a complete independent isolation
    basis for that mechanism.
    """

    steering_actuator: ActuatorDOF
    held_coordinates: tuple[IsolationCoordinate, ...]
    owner: str
    definition_id: str
    requested_option_id: str | None = None
    selection_source: SteeringProbeSelectionSource = (
        SteeringProbeSelectionSource.LAYOUT_DEFAULT
    )
    option_class: SteeringProbeOptionClass = SteeringProbeOptionClass.CANONICAL
    label: str = "Steering isolation"
    description: str = "Topology-defined fixed-travel steering response."
    warning: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported declarations at topology build time."""
        if not self.owner or not self.definition_id:
            raise ValueError(
                "Steering-response definition owner and identifier must not be empty"
            )
        identities = tuple(
            coordinate.coordinate_identity for coordinate in self.held_coordinates
        )
        if len(set(identities)) != len(identities):
            raise ValueError(
                "Steering-response definition has duplicate held coordinates"
            )

    @property
    def provenance(self) -> str:
        """Return a stable, human-readable owner/definition identifier."""
        return f"{self.owner}:{self.definition_id}"

    @property
    def steering_coordinate_id(self) -> str:
        """Return the topology-owned steering coordinate identifier."""
        return self.steering_actuator.id

    @property
    def held_coordinate_ids(self) -> tuple[str, ...]:
        """Return held coordinate IDs in deterministic topology order."""
        return tuple(coordinate.id for coordinate in self.held_coordinates)

    @property
    def held_coordinate_sides(self) -> tuple[Side | None, ...]:
        """Return structural sides aligned with :attr:`held_coordinate_ids`."""
        return tuple(coordinate.side for coordinate in self.held_coordinates)


@dataclass(frozen=True)
class SteeringResponseProbe:
    """A materialized steering target basis at one already-solved state."""

    definition: SteeringResponseDefinition
    targets: tuple[ScalarCoordinateTarget, ...]

    def __post_init__(self) -> None:
        """Require the steering target first and one target for every hold."""
        expected_count = 1 + len(self.definition.held_coordinates)
        if len(self.targets) != expected_count:
            raise ValueError(
                "Steering-response probe target count does not match its definition"
            )
        if not self.definition.steering_actuator.matches(self.targets[0]):
            raise ValueError("Steering-response probe must begin with its actuator")
        if any(target.mode is not TargetValueMode.ABSOLUTE for target in self.targets):
            raise ValueError("Steering-response probe targets must be absolute")
        if any(
            target.coordinate_identity != coordinate.coordinate_identity
            for target, coordinate in zip(
                self.targets[1:], self.definition.held_coordinates, strict=True
            )
        ):
            raise ValueError(
                "Steering-response probe holds do not match its definition order"
            )

    @property
    def steering_target(self) -> ScalarCoordinateTarget:
        """Return the unit-response coordinate, always at index zero."""
        return self.targets[0]

    @property
    def held_targets(self) -> tuple[ScalarCoordinateTarget, ...]:
        """Return declared isolation targets in topology order."""
        return self.targets[1:]


def materialize_steering_response_targets(
    definition: SteeringResponseDefinition,
    state: SuspensionState,
) -> tuple[ScalarCoordinateTarget, ...]:
    """Build absolute steering-probe targets measured at ``state``.

    The steering target is first, followed by topology-declared travel holds in
    their stored order.  Current-value factories own the coordinate measurement
    formulas, avoiding a steering-specific copy of projected-position or
    pin-centre-length math.
    """
    steering_target = definition.steering_actuator.current_value_target(state.positions)
    held_targets = tuple(
        coordinate.current_value_target(state.positions)
        for coordinate in definition.held_coordinates
    )
    return (steering_target, *held_targets)


def materialize_steering_response_probe(
    definition: SteeringResponseDefinition | None,
    state: SuspensionState,
) -> SteeringResponseProbe | None:
    """Materialize a probe, or return unavailable when topology declares none."""
    if definition is None:
        return None
    return SteeringResponseProbe(
        definition=definition,
        targets=materialize_steering_response_targets(definition, state),
    )
