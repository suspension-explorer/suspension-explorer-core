"""Topology-owned suspension holds for steering-response derivatives.

The virtual steering axis is the screw axis of a *particular* velocity field;
the screw-axis fitter cannot decide which field is steering.  At a solved state
``q_k``, a topology declares one steering actuator ``s`` and any
suspension-travel coordinates ``l`` that it wants held.  The analytical
response solves the permanent constraint rates together with

``grad(s) q_s = 1`` and ``L_q q_s = 0``.

This module owns only the domain declaration and the corresponding absolute
target basis.  It neither performs the tangent solve nor fits an axis.  In
particular, it never consults authored sweep targets: a wheel-centre-height
target and a damper-length target can reach the same state but define different
partials.  The held values are measured from the supplied solved state so that
the target set is a complete, internally consistent description of the local
counterfactual problem.  Materialisation is pure and does not alter the state
or any authored target collection.

The tangent solver measures the local mechanism mobility and the rank of this
target basis explicitly.  A one-DOF mechanism can therefore require no holds,
while redundant but consistent holds are harmless.  An absent definition or
an insufficient or conflicting basis is reported rather than completed from
an authored target or an unnamed minimum-norm null-space direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kinematics.core.coordinates import (
    PhysicalCoordinate,
    actuator_coordinate_matches,
)
from kinematics.core.enums import TargetValueMode
from kinematics.core.holds import CoordinateHold
from kinematics.core.primitives.point_ref import Side
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import ScalarCoordinateTarget


class SuspensionHoldAvailability(StrEnum):
    """Whether one option can be evaluated for the current configuration."""

    AVAILABLE = "available"
    AVAILABLE_WITH_WARNING = "available_with_warning"
    UNAVAILABLE = "unavailable"


class SuspensionHoldSelectionSource(StrEnum):
    """How the resolved option was selected."""

    LAYOUT_DEFAULT = "layout_default"
    USER_OVERRIDE = "user_override"


@dataclass(frozen=True)
class SuspensionHoldOption:
    """One semantic, topology-owned suspension hold."""

    id: str
    label: str
    description: str
    hold: CoordinateHold
    availability: SuspensionHoldAvailability = SuspensionHoldAvailability.AVAILABLE
    warning: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.description:
            raise ValueError(
                "Suspension-hold option identity and copy must not be empty"
            )
        if (
            self.availability is SuspensionHoldAvailability.UNAVAILABLE
            and not self.unavailable_reason
        ):
            raise ValueError(
                f"Unavailable suspension-hold option '{self.id}' needs a reason"
            )
        if (
            self.availability is SuspensionHoldAvailability.AVAILABLE_WITH_WARNING
            and not self.warning
        ):
            raise ValueError(f"Suspension-hold option '{self.id}' needs warning copy")

    @property
    def composition_signature(self) -> tuple[object, ...]:
        """Return side-independent semantics required for axle composition."""
        return (
            self.label,
            self.description,
            tuple(
                (
                    coordinate.id,
                    coordinate.kind.value,
                    coordinate.unit,
                    coordinate.scope.value,
                )
                for coordinate in self.hold.coordinates
            ),
        )


@dataclass(frozen=True)
class SuspensionHoldCatalogue:
    """All suspension holds published by one topology for virtual steering."""

    default_option_id: str
    options: tuple[SuspensionHoldOption, ...]

    def __post_init__(self) -> None:
        ids = tuple(option.id for option in self.options)
        if len(set(ids)) != len(ids):
            raise ValueError("Suspension-hold option IDs must be unique")
        if self.default_option_id not in ids:
            raise ValueError("Suspension-hold default must identify a published option")
        default = self.option(self.default_option_id)
        if default.availability is SuspensionHoldAvailability.UNAVAILABLE:
            raise ValueError("Default suspension-hold option must be available")

    def option(self, option_id: str) -> SuspensionHoldOption:
        """Return a published option or raise a user-facing validation error."""
        match = next(
            (option for option in self.options if option.id == option_id),
            None,
        )
        if match is None:
            available = ", ".join(option.id for option in self.options)
            raise ValueError(
                f"Unknown suspension hold '{option_id}'. "
                f"Available options: {available}."
            )
        return match


@dataclass(frozen=True)
class SteeringResponseDefinition:
    """The topology's semantic definition of an isolated steering response.

    ``hold`` is deliberately explicit rather than inferred from
    ``drive_coordinates()``. They may be authored element lengths or internal
    analytical coordinates such as a signed chassis-arm angle. Their
    adequacy is a property of the local constraint and target Jacobians, not a
    hard-coded hold-count rule.
    """

    steering_actuator: PhysicalCoordinate
    hold: CoordinateHold
    owner: str
    definition_id: str
    requested_option_id: str | None = None
    selection_source: SuspensionHoldSelectionSource = (
        SuspensionHoldSelectionSource.LAYOUT_DEFAULT
    )
    label: str = "Suspension hold"
    description: str = "Topology-defined fixed-travel steering response."
    warning: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported declarations at topology build time."""
        if not self.owner or not self.definition_id:
            raise ValueError(
                "Steering-response definition owner and identifier must not be empty"
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
        return tuple(coordinate.id for coordinate in self.hold.coordinates)

    @property
    def held_coordinate_sides(self) -> tuple[Side | None, ...]:
        """Return structural sides aligned with :attr:`held_coordinate_ids`."""
        return tuple(coordinate.side for coordinate in self.hold.coordinates)


@dataclass(frozen=True)
class SteeringResponseTargets:
    """Materialized steering-response targets at one already-solved state."""

    definition: SteeringResponseDefinition
    targets: tuple[ScalarCoordinateTarget, ...]

    def __post_init__(self) -> None:
        """Require the steering target first and one target for every hold."""
        expected_count = 1 + len(self.definition.hold.coordinates)
        if len(self.targets) != expected_count:
            raise ValueError(
                "Steering-response target count does not match its definition"
            )
        if not actuator_coordinate_matches(
            self.definition.steering_actuator,
            self.targets[0],
        ):
            raise ValueError("Steering-response targets must begin with the actuator")
        if any(target.mode is not TargetValueMode.ABSOLUTE for target in self.targets):
            raise ValueError("Steering-response targets must be absolute")
        if any(
            target.coordinate_identity != coordinate.coordinate_identity
            for target, coordinate in zip(
                self.targets[1:], self.definition.hold.coordinates, strict=True
            )
        ):
            raise ValueError(
                "Steering-response holds do not match the definition order"
            )

    @property
    def steering_target(self) -> ScalarCoordinateTarget:
        """Return the unit-response coordinate, always at index zero."""
        return self.targets[0]

    @property
    def held_targets(self) -> tuple[ScalarCoordinateTarget, ...]:
        """Return declared suspension-hold targets in topology order."""
        return self.targets[1:]


def materialize_steering_response_targets(
    definition: SteeringResponseDefinition | None,
    state: SuspensionState,
) -> SteeringResponseTargets | None:
    """Build absolute steering-response targets measured at ``state``.

    The steering target is first, followed by topology-declared travel holds in
    their stored order.  Current-value factories own the coordinate measurement
    formulas, avoiding a steering-specific copy of projected-position or
    pin-centre-length math.
    """
    if definition is None:
        return None
    steering_target = definition.steering_actuator.current_value_target(state.positions)
    held_targets = definition.hold.materialize(state.positions)
    return SteeringResponseTargets(
        definition=definition,
        targets=(steering_target, *held_targets),
    )
