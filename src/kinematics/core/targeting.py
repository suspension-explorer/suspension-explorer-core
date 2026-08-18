"""Sweep assembly and geometry-independent coordinate vocabulary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from kinematics.core.coordinates import CoordinateTarget, CoordinateType
from kinematics.core.enums import (
    ActuatorPositionCoordinateID,
    ElementLengthCoordinateID,
    PointID,
)
from kinematics.core.primitives.point_ref import Side

if TYPE_CHECKING:
    from kinematics.core.holds import CoordinateHold


def _empty_coordinate_hold() -> CoordinateHold:
    """Construct the default lazily to avoid a coordinate/hold import cycle."""
    from kinematics.core.holds import CoordinateHold

    return CoordinateHold()


@dataclass
class SweepConfig:
    """Configuration for corresponding steps across scalar target dimensions."""

    target_sweeps: Sequence[Sequence[CoordinateTarget]]
    hold: CoordinateHold = field(default_factory=_empty_coordinate_hold)
    suspension_hold_id: str = "layout_default"

    def __post_init__(self) -> None:
        if not self.suspension_hold_id.strip():
            raise ValueError("Suspension-hold ID must not be empty")
        if not self.target_sweeps:
            return

        lengths = [len(sweep) for sweep in self.target_sweeps]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All sweep dimensions must have the same length. Got: {lengths}"
            )

        for step_index in range(lengths[0]):
            coordinates = [
                *(dimension[step_index].coordinate for dimension in self.target_sweeps),
                *self.hold.coordinates,
            ]
            indices_by_identity: dict[tuple[object, ...], list[int]] = {}
            for target_index, coordinate in enumerate(coordinates):
                identity = coordinate.coordinate_identity
                indices_by_identity.setdefault(identity, []).append(target_index)
            duplicate = next(
                (
                    (identity, indices)
                    for identity, indices in indices_by_identity.items()
                    if len(indices) > 1
                ),
                None,
            )
            if duplicate is None:
                continue
            _identity, target_indices = duplicate
            description = coordinates[target_indices[0]].coordinate_description
            rendered_indices = ", ".join(str(index) for index in target_indices)
            raise ValueError(
                f"Sweep controls {rendered_indices} drive or hold the same "
                f"{description} at step {step_index}."
            )

    @property
    def n_steps(self) -> int:
        """Number of steps in the sweep."""
        if not self.target_sweeps:
            return 0
        return len(self.target_sweeps[0])


# These are syntactic wire vocabularies, not promises that a coordinate exists
# or can move in every suspension. Topology availability is a build-time rule.
POINT_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    point.name.lower() for point in PointID if point is not PointID.NOT_ASSIGNED
)
ACTUATOR_POSITION_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    coordinate.value for coordinate in ActuatorPositionCoordinateID
)
ELEMENT_LENGTH_TARGET_IDS: Final[tuple[str, ...]] = tuple(
    coordinate.value for coordinate in ElementLengthCoordinateID
)
SidePolicy = Literal["corner", "shared"]


def sweep_target_side_policy(
    coordinate_type: CoordinateType,
    coordinate_id: str,
) -> SidePolicy:
    """Return static side ownership for one globally recognized coordinate."""
    try:
        policy = {
            CoordinateType.POINT: lambda: PointID[
                coordinate_id.upper()
            ].sweep_side_policy,
            CoordinateType.ACTUATOR_POSITION: lambda: ActuatorPositionCoordinateID(
                coordinate_id
            ).side_policy,
            CoordinateType.ELEMENT_LENGTH: lambda: ElementLengthCoordinateID(
                coordinate_id
            ).side_policy,
        }[coordinate_type]()
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Unknown {coordinate_type.value} coordinate ID '{coordinate_id}'."
        ) from error
    return policy.value


def resolve_published_target_side(
    description: str,
    candidate_sides: Sequence[Side | None],
    requested_side: Side | None,
) -> Side | None:
    """Resolve an explicit side against published coordinate ownership."""
    available_sides = tuple(dict.fromkeys(candidate_sides))
    if requested_side in available_sides:
        return requested_side

    if requested_side is None:
        names = " or ".join(
            side.name.lower() for side in available_sides if side is not None
        )
        raise ValueError(f"{description} requires side {names}.")

    if None in available_sides:
        raise ValueError(f"{description} is shared and does not accept a side.")

    names = ", ".join(side.name.lower() for side in available_sides if side is not None)
    raise ValueError(
        f"{description} is unavailable on side '{requested_side.name.lower()}'. "
        f"Available sides: {names}."
    )


class PointTargetVocabularyItem(TypedDict):
    """One JSON-native point-coordinate vocabulary entry."""

    type: Literal["point"]
    id: str
    label: str
    featured: bool
    side_policy: SidePolicy


class ActuatorPositionVocabularyItem(TypedDict):
    """One JSON-native actuator-position vocabulary entry."""

    type: Literal["actuator_position"]
    id: str
    label: str
    featured: bool
    side_policy: SidePolicy


class ElementLengthVocabularyItem(TypedDict):
    """One JSON-native element-length vocabulary entry."""

    type: Literal["element_length"]
    id: str
    label: str
    unit: str
    featured: bool
    side_policy: SidePolicy


class SweepTargetVocabulary(TypedDict):
    """Geometry-independent identifiers accepted by sweep documents."""

    positions: list[PointTargetVocabularyItem | ActuatorPositionVocabularyItem]
    element_lengths: list[ElementLengthVocabularyItem]


def _point_target_label(point_id: str) -> str:
    """Humanize one canonical point ID for geometry-independent editors."""
    label = point_id.replace("_", " ").title()
    return (
        label.replace("Arb", "ARB").replace("U Bar", "U-Bar").replace("T Bar", "T-Bar")
    )


def sweep_target_vocabulary() -> SweepTargetVocabulary:
    """Return the stable, JSON-native sweep target vocabulary."""
    featured_point_order = ("wheel_center",)
    featured_point_ids = frozenset(featured_point_order)
    ordered_point_ids = (
        *featured_point_order,
        *(
            point_id
            for point_id in POINT_TARGET_IDS
            if point_id not in featured_point_ids
        ),
    )
    positions: list[PointTargetVocabularyItem | ActuatorPositionVocabularyItem] = [
        ActuatorPositionVocabularyItem(
            type="actuator_position",
            id=ActuatorPositionCoordinateID.RACK.value,
            label="Rack",
            featured=True,
            side_policy=sweep_target_side_policy(
                CoordinateType.ACTUATOR_POSITION,
                ActuatorPositionCoordinateID.RACK.value,
            ),
        )
    ]
    positions.extend(
        PointTargetVocabularyItem(
            type="point",
            id=point_id,
            label=_point_target_label(point_id),
            featured=point_id in featured_point_ids,
            side_policy=sweep_target_side_policy(CoordinateType.POINT, point_id),
        )
        for point_id in ordered_point_ids
    )
    return {
        "positions": positions,
        "element_lengths": [
            {
                "type": "element_length",
                "id": coordinate.value,
                "label": coordinate.label,
                "unit": coordinate.unit,
                "featured": coordinate is ElementLengthCoordinateID.DAMPER,
                "side_policy": sweep_target_side_policy(
                    CoordinateType.ELEMENT_LENGTH,
                    coordinate.value,
                ),
            }
            for coordinate in ElementLengthCoordinateID
        ],
    }
