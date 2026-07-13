"""Double-wishbone full axle composed from two explicit corner models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    RackElement,
    SuspensionElement,
    map_element_points,
)
from kinematics.core.enums import Axis, PointID, SuspensionType
from kinematics.core.metrics.main import compute_metrics_for_axle_state
from kinematics.core.points.derived.manager import (
    DerivedPointsSpec,
    PositionFn,
    PositionValue,
)
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import (
    PointKey,
    PointRef,
    Side,
    side_qualified,
)
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle.mechanisms import (
    ArbNone,
    AxleArb,
    AxleHeaveLink,
    HeaveLinkNone,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension

if TYPE_CHECKING:
    from kinematics.core.diagnostics import DiagnosticIssue
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.main import AxleMetricRows, MetricRow
    from kinematics.core.sensitivity import TangentField


class _CornerPositionView(Mapping[PointID, Any]):
    """Expose one side of a PointRef-keyed position map as PointID-keyed."""

    def __init__(self, positions: Mapping[PointKey, Any], side: Side) -> None:
        self._positions = positions
        self._side = side

    def __getitem__(self, point: PointID) -> Any:
        return self._positions[PointRef(self._side, point)]

    def __iter__(self) -> Iterator[PointID]:
        for key in self._positions:
            if isinstance(key, PointRef) and key.side is self._side:
                yield key.point

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass
class DoubleWishboneAxleSuspension(Suspension):
    """Two double-wishbone corners coupled by their inboard trackrod points."""

    TYPE_KEY: ClassVar[SuspensionType] = SuspensionType.DOUBLE_WISHBONE_AXLE
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    corners: dict[Side, DoubleWishboneSuspension] = field(default_factory=dict)
    anti_roll: AxleArb = field(default_factory=ArbNone, kw_only=True)
    heave_link: AxleHeaveLink = field(default_factory=HeaveLinkNone, kw_only=True)

    @property
    def is_axle(self) -> bool:
        """Whether this topology composes multiple corner suspensions."""
        return True

    def validate_hardpoints(self) -> None:
        """Require one explicitly sided corner on each side."""
        if set(self.corners) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("Axle requires exactly LEFT and RIGHT corner models.")
        for side, corner in self.corners.items():
            if corner.side is not side:
                raise ValueError(
                    f"Axle {side.name.lower()} corner must declare side "
                    f"'{side.name.lower()}'."
                )
            corner.validate_hardpoints()
        self.anti_roll.validate(self)
        self.heave_link.validate(self)

    def initial_state(self) -> SuspensionState:
        """Combine both corner states under side-qualified point keys."""
        if self._initial_state is not None:
            return self._initial_state

        positions: dict[PointKey, Point3] = {}
        free_points: set[PointKey] = set()
        for side, corner in self.corners.items():
            corner_state = corner.initial_state()
            positions.update(
                {
                    PointRef(side, point): position.copy()
                    for point, position in corner_state.positions.items()
                }
            )
            free_points.update(
                PointRef(side, point) for point in corner_state.free_points
            )

        state = SuspensionState(positions, free_points)
        self.anti_roll.add_to_state(state)
        state.free_points_order = sorted(state.free_points)
        self._initial_state = state
        return self._initial_state

    def free_points(self) -> Sequence[PointKey]:
        """Return both corners' free points under side-qualified keys."""
        corner_points = tuple(
            PointRef(side, point)
            for side, corner in self.corners.items()
            for point in corner.free_points()
        )
        return (*corner_points, *self.anti_roll.free_points)

    def output_points(self) -> tuple[PointKey, ...]:
        """Return composed corner and shared mechanism output points."""
        corner_points = tuple(
            side_qualified(side, point)
            for side in (Side.LEFT, Side.RIGHT)
            for point in self.corners[side].output_points()
        )
        return tuple(dict.fromkeys((*corner_points, *self.anti_roll.output_points)))

    def constraints(self) -> list[Constraint]:
        """Combine remapped corner constraints and trackrod coupling."""
        constraints = [
            constraint.remap(lambda point, side=side: side_qualified(side, point))
            for side, corner in self.corners.items()
            for constraint in corner.constraints()
        ]
        left = (
            self.corners[Side.LEFT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        right = (
            self.corners[Side.RIGHT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        constraints.append(
            DistanceConstraint(
                PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                compute_point_point_distance(left, right),
            )
        )
        constraints.extend(self.anti_roll.constraints(self))
        return constraints

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Compose axle mechanism derivative declarations."""
        return (
            *self.anti_roll.derivative_metric_definitions(self),
            *self.heave_link.derivative_metric_definitions(self),
        )

    def topology_metric_values(self, state: SuspensionState) -> MetricRow:
        """Compose axle mechanism state metrics."""
        row: MetricRow = OrderedDict()
        row.update(self.anti_roll.topology_metric_values(self, state))
        row.update(self.heave_link.topology_metric_values(state))
        return row

    def topology_diagnostics(
        self,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        """Return diagnostics owned by shared axle mechanisms."""
        return self.anti_roll.topology_diagnostics(self, states)

    def derived_spec(self) -> DerivedPointsSpec:
        """Combine remapped corner derived-point specifications."""
        functions: dict[PointKey, PositionFn] = {}
        dependencies: dict[PointKey, set[PointKey]] = {}
        for side, corner in self.corners.items():
            spec = corner.derived_spec()
            for point, function in spec.functions.items():
                functions[PointRef(side, point)] = self._wrap_derived(function, side)
            for point, point_dependencies in spec.dependencies.items():
                dependencies[PointRef(side, point)] = {
                    PointRef(side, dependency) for dependency in point_dependencies
                }
        return DerivedPointsSpec(functions, dependencies)

    @staticmethod
    def _wrap_derived(function: PositionFn, side: Side) -> PositionFn:
        """Adapt a PointID-based derived function to an axle position map."""

        def wrapped(positions: dict[PointKey, PositionValue]) -> PositionValue:
            # _CornerPositionView duck-types as the positions mapping the derived
            # function expects; ty cannot see the structural match through the view.
            return function(_CornerPositionView(positions, side))  # ty: ignore[invalid-argument-type]

        return wrapped

    def corner_state(self, state: SuspensionState, side: Side) -> SuspensionState:
        """Return one side of an axle state with its side qualifiers removed."""
        positions = {
            key.point: position
            for key, position in state.positions.items()
            if isinstance(key, PointRef) and key.side is side
        }
        free_points = {
            key.point
            for key in state.free_points
            if isinstance(key, PointRef) and key.side is side
        }
        return SuspensionState(positions, free_points)

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """Reject axle-level use of a per-corner construction."""
        raise NotImplementedError("Use corner_state() and the selected corner.")

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """Reject axle-level use of a per-corner construction."""
        raise NotImplementedError("Use corner_state() and the selected corner.")

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Require every axle target to select a physical side."""
        if side not in (Side.LEFT, Side.RIGHT):
            raise ValueError(
                f"Axle sweep target for '{point.name}' requires side left or right."
            )
        return PointRef(side, point)

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "AxleMetricRows":
        """Compute structural corner and axle-level metric rows."""
        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_axle_state(
            state,
            self,
            self.config,
            tangents,
        )

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return side-qualified corner elements and the trackrod coupling."""
        elements = tuple(
            map_element_points(
                element,
                lambda point, side=side: side_qualified(side, point),
                label=f"{side.name.title()} {element.label}",
            )
            for side, corner in self.corners.items()
            for element in corner.elements()
        )
        base_elements = elements + (
            RackElement(
                label="Steering Rack",
                left_inner=PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                right_inner=PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                translation_axis=Axis.Y,
            ),
        )
        return (
            *base_elements,
            *self.anti_roll.elements(self),
            *self.heave_link.elements(),
        )
