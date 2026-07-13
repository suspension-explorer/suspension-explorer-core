"""Double-wishbone full axle composed from two explicit corner models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    RackElement,
    SuspensionElement,
    map_element_points,
)
from kinematics.core.metrics.main import compute_metrics_for_axle_state
from kinematics.core.points.derived.manager import (
    DerivedPointsSpec,
    PositionFn,
    PositionValue,
)
from kinematics.core.primitives.enums import Axis, PointID
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
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension

if TYPE_CHECKING:
    from kinematics.core.metrics.main import AxleMetricRows
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

    TYPE_KEY: ClassVar[str] = "double_wishbone_axle"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OUTPUT_POINTS: ClassVar[tuple[PointKey, ...]] = tuple(
        PointRef(Side.LEFT, point) for point in DoubleWishboneSuspension.OUTPUT_POINTS
    ) + tuple(
        PointRef(Side.RIGHT, point) for point in DoubleWishboneSuspension.OUTPUT_POINTS
    )

    corners: dict[Side, DoubleWishboneSuspension] = field(default_factory=dict)

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

        self._initial_state = SuspensionState(positions, free_points)
        return self._initial_state

    def free_points(self) -> Sequence[PointKey]:
        """Return both corners' free points under side-qualified keys."""
        return tuple(
            PointRef(side, point)
            for side, corner in self.corners.items()
            for point in corner.free_points()
        )

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
        return constraints

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
        return elements + (
            RackElement(
                label="Steering Rack",
                left_inner=PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                right_inner=PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                translation_axis=Axis.Y,
            ),
        )
