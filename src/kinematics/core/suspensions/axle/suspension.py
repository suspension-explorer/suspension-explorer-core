"""Generic full axle composed from two explicit corner suspensions.

There is one axle class for every corner architecture: the composer
side-qualifies and couples the two built corners it is given, and reads the
corner role hooks for anything architecture-specific (currently the rack
coupling). Input mirroring belongs to the builder. New locating architectures
add a corner class, not an axle class.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.coordinates import PhysicalCoordinate
from kinematics.core.elements import (
    ElementType,
    RackElement,
    SuspensionElement,
    VariableLengthLinkElement,
    map_element_points,
)
from kinematics.core.enums import (
    ActuatorPositionCoordinateID,
    Axis,
    ElementLengthCoordinateID,
    PointID,
    Scope,
    SuspensionType,
)
from kinematics.core.holds import CoordinateHold
from kinematics.core.metrics.main import AxleMetricRows, compute_metrics_for_axle_state
from kinematics.core.points.derived.ground import (
    seed_from_contact_centres,
    solve_axle_wheel_contact_centres,
)
from kinematics.core.points.derived.manager import (
    DerivedPointsSpec,
    PositionFn,
    PositionValue,
)
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import (
    PointKey,
    PointRef,
    Side,
    side_qualified,
)
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.core.road import RoadPlane
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.axle.mechanisms import (
    ArbNone,
    AxleArb,
    AxleHeaveLink,
    HeaveLinkNone,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.targeting import (
    PointTargetAxis,
    TargetKind,
    resolve_published_target_side,
    sweep_target_side_policy,
)

if TYPE_CHECKING:
    from kinematics.core.diagnostics import DiagnosticIssue
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.registry import MetricSpec
    from kinematics.core.screw_axis import UprightScrewAxisResult
    from kinematics.core.sensitivity import TangentField
    from kinematics.core.steering_response import SuspensionHoldCatalogue


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
class AxleSuspension(Suspension):
    """Two corner suspensions coupled by shared rack and axle mechanisms."""

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    type_key: SuspensionType = field(kw_only=True)
    corners: dict[Side, CornerSuspension] = field(default_factory=dict)
    anti_roll: AxleArb = field(default_factory=ArbNone, kw_only=True)
    heave_link: AxleHeaveLink = field(default_factory=HeaveLinkNone, kw_only=True)

    @property
    def is_axle(self) -> bool:
        """Whether this topology composes multiple corner suspensions."""
        return True

    def reported_type_key(self) -> SuspensionType:
        """Return the builder-supplied geometry type identity."""
        return self.type_key

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
        # Raises when one corner is steered and the other is not.
        self.rack_attachment_points()
        self.anti_roll.validate(self)
        self.heave_link.validate(self)

    def rack_attachment_points(self) -> tuple[PointID, PointID] | None:
        """
        Rack attachment points as (left, right), or None for an unsteered axle.

        Raises:
            ValueError: If exactly one corner exposes a rack attachment.
        """
        left = self.corners[Side.LEFT].rack_attachment_point()
        right = self.corners[Side.RIGHT].rack_attachment_point()
        if (left is None) != (right is None):
            raise ValueError(
                "Axle corners disagree on rack attachment: one corner is "
                "steered and the other is not."
            )
        if left is None or right is None:
            return None
        return (left, right)

    def required_actuator_coordinates(self) -> tuple[PhysicalCoordinate, ...]:
        """Require one shared rack coordinate for a steered axle."""
        steering = self.steering_actuator_coordinate()
        return (steering,) if steering is not None else ()

    def steering_actuator_coordinate(self) -> PhysicalCoordinate | None:
        """Return the shared rack coordinate for a steered axle."""
        return next(
            (
                coordinate
                for coordinate in self.drive_coordinates()
                if coordinate.kind is TargetKind.ACTUATOR_POSITION
                and coordinate.id == ActuatorPositionCoordinateID.RACK
            ),
            None,
        )

    def drive_coordinates(self) -> tuple[PhysicalCoordinate, ...]:
        """Compose sided corner dampers and axle-owned variable coordinates."""
        coordinates: list[PhysicalCoordinate] = []
        rack = self.rack_attachment_points()
        if rack is not None:
            coordinates.append(
                PhysicalCoordinate(
                    id=ActuatorPositionCoordinateID.RACK,
                    kind=TargetKind.ACTUATOR_POSITION,
                    label=ActuatorPositionCoordinateID.RACK.label,
                    unit=ActuatorPositionCoordinateID.RACK.unit,
                    point_keys=(
                        PointRef(Side.LEFT, rack[0]),
                        PointRef(Side.RIGHT, rack[1]),
                    ),
                    scope=Scope.AXLE,
                    direction=PointTargetAxis(Axis.Y),
                )
            )
        for side in (Side.LEFT, Side.RIGHT):
            for coordinate in self.corners[side].drive_coordinates():
                if coordinate.kind is TargetKind.ACTUATOR_POSITION:
                    continue
                coordinates.append(
                    coordinate.map_points(
                        lambda point, selected_side=side: side_qualified(
                            selected_side, point
                        ),
                        side=side,
                    )
                )

        for element in self.heave_link.elements():
            if (
                isinstance(element, VariableLengthLinkElement)
                and element.type is ElementType.HEAVE_LINK
            ):
                coordinate_id = ElementLengthCoordinateID.HEAVE_LINK
                coordinates.append(
                    PhysicalCoordinate(
                        id=coordinate_id,
                        kind=TargetKind.ELEMENT_LENGTH,
                        label=coordinate_id.label,
                        unit=coordinate_id.unit,
                        point_keys=(element.point_a, element.point_b),
                        scope=Scope.AXLE,
                    )
                )
        return tuple(coordinates)

    def suspension_hold_catalogue(self) -> "SuspensionHoldCatalogue | None":
        """Compose compatible corner options into one axle-level selection.

        Each semantic option remains one user choice but expands to one
        independent held coordinate per corner.  Shared axle mechanisms are
        deliberately excluded: their motion may follow from the two corner
        travel coordinates and is not an additional jounce degree of freedom.
        """
        from kinematics.core.steering_response import (
            SuspensionHoldAvailability,
            SuspensionHoldCatalogue,
            SuspensionHoldOption,
        )

        catalogues = tuple(
            self.corners[side].suspension_hold_catalogue()
            for side in (Side.LEFT, Side.RIGHT)
        )
        if any(catalogue is None for catalogue in catalogues):
            return None
        left_catalogue, right_catalogue = catalogues
        assert left_catalogue is not None and right_catalogue is not None
        if left_catalogue.default_option_id != right_catalogue.default_option_id:
            raise ValueError(
                "Axle corners declare incompatible suspension-hold defaults"
            )

        right_by_id = {option.id: option for option in right_catalogue.options}
        composed: list[SuspensionHoldOption] = []
        for left_option in left_catalogue.options:
            right_option = right_by_id.get(left_option.id)
            if right_option is None:
                continue
            if left_option.composition_signature != right_option.composition_signature:
                raise ValueError(
                    f"Axle corners assign incompatible semantics to suspension "
                    f"hold '{left_option.id}'"
                )
            held = tuple(
                coordinate.map_points(
                    lambda point, selected_side=side: side_qualified(
                        selected_side, point
                    ),
                    side=side,
                )
                for side, option in (
                    (Side.LEFT, left_option),
                    (Side.RIGHT, right_option),
                )
                for coordinate in option.hold.coordinates
            )
            unavailable_reasons = tuple(
                reason
                for reason in (
                    left_option.unavailable_reason,
                    right_option.unavailable_reason,
                )
                if reason
            )
            warnings = tuple(
                warning
                for warning in (left_option.warning, right_option.warning)
                if warning
            )
            availability = max(
                (left_option.availability, right_option.availability),
                key=(
                    SuspensionHoldAvailability.AVAILABLE,
                    SuspensionHoldAvailability.AVAILABLE_WITH_WARNING,
                    SuspensionHoldAvailability.UNAVAILABLE,
                ).index,
            )
            composed.append(
                SuspensionHoldOption(
                    id=left_option.id,
                    label=left_option.label,
                    description=left_option.description,
                    hold=CoordinateHold(held),
                    availability=availability,
                    warning=" ".join(dict.fromkeys(warnings)) or None,
                    unavailable_reason=(
                        " ".join(dict.fromkeys(unavailable_reasons)) or None
                    ),
                )
            )

        if not composed:
            return None
        return SuspensionHoldCatalogue(
            default_option_id=left_catalogue.default_option_id,
            options=tuple(composed),
        )

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
        # Corners construct their own flat-road contact centres before they are
        # composed; the closure overwrites both with the coupled shared-plane
        # solution so the design state carries one consistent ground.
        self.apply_ground_closure(state.positions)
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

    def output_only_points(self) -> tuple[PointKey, ...]:
        """Return each corner's undriveable outputs under side-qualified keys."""
        return tuple(
            dict.fromkeys(
                side_qualified(side, point)
                for side in (Side.LEFT, Side.RIGHT)
                for point in self.corners[side].output_only_points()
            )
        )

    def closure_points(self) -> tuple[PointKey, ...]:
        """Return the coupled contact centres when ground closure owns them."""
        if self._ground_closure_plan() is None:
            return ()
        return (
            PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE),
            PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE),
        )

    @cached_property
    def design_road_plane(self) -> RoadPlane | None:
        """Return the authored level road plane in chassis coordinates.

        ISO 8855 vehicle and earth-fixed axes are assumed aligned at design.
        A materially banked axle contact line violates that contract and is
        rejected rather than reinterpreted as a rolled design condition.
        """
        state = self.initial_state()
        left = state.get(PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE))
        right = state.get(PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE))
        try:
            road = RoadPlane.from_axle_contact_centres(left, right)
        except ValueError:
            return None
        if not road.normal.almost_equals(
            Direction3((0.0, 0.0, 1.0)),
            atol=1e-7,
            rtol=0.0,
        ):
            return None
        return road

    def constraints(self) -> list[Constraint]:
        """Combine remapped corner constraints and the rigid rack coupling."""
        constraints = [
            constraint.remap(lambda point, side=side: side_qualified(side, point))
            for side, corner in self.corners.items()
            for constraint in corner.constraints()
        ]
        rack = self.rack_attachment_points()
        if rack is not None:
            left_point, right_point = rack
            left = self.corners[Side.LEFT].initial_state().positions[left_point]
            right = self.corners[Side.RIGHT].initial_state().positions[right_point]
            # The rigid rack keeps the two attachment points a fixed distance
            # apart; each corner constrains its own point to the rack axis.
            constraints.append(
                DistanceConstraint(
                    PointRef(Side.LEFT, left_point),
                    PointRef(Side.RIGHT, right_point),
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

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Compose state metric metadata from installed axle mechanisms."""
        return (
            *self.anti_roll.topology_metric_specs(),
            *self.heave_link.topology_metric_specs(),
        )

    def topology_metric_rows(self, state: SuspensionState) -> AxleMetricRows:
        """Compose typed axle and per-corner mechanism metric rows."""
        result = AxleMetricRows(axle=OrderedDict(), corners={})
        for mechanism_rows in (
            self.anti_roll.topology_metric_values(self, state),
            self.heave_link.topology_metric_values(state),
        ):
            result.axle.update(mechanism_rows.axle)
            for side, row in mechanism_rows.corners.items():
                result.corners.setdefault(side, OrderedDict()).update(row)
        return result

    def topology_diagnostics(
        self,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        """Return corner-owned diagnostics followed by shared axle checks."""
        issues: list[DiagnosticIssue] = []
        for side in (Side.LEFT, Side.RIGHT):
            corner_states = [self.corner_state(state, side) for state in states]
            issues.extend(self.corners[side].topology_diagnostics(corner_states))
        issues.extend(self.anti_roll.topology_diagnostics(self, states))
        return issues

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
        anti_roll_spec = self.anti_roll.derived_spec()
        functions.update(anti_roll_spec.functions)
        dependencies.update(anti_roll_spec.dependencies)
        if self._ground_closure_plan() is not None:
            # The coupled contact centres are post-solve closure outputs, not derived
            # points: dropping the composed flat-ground entries means nothing
            # can silently write per-corner flat-road centres into an axle state.
            for side in (Side.LEFT, Side.RIGHT):
                contact_centre = PointRef(side, PointID.WHEEL_CONTACT_CENTRE)
                functions.pop(contact_centre, None)
                dependencies.pop(contact_centre, None)
        return DerivedPointsSpec(functions, dependencies)

    def _ground_closure_plan(self) -> dict[str, Any] | None:
        """Return the coupled tangency solve arguments, or None when inapplicable.

        The closure needs both corners' wheel radii and spin axes. Corners
        without a wheel configuration have no radius to couple, and a custom
        corner that solves its own authored contact centre as a free point owns
        that point outright; both cases leave contact-centre ownership with the
        corners.
        """
        left_corner = self.corners[Side.LEFT]
        right_corner = self.corners[Side.RIGHT]
        if left_corner.config is None or right_corner.config is None:
            return None
        if (
            PointID.WHEEL_CONTACT_CENTRE in left_corner.free_points()
            or PointID.WHEEL_CONTACT_CENTRE in right_corner.free_points()
        ):
            return None

        left_axis_inboard, left_axis_outboard = left_corner.wheel_axis_points()
        right_axis_inboard, right_axis_outboard = right_corner.wheel_axis_points()
        return {
            "left_center": PointRef(Side.LEFT, PointID.WHEEL_CENTER),
            "left_axis_inboard": PointRef(Side.LEFT, left_axis_inboard),
            "left_axis_outboard": PointRef(Side.LEFT, left_axis_outboard),
            "left_radius": left_corner.config.wheel.tire.nominal_radius,
            "right_center": PointRef(Side.RIGHT, PointID.WHEEL_CENTER),
            "right_axis_inboard": PointRef(Side.RIGHT, right_axis_inboard),
            "right_axis_outboard": PointRef(Side.RIGHT, right_axis_outboard),
            "right_radius": right_corner.config.wheel.tire.nominal_radius,
        }

    def apply_ground_closure(
        self,
        positions: dict[PointKey, Any],
        seed: float | None = None,
    ) -> float | None:
        """Overwrite both wheel contact centres with the coupled solution.

        Runs once per accepted state, after solving. With no explicit ``seed``,
        the angle implied by the contact-centre values already stored in
        ``positions``
        is recovered as the seed, so branch continuity needs no hidden state.
        Returns the solved primal ground-normal angle, or ``None`` when the
        corners own their contact centres.
        """
        plan = self._ground_closure_plan()
        if plan is None:
            return None
        left_contact_centre = PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE)
        right_contact_centre = PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE)
        if seed is None:
            stored_left = positions.get(left_contact_centre)
            stored_right = positions.get(right_contact_centre)
            if stored_left is not None and stored_right is not None:
                seed = seed_from_contact_centres(stored_left, stored_right)
        contact_centres = solve_axle_wheel_contact_centres(positions, **plan, seed=seed)
        positions[left_contact_centre] = contact_centres.left
        positions[right_contact_centre] = contact_centres.right
        return contact_centres.normal_angle

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
        """Resolve shared center points or require a side for corner points."""
        side_policy = sweep_target_side_policy(TargetKind.POINT, point.name.lower())
        candidate_sides = (
            (None,) if side_policy == "shared" else (Side.LEFT, Side.RIGHT)
        )
        selected_side = resolve_published_target_side(
            f"Axle sweep target for '{point.name}'",
            candidate_sides,
            side,
        )
        if selected_side is None:
            return PointRef(Side.CENTER, point)
        return PointRef(selected_side, point)

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
        steering_response_axes: "Sequence[UprightScrewAxisResult] | None" = None,
    ) -> "AxleMetricRows":
        """Compute structural corner and axle-level metric rows."""
        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_axle_state(
            state,
            self,
            self.config,
            tangents,
            steering_response_axes,
        )

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return side-qualified corner elements and shared axle hardware."""
        elements = tuple(
            map_element_points(
                element,
                lambda point, side=side: side_qualified(side, point),
                label=f"{side.name.title()} {element.label}",
            )
            for side, corner in self.corners.items()
            for element in corner.elements()
        )
        rack = self.rack_attachment_points()
        rack_elements: tuple[SuspensionElement, ...] = ()
        if rack is not None:
            rack_elements = (
                RackElement(
                    label="Steering Rack",
                    left_inner=PointRef(Side.LEFT, rack[0]),
                    right_inner=PointRef(Side.RIGHT, rack[1]),
                    translation_axis=Axis.Y,
                ),
            )
        return (
            *elements,
            *rack_elements,
            *self.anti_roll.elements(self),
            *self.heave_link.elements(),
        )
