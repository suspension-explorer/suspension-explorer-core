"""
Base class for suspension types.

This module defines the abstract Suspension class that combines assembly definition
(required points, shim support) with behavior implementation (constraints and
physical elements) in a single unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.constraints import Constraint
from kinematics.core.elements import SuspensionElement
from kinematics.core.enums import (
    ElementLengthCoordinateID,
    PointID,
    Scope,
    ShimType,
    SuspensionType,
    Units,
)
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, PointRef, Side
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.diagnostics import DiagnosticIssue
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.main import AxleMetricRows, MetricRow
    from kinematics.core.metrics.registry import MetricSpec
    from kinematics.core.rigid_motion import UprightScrewAxisResult
    from kinematics.core.sensitivity import TangentField
    from kinematics.core.steering_response import (
        SteeringProbeCatalogue,
        SteeringResponseDefinition,
    )
    from kinematics.core.targeting import (
        ActuatorDOF,
        DriveCoordinate,
        ScalarTarget,
        TargetKind,
    )


@dataclass
class Suspension(ABC):
    """
    Base class for all suspension types.

    Subclasses define:
    - Architecture and mechanism-specific point declarations
    - Instance-level storage for geometry and configuration
    - Methods for constraints, physical elements, and kinematic behavior

    This class implements the provider interface directly - no separate provider needed.
    """

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OUTPUT_POINTS: ClassVar[tuple[PointKey, ...]] = ()
    OUTPUT_ONLY_POINTS: ClassVar[tuple[PointKey, ...]] = ()
    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset()

    name: str = "unnamed"
    version: str = "0.0.0"
    units: Units = Units.MILLIMETERS
    hardpoints: dict[PointKey, Point3] = field(default_factory=dict)
    config: SuspensionConfig | None = None
    side: Side = field(kw_only=True)

    # Internal state cache.
    _initial_state: SuspensionState | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _assembly_cache: SuspensionAssembly | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate instance after creation."""
        self.validate_hardpoints()

    def required_points(self) -> frozenset[PointID]:
        """Return authored points required by this suspension instance."""
        return self.REQUIRED_POINTS

    def optional_points(self) -> frozenset[PointID]:
        """Return additional authored points accepted by this suspension instance."""
        return self.OPTIONAL_POINTS

    def all_valid_points(self) -> frozenset[PointID]:
        """Return every authored point accepted by this suspension instance."""
        return self.required_points() | self.optional_points()

    @abstractmethod
    def reported_type_key(self) -> SuspensionType:
        """Return the public geometry type identity exported with results."""
        ...

    @abstractmethod
    def initial_state(self) -> SuspensionState:
        """
        Build the initial suspension state from hardpoints.

        Returns:
            SuspensionState with all point positions and free points set.
        """
        ...

    @abstractmethod
    def free_points(self) -> Sequence[PointKey]:
        """
        Get the points that can move during solving.

        Returns:
            Sequence of PointIDs that are free to move.
        """
        ...

    @abstractmethod
    def constraints(self) -> list[Constraint]:
        """
        Build geometric constraints for this suspension.

        Returns:
            List of constraints that must be satisfied during solving.
        """
        ...

    @abstractmethod
    def derived_spec(self) -> DerivedPointsSpec:
        """
        Get the specification for computing derived points.

        Returns:
            Specification defining how derived points are calculated.
        """
        ...

    @abstractmethod
    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """
        Compute the side view instant center.

        Args:
            state: Current suspension state.

        Returns:
            SVIC coordinates, or None if not applicable.
        """
        ...

    @abstractmethod
    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """
        Compute the front view instant center.

        Args:
            state: Current suspension state.

        Returns:
            FVIC coordinates, or None if not applicable.
        """
        ...

    @abstractmethod
    def elements(self) -> tuple[SuspensionElement, ...]:
        """
        Return the physical elements composing this suspension.

        Returns:
            Physical suspension elements referencing points in the solver state.
        """
        ...

    def validate_hardpoints(self) -> None:
        """Validate the exact authored point set for this suspension instance."""
        present = set(self.hardpoints.keys())
        missing = self.required_points() - present
        if missing:
            missing_names = sorted(p.name for p in missing)
            raise ValueError(f"Missing required hardpoints: {', '.join(missing_names)}")

        unknown = present - self.all_valid_points()
        if unknown:
            unknown_names = sorted(point.name for point in unknown)
            raise ValueError(f"Invalid hardpoints: {', '.join(unknown_names)}")

    def get_hardpoints_copy(self) -> dict[PointKey, Point3]:
        """
        Return a mutable copy of the hardpoints dictionary.

        Each Point3 is copied so callers can modify positions without
        affecting the stored design values.
        """
        return {pid: pos.copy() for pid, pos in self.hardpoints.items()}

    def damper_points(self) -> tuple[PointKey, PointKey] | None:
        """Return installed spring/damper endpoints, if present."""
        return None

    def drive_coordinates(self) -> "tuple[DriveCoordinate, ...]":
        """Return explicitly driveable scalar coordinates in stable order."""
        from kinematics.core.targeting import DriveCoordinate, TargetKind

        endpoints = self.damper_points()
        if endpoints is None:
            return ()
        coordinate_id = ElementLengthCoordinateID.DAMPER
        return (
            DriveCoordinate(
                id=coordinate_id,
                kind=TargetKind.ELEMENT_LENGTH,
                label=coordinate_id.label,
                unit=coordinate_id.unit,
                point_keys=endpoints,
                scope=Scope.CORNER,
                side=Side.LEFT,
            ),
        )

    def resolve_drive_coordinate(
        self,
        coordinate_id: str,
        side: Side | None,
        kind: "TargetKind | None" = None,
    ) -> "DriveCoordinate":
        """Resolve one stable drive-coordinate ID under topology side rules."""
        from kinematics.core.targeting import (
            TargetKind,
            resolve_published_target_side,
        )

        available = self.drive_coordinates()
        candidates = [
            item
            for item in available
            if item.id == coordinate_id and (kind is None or item.kind is kind)
        ]
        available_ids = sorted(
            {item.id for item in available if kind is None or item.kind is kind}
        )
        available_text = ", ".join(available_ids) if available_ids else "none"
        coordinate_kind = {
            TargetKind.ELEMENT_LENGTH: "element-length target",
            TargetKind.ACTUATOR_POSITION: "actuator-position target",
        }.get(kind, "drive coordinate")
        if not candidates:
            raise ValueError(
                f"Driveable {coordinate_kind} '{coordinate_id}' is unavailable for "
                f"suspension type '{self.reported_type_key()}'. Available "
                f"driveable {coordinate_kind} IDs: {available_text}."
            )

        selected_side = resolve_published_target_side(
            f"{coordinate_kind.capitalize()} '{coordinate_id}'",
            tuple(item.side for item in candidates),
            side,
        )
        return next(item for item in candidates if item.side is selected_side)

    @property
    def is_axle(self) -> bool:
        """Whether this topology composes multiple corner suspensions."""
        return False

    @abstractmethod
    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
        steering_response_axes: "Sequence[UprightScrewAxisResult] | None" = None,
    ) -> "MetricRow | AxleMetricRows":
        """Compute metric output for one solved state."""
        ...

    def derivative_metric_definitions(
        self,
    ) -> "tuple[DerivativeMetricDefinition, ...]":
        """Topology-specific declarative derivative metrics."""
        return ()

    def topology_metric_values(self, state: SuspensionState) -> "MetricRow":
        """Return non-derivative metrics owned by this topology."""
        return OrderedDict()

    def topology_metric_specs(self) -> "tuple[MetricSpec, ...]":
        """Return state metric metadata owned by this topology."""
        return ()

    def topology_diagnostics(
        self,
        states: "list[SuspensionState]",
    ) -> "list[DiagnosticIssue]":
        """Return advisory checks owned by this concrete topology."""
        return []

    def output_points(self) -> tuple[PointKey, ...]:
        """Return the points exported for a solved state."""
        return self.OUTPUT_POINTS

    def output_only_points(self) -> tuple[PointKey, ...]:
        """
        Return exported solved points that cannot be driven as sweep targets.

        The designation is a solver and product policy for observables that are
        not supported as actuators, such as branch-sensitive outputs from a
        coupled construction; it does not imply that their forward value or
        Jacobian is unavailable. An output-only point is computed either as an
        ordinary derived point or by the post-solve ground closure.
        """
        return self.OUTPUT_ONLY_POINTS

    def apply_ground_closure(
        self,
        positions: "dict[PointKey, Any]",
        seed: float | None = None,
    ) -> float | None:
        """
        Write post-solve ground outputs into ``positions``; return the solved seed.

        The base suspension has no coupled ground geometry, so this is a no-op.
        Axles overwrite both wheel contact centres with the coupled shared-plane
        solution and return the solved ground-normal angle so a sweep can thread
        it into the next state's solve as an explicit, stateless seed.
        """
        return None

    def validate_sweep_target_points(
        self,
        point_keys: Iterable[PointKey],
    ) -> None:
        """Require every sweep target to be present, movable, and supported."""
        point_catalog = self.assembly().points
        suspension_type = self.reported_type_key()
        for point_key in dict.fromkeys(point_keys):
            if point_key not in point_catalog.all:
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is not present in "
                    f"suspension type '{suspension_type}'."
                )
            if point_key in point_catalog.output_only:
                point_id = (
                    point_key.point if isinstance(point_key, PointRef) else point_key
                )
                guidance = point_id.output_only_target_guidance
                guidance_suffix = f" {guidance}" if guidance is not None else ""
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is a derived output of "
                    f"suspension type '{suspension_type}' and cannot be driven."
                    f"{guidance_suffix}"
                )
            if point_key in point_catalog.fixed:
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is fixed in suspension "
                    f"type '{suspension_type}'."
                )

    def validate_sweep_targets(self, targets: Iterable["ScalarTarget"]) -> None:
        """Validate driven points and topology-declared scalar coordinates."""
        target_list = tuple(targets)
        self.validate_sweep_target_points(
            point for target in target_list for point in target.driven_points
        )
        available = self.drive_coordinates()
        for target in target_list:
            target_key = target.drive_coordinate_key
            if target_key is None:
                continue
            target_kind, target_id, target_points, target_side = target_key
            if not any(
                coordinate.kind is target_kind
                and coordinate.id == target_id
                and (target_points is None or coordinate.point_keys == target_points)
                and coordinate.side is target_side
                for coordinate in available
            ):
                available_ids = (
                    ", ".join(
                        sorted(
                            {
                                coordinate.id
                                for coordinate in available
                                if coordinate.kind is target_kind
                            }
                        )
                    )
                    or "none"
                )
                raise ValueError(
                    f"{target.coordinate_description.capitalize()} is not declared "
                    f"driveable for this suspension. Available driveable "
                    f"{target_kind.value} IDs: {available_ids}."
                )

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Resolve a sweep target for a single-corner suspension."""
        from kinematics.core.targeting import (
            TargetKind,
            resolve_published_target_side,
            sweep_target_side_policy,
        )

        side_policy = sweep_target_side_policy(TargetKind.POINT, point.name.lower())
        candidate_sides = (None,) if side_policy == "shared" else (Side.LEFT,)
        resolve_published_target_side(
            f"Sweep target for '{point.name}'",
            candidate_sides,
            side,
        )
        return point

    def actuator_dofs(self) -> "tuple[ActuatorDOF, ...]":
        """Return physical actuator coordinates that every sweep must control."""
        return ()

    def steering_actuator_dof(self) -> "ActuatorDOF | None":
        """Return the steering actuator coordinate, if this suspension has one."""
        return None

    def steering_probe_catalogue(self) -> "SteeringProbeCatalogue | None":
        """Return topology-owned steering isolation choices, if implemented."""
        return None

    def resolve_steering_probe(
        self,
        requested_option_id: str | None = None,
    ) -> "SteeringResponseDefinition | None":
        """Resolve one explicit option or this topology's canonical default."""
        from kinematics.core.steering_response import (
            SteeringProbeOptionAvailability,
            SteeringProbeSelectionSource,
            SteeringResponseDefinition,
        )

        steering = self.steering_actuator_dof()
        catalogue = self.steering_probe_catalogue()
        if steering is None or catalogue is None:
            if requested_option_id not in (None, "layout_default"):
                raise ValueError(
                    f"Suspension type '{self.reported_type_key().value}' does not "
                    "publish steering-probe isolation options."
                )
            return None
        uses_default = requested_option_id in (None, "layout_default")
        option_id = catalogue.default_option_id if uses_default else requested_option_id
        assert option_id is not None
        option = catalogue.option(option_id)
        if option.availability is SteeringProbeOptionAvailability.UNAVAILABLE:
            reason = option.unavailable_reason or "The option is unavailable."
            raise ValueError(
                f"Steering-probe isolation '{option.id}' is unavailable: {reason}"
            )
        return SteeringResponseDefinition(
            steering_actuator=steering,
            held_coordinates=option.held_coordinates,
            owner=self.reported_type_key().value,
            definition_id=option.id,
            requested_option_id=requested_option_id,
            selection_source=(
                SteeringProbeSelectionSource.LAYOUT_DEFAULT
                if uses_default
                else SteeringProbeSelectionSource.USER_OVERRIDE
            ),
            option_class=option.option_class,
            label=option.label,
            description=option.description,
            warning=option.warning,
        )

    def closure_points(self) -> tuple[PointKey, ...]:
        """
        Return points written by the post-solve closure rather than the graph.

        These classify as derived in the point catalog: they are computed from
        the state on every solve, independent of whether targeting policy also
        marks them output-only. The base suspension has none.
        """
        return ()

    def assembly(self) -> SuspensionAssembly:
        """Return the validated point and element composition."""
        if self._assembly_cache is None:
            self._assembly_cache = SuspensionAssembly.from_state(
                self.initial_state(),
                self.derived_spec(),
                self.elements(),
                self.output_points(),
                self.output_only_points(),
                self.closure_points(),
            )
        return self._assembly_cache
