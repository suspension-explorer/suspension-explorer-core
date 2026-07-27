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
from typing import TYPE_CHECKING, ClassVar, Iterable, Sequence

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.constraints import Constraint
from kinematics.core.elements import SuspensionElement
from kinematics.core.enums import PointID, ShimType, SuspensionType, Units
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.diagnostics import DiagnosticIssue
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.main import AxleMetricRows, MetricRow
    from kinematics.core.metrics.registry import MetricSpec
    from kinematics.core.sensitivity import TangentField
    from kinematics.core.targeting import ActuatorDOF


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

    @property
    def is_axle(self) -> bool:
        """Whether this topology composes multiple corner suspensions."""
        return False

    @abstractmethod
    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
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
        Return exported derived points that cannot be driven as sweep targets.

        These are always a subset of the derived points. The designation is a
        solver and product policy for observables that are not supported as
        actuators, such as branch-sensitive outputs from a coupled construction;
        it does not imply that their forward value or Jacobian is unavailable.
        """
        return self.OUTPUT_ONLY_POINTS

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
            if point_key in point_catalog.fixed:
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is fixed in suspension "
                    f"type '{suspension_type}'."
                )
            if point_key in point_catalog.output_only:
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is a derived output of "
                    f"suspension type '{suspension_type}' and cannot be driven. "
                    "Target 'wheel_center' along Z to control ride height, and "
                    "read ground geometry from the 'ground_z_centerline' metric."
                )

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Resolve a sweep target for a single-corner suspension."""
        if side is not None:
            raise ValueError(
                f"Sweep target for '{point.name}' specifies side "
                f"'{side.name.lower()}', but suspension type "
                f"'{self.reported_type_key()}' "
                "is a single corner and does not accept a side."
            )
        return point

    def actuator_dofs(self) -> "tuple[ActuatorDOF, ...]":
        """Return physical actuator coordinates that every sweep must control."""
        return ()

    def assembly(self) -> SuspensionAssembly:
        """Return the validated point and element composition."""
        return SuspensionAssembly.from_state(
            self.initial_state(),
            self.derived_spec(),
            self.elements(),
            self.output_points(),
            self.output_only_points(),
        )
