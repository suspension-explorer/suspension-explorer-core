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
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.constraints import Constraint
from kinematics.core.elements import SuspensionElement
from kinematics.core.metrics.main import (
    AxleMetricRows,
    MetricRow,
    compute_metrics_for_state,
)
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.enums import PointID, ShimType, Units
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey, Side
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.diagnostics import DiagnosticIssue
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.sensitivity import TangentField


@dataclass
class Suspension(ABC):
    """
    Base class for all suspension types.

    Subclasses define:
    - Class-level attributes for geometry (required/optional points, shim support)
    - Instance-level storage for geometry and configuration
    - Methods for constraints, physical elements, and kinematic behavior

    This class implements the provider interface directly - no separate provider needed.
    """

    TYPE_KEY: ClassVar[str] = ""
    ALIASES: ClassVar[frozenset[str]] = frozenset()
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OUTPUT_POINTS: ClassVar[tuple[PointKey, ...]] = ()
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

    @classmethod
    def all_valid_points(cls) -> frozenset[PointID]:
        """All points valid for this suspension type."""
        return cls.REQUIRED_POINTS | cls.OPTIONAL_POINTS

    @classmethod
    def matches_type(cls, type_key: str) -> bool:
        """Check if this class handles the given type key."""
        key_lower = type_key.lower()
        return key_lower == cls.TYPE_KEY or key_lower in cls.ALIASES

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
        """Validate that required hardpoints are present."""
        present = set(self.hardpoints.keys())
        missing = self.REQUIRED_POINTS - present
        if missing:
            missing_names = sorted(p.name for p in missing)
            raise ValueError(f"Missing required hardpoints: {', '.join(missing_names)}")

    def get_hardpoints_copy(self) -> dict[PointKey, Point3]:
        """
        Return a mutable copy of the hardpoints dictionary.

        Each Point3 is copied so callers can modify positions without
        affecting the stored design values.
        """
        return {pid: pos.copy() for pid, pos in self.hardpoints.items()}

    @property
    def has_strut(self) -> bool:
        """Whether this explicit topology includes a spring/damper element."""
        return False

    @property
    def is_axle(self) -> bool:
        """Whether this topology composes multiple corner suspensions."""
        return False

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "MetricRow | AxleMetricRows":
        """Compute one metric row, including derivatives when tangents exist."""
        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_state(state, self, self.config, tangents)

    def derivative_metric_definitions(
        self,
    ) -> "tuple[DerivativeMetricDefinition, ...]":
        """Topology-specific declarative derivative metrics."""
        return ()

    def topology_metric_values(self, state: SuspensionState) -> "MetricRow":
        """Return non-derivative metrics owned by this topology."""
        return OrderedDict()

    def topology_diagnostics(
        self,
        states: "list[SuspensionState]",
    ) -> "list[DiagnosticIssue]":
        """Return advisory checks owned by this concrete topology."""
        return []

    def output_points(self) -> tuple[PointKey, ...]:
        """Return the points exported for a solved state."""
        return self.OUTPUT_POINTS

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Resolve a sweep target for a single-corner suspension."""
        if side is not None:
            raise ValueError(
                f"Sweep target for '{point.name}' specifies side "
                f"'{side.name.lower()}', but suspension type '{self.TYPE_KEY}' "
                "is a single corner and does not accept a side."
            )
        return point

    def assembly(self) -> SuspensionAssembly:
        """Return the validated point and element composition."""
        return SuspensionAssembly.from_state(
            self.initial_state(),
            self.derived_spec(),
            self.elements(),
            self.output_points(),
        )
