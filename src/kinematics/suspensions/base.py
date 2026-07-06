"""
Base class for suspension types.

This module defines the abstract Suspension class that combines topology definition
(required points, shim support) with behavior implementation (constraints,
visualization) in a single unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from kinematics.constraints import Constraint
from kinematics.core.enums import PointID, ShimType, Units
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey, Side
from kinematics.points.derived.manager import DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.config.settings import SuspensionConfig

if TYPE_CHECKING:
    from kinematics.metrics.main import MetricRow
    from kinematics.sensitivity import TangentField
    from kinematics.visualization.main import LinkVisualization, WheelAnchors


@dataclass
class Suspension(ABC):
    """
    Base class for all suspension types.

    Subclasses define:
    - Class-level attributes for topology (required/optional points, shim support)
    - Instance-level storage for geometry and configuration
    - Methods for constraints, visualization, and kinematic behavior

    This class implements the provider interface directly - no separate provider needed.
    """

    TYPE_KEY: ClassVar[str] = ""
    ALIASES: ClassVar[frozenset[str]] = frozenset()
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = ()
    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset()

    name: str = "unnamed"
    version: str = "0.0.0"
    units: Units = Units.MILLIMETERS
    # Keyed by PointKey so the same field can hold single-corner PointID keys or
    # (for the axle model) side-qualified PointRef keys. Runtime is unchanged.
    hardpoints: dict[PointKey, Point3] = field(default_factory=dict)
    config: SuspensionConfig | None = None

    # Internal state cache.
    _initial_state: SuspensionState | None = field(default=None, repr=False)

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

    @classmethod
    def from_yaml_data(cls, yaml_data: dict[str, Any]) -> "Suspension":
        """
        Build a suspension instance from parsed YAML data (type key removed).

        The base implementation is the single-corner loader: it parses a flat
        hardpoints block and a single config into ``cls``. Multi-corner models
        (e.g. the axle) override this to assemble their sub-models from a
        side-structured schema.

        Args:
            yaml_data: Parsed YAML mapping with the ``type`` key already removed.

        Returns:
            An instantiated suspension of type ``cls``.
        """
        # Deferred import: geometry_loader imports this module.
        from kinematics.io.geometry_loader import load_suspension

        return load_suspension(yaml_data, cls)

    @abstractmethod
    def initial_state(self) -> SuspensionState:
        """
        Build the initial suspension state from hardpoints.

        Returns:
            SuspensionState with all point positions and free points set.
        """
        ...

    @abstractmethod
    def free_points(self) -> Sequence[PointID]:
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
    def get_visualization_links(self) -> list[LinkVisualization]:
        """
        Get visualization links for rendering the suspension.

        Returns:
            List of link definitions for visualisation.
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

    def output_points(self) -> tuple[PointKey, ...]:
        """
        Point keys to write to the solver output, in column order.

        The base implementation returns the static :attr:`OUTPUT_POINTS`.
        Subclasses override this to append points that are only present when an
        optional feature is configured (e.g. the pushrod/rocker group, or the
        axle's per-side ARB droplink), so that output columns match the geometry
        actually loaded.
        """
        return self.OUTPUT_POINTS

    @property
    def has_rocker(self) -> bool:
        """
        Whether this suspension has an inboard pushrod/rocker group.

        The base implementation is ``False``. Corner models that support the
        pushrod/rocker group override this, and metric computation keys off it to
        emit rocker/torsion-bar angle columns.
        """
        return False

    @property
    def has_strut(self) -> bool:
        """
        Whether this suspension has an optional spring/damper (coilover) element.

        The base implementation is ``False``. Corner models that support the
        strut group override this.
        """
        return False

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "MetricRow":
        """
        Compute the export metric row for a single solved state.

        The base implementation computes the corner-level metric catalog. The
        axle model overrides this to emit per-side and axle-level metrics. This
        method is the CLI's single, branch-free metrics entry point.

        Args:
            state: The solved state to analyze.
            tangents: Optional solution-manifold tangents for this state
                (from kinematics.sensitivity). When given, derivative
                metrics (motion ratios, camber gain, ...) are appended.

        Returns:
            An ordered mapping of metric column names to values.

        Raises:
            ValueError: If the suspension has no configuration.
        """
        # Deferred import: metrics imports suspension types.
        from kinematics.metrics.main import compute_metrics_for_state

        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_state(state, self, self.config, tangents)

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """
        Resolve a sweep target's (point, side) pair into a concrete point key.

        Single-corner models key on plain ``PointID`` and reject a ``side``.
        The axle model requires a side and returns a ``PointRef``.

        Args:
            point: The point the sweep target drives.
            side: The side qualifier from the sweep spec, or ``None``.

        Returns:
            The concrete point key used in this model's state.

        Raises:
            ValueError: If a side is given for a single-corner model.
        """
        if side is not None:
            raise ValueError(
                f"Sweep target for '{point.name}' specifies side "
                f"'{side.name.lower()}', but suspension type '{self.TYPE_KEY}' "
                "is a single corner and does not accept a side."
            )
        return point

    def wheel_visualization_anchors(self) -> "list[WheelAnchors]":
        """
        Point keys anchoring each drawn wheel for visualization.

        Single-corner models draw one wheel from the corner's derived wheel and
        axle points. The axle model overrides this to return one anchor set per
        side so both wheels are rendered.

        Returns:
            A list of wheel anchor descriptors (one per drawn wheel).
        """
        from kinematics.visualization.main import WheelAnchors

        return [
            WheelAnchors(
                center=PointID.WHEEL_CENTER,
                inboard=PointID.WHEEL_INBOARD,
                outboard=PointID.WHEEL_OUTBOARD,
                axle_inboard=PointID.AXLE_INBOARD,
                axle_outboard=PointID.AXLE_OUTBOARD,
            )
        ]
