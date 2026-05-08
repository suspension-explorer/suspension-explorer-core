"""
Base class for suspension types.

This module defines the abstract Suspension class that combines topology definition
(required points, shim support) with behavior implementation (constraints,
visualization) in a single unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.constraints import Constraint
from kinematics.core.enums import PointID, ShimType, Units
from kinematics.core.geometry import Point3
from kinematics.points.derived.manager import DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.config.settings import SuspensionConfig

if TYPE_CHECKING:
    from kinematics.visualization.main import LinkVisualization


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
    hardpoints: dict[PointID, Point3] = field(default_factory=dict)
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

    def get_hardpoints_copy(self) -> dict[PointID, Point3]:
        """
        Return a mutable copy of the hardpoints dictionary.

        Each Point3 is copied so callers can modify positions without
        affecting the stored design values.
        """
        return {pid: pos.copy() for pid, pos in self.hardpoints.items()}
