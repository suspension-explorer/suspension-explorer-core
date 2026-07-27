"""
Validated composition of suspension points and physical elements.
"""

from dataclasses import dataclass

from kinematics.core.elements import (
    SuspensionElement,
    WheelElement,
)
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.state import SuspensionState


@dataclass(frozen=True)
class PointCatalog:
    """
    Identifier-only classification of points in a suspension assembly.

    The fixed, free, and derived sets are mutually exclusive and partition
    every point in the assembly. ``fixed`` means authored geometry that never
    moves; ``derived`` means computed from the state — whether by the
    derived-point graph or by a post-solve closure (an axle's coupled
    wheel-ground tangents). ``output_only`` is not a fourth class: it is an
    overlay marking the subset of derived points that are reported but cannot
    be driven under the solver's actuator policy.
    """

    fixed: frozenset[PointKey]
    free: frozenset[PointKey]
    derived: frozenset[PointKey]
    output_only: frozenset[PointKey] = frozenset()

    def __post_init__(self) -> None:
        """
        Require mutually exclusive point classifications.
        """
        overlaps = (
            (self.fixed & self.free)
            | (self.fixed & self.derived)
            | (self.free & self.derived)
        )
        if overlaps:
            raise ValueError(f"Point classifications overlap: {sorted(overlaps)!r}")

        if not self.output_only <= self.derived:
            invalid = sorted(self.output_only - self.derived)
            raise ValueError(f"Output-only points must be derived points: {invalid!r}")

    @property
    def all(self) -> frozenset[PointKey]:
        """
        Return every point in the catalog.
        """
        return self.fixed | self.free | self.derived

    @classmethod
    def from_state(
        cls,
        state: SuspensionState,
        derived_spec: DerivedPointsSpec,
        output_only_points: tuple[PointKey, ...] = (),
        closure_points: tuple[PointKey, ...] = (),
    ) -> "PointCatalog":
        """
        Classify point identifiers without copying solver positions.

        How a point is computed and whether it may be driven are separate
        facts: ``closure_points`` declares points the post-solve closure
        writes, and they classify as derived because they are computed from
        the state each solve — publishing them as fixed would misstate
        geometry that moves with every state. ``output_only_points`` is pure
        targeting policy and never changes classification.
        """
        state_points = frozenset[PointKey](state.positions)
        derived = frozenset[PointKey](derived_spec.functions) | frozenset(
            closure_points
        )
        free = frozenset[PointKey](state.free_points)
        base = state_points - derived

        if not free <= base:
            invalid = sorted(free - base)
            raise ValueError(
                f"Free points must be non-derived state points: {invalid!r}"
            )

        catalog = cls(
            fixed=base - free,
            free=free,
            derived=derived,
            output_only=frozenset(output_only_points),
        )
        if state_points != catalog.all:
            raise ValueError("Initial-state points do not match the point catalog")

        missing_dependencies = {
            dependency
            for dependencies in derived_spec.dependencies.values()
            for dependency in dependencies
            if dependency not in catalog.all
        }
        if missing_dependencies:
            raise ValueError(
                "Derived-point dependencies are absent from the point catalog: "
                f"{sorted(missing_dependencies)!r}"
            )
        return catalog


@dataclass(frozen=True)
class SuspensionAssembly:
    """
    Complete physical composition of one suspension model.
    """

    points: PointCatalog
    elements: tuple[SuspensionElement, ...]
    output_points: tuple[PointKey, ...]

    def __post_init__(self) -> None:
        """
        Validate that every exported and element point exists.
        """
        element_points = {
            point for element in self.elements for point in element.point_keys
        }
        missing_element_points = element_points - self.points.all
        if missing_element_points:
            raise ValueError(
                "Assembly elements reference unknown points: "
                f"{sorted(missing_element_points)!r}"
            )

        missing_output_points = set(self.output_points) - self.points.all
        if missing_output_points:
            raise ValueError(
                "Assembly output references unknown points: "
                f"{sorted(missing_output_points)!r}"
            )

    @property
    def referenced_point_keys(self) -> tuple[PointKey, ...]:
        """
        Return output and element point keys in stable declaration order.
        """
        ordered = list(self.output_points)
        seen = set(ordered)
        for element in self.elements:
            for point in element.point_keys:
                if point not in seen:
                    ordered.append(point)
                    seen.add(point)
        return tuple(ordered)

    @property
    def wheels(self) -> tuple[WheelElement, ...]:
        """
        Return every wheel in assembly declaration order.
        """
        return tuple(
            element for element in self.elements if isinstance(element, WheelElement)
        )

    @classmethod
    def from_state(
        cls,
        state: SuspensionState,
        derived_spec: DerivedPointsSpec,
        elements: tuple[SuspensionElement, ...],
        output_points: tuple[PointKey, ...],
        output_only_points: tuple[PointKey, ...] = (),
        closure_points: tuple[PointKey, ...] = (),
    ) -> "SuspensionAssembly":
        """
        Build and validate an assembly from existing solver declarations.
        """
        return cls(
            points=PointCatalog.from_state(
                state, derived_spec, output_only_points, closure_points
            ),
            elements=elements,
            output_points=output_points,
        )
