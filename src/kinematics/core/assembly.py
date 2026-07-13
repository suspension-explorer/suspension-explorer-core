"""
Validated composition of suspension points and physical elements.
"""

from dataclasses import dataclass

from kinematics.core.elements import (
    ElementPath,
    ElementType,
    RockerElement,
    SuspensionElement,
    TorsionElement,
    WheelElement,
    element_paths,
)
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.state import SuspensionState


@dataclass(frozen=True)
class PointCatalog:
    """
    Identifier-only classification of points in a suspension assembly.
    """

    fixed: frozenset[PointKey]
    free: frozenset[PointKey]
    derived: frozenset[PointKey]

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
    ) -> "PointCatalog":
        """
        Classify point identifiers without copying solver positions.
        """
        state_points = frozenset[PointKey](state.positions)
        derived = frozenset[PointKey](derived_spec.functions)
        free = frozenset[PointKey](state.free_points)
        base = state_points - derived

        if not free <= base:
            invalid = sorted(free - base)
            raise ValueError(
                f"Free points must be non-derived state points: {invalid!r}"
            )

        catalog = cls(fixed=base - free, free=free, derived=derived)
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
    def element_paths(self) -> tuple[ElementPath, ...]:
        """
        Compose renderer-neutral paths for the complete assembly.

        A torsion bar and its rocker can share the same physical rotation axis.
        In that case the torsion bar owns the path so clients receive it once.
        """
        torsion_bar_paths = {
            torsion.path
            for torsion in self.elements
            if isinstance(torsion, TorsionElement)
            and torsion.type is ElementType.TORSION_BAR
            and len(torsion.path) == 2
        }
        torsion_bar_paths |= {(end, start) for start, end in torsion_bar_paths}

        paths: list[ElementPath] = []
        for element in self.elements:
            for path in element_paths(element):
                rocker_axis_owned_by_torsion_bar = (
                    isinstance(element, RockerElement)
                    and path.points == element.rotation_axis
                    and element.rotation_axis in torsion_bar_paths
                )
                if not rocker_axis_owned_by_torsion_bar:
                    paths.append(path)
        return tuple(paths)

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
    ) -> "SuspensionAssembly":
        """
        Build and validate an assembly from existing solver declarations.
        """
        return cls(
            points=PointCatalog.from_state(state, derived_spec),
            elements=elements,
            output_points=output_points,
        )
