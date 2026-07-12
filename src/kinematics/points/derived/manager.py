"""
Derived point specifications and management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Container, Generic, Mapping, Set, TypeAlias, TypeVar, cast

import numpy as np

from kinematics.core.dual import DualVec3
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey

PositionValue: TypeAlias = Point3 | DualVec3

# Function signature for computing a derived point position.
PositionFn = Callable[[dict[PointKey, PositionValue]], PositionValue]

_V = TypeVar("_V", bound=PositionValue)

# A single spec keys homogeneously on one concrete key type: single-corner
# models use PointID, axle models use PointRef. Parametrizing over that type
# keeps the function and dependency dicts precise rather than widening to the
# invariant PointKey union, which would reject the narrow dicts callers build.
_K = TypeVar("_K", bound=PointKey)


@dataclass(frozen=True)
class DerivedPointsSpec(Generic[_K]):
    """
    Specification for derived point calculations.

    Contains the functions to compute derived points and their dependencies in a self-
    describing format that can be validated and sorted.
    """

    # The spec only reads these maps. Declaring functions as a Mapping keeps its
    # value type covariant, so a concrete function map whose values are subtypes
    # of PositionFn is accepted instead of requiring an exact PositionFn match.
    functions: Mapping[_K, PositionFn]
    dependencies: Mapping[_K, Set[_K]]

    def all_points(self) -> Set[_K]:
        """
        Get all derived point IDs defined in this spec.

        Returns:
            Set of PointID values representing all derived points that can be
            computed using this specification.
        """
        return set(self.functions.keys())

    def validate(self) -> None:
        """
        Ensures that every point with a calculation function also has its dependencies
        properly defined, and that no orphaned dependency entries exist without
        corresponding functions.

        Raises:
            ValueError: If the specification is inconsistent, such as when
                       functions exist without dependency definitions or
                       dependency definitions exist without functions.
        """
        # Check that all points in functions have dependencies defined.
        function_points = set(self.functions.keys())
        dependency_points = set(self.dependencies.keys())

        if function_points != dependency_points:
            missing_deps = function_points - dependency_points
            extra_deps = dependency_points - function_points

            msg_parts = []
            if missing_deps:
                msg_parts.append(f"Missing dependencies for: {missing_deps}")
            if extra_deps:
                msg_parts.append(f"Extra dependencies for: {extra_deps}")

            raise ValueError("; ".join(msg_parts))

    def __post_init__(self):
        """
        Validate the spec after initialization.
        """
        self.validate()


class DerivedPointsManager:
    """
    Manages the calculation of derived points by building and resolving a dependency
    graph to ensure the correct update order.
    """

    def __init__(self, spec: DerivedPointsSpec):
        self.spec = spec
        self.dependency_graph = spec.dependencies

        # This will raise an error if cycles are detected.
        self.update_order = self.get_topological_sort()

        # Cache of single-point computation plans, built lazily.
        self._computation_plans: dict[
            PointKey, tuple[tuple[PointKey, ...], tuple[PointKey, ...]]
        ] = {}

    def _dependency_path_contains_cycle(
        self,
        node: PointKey,
        visited: set[PointKey],
        recursion_stack: set[PointKey],
    ) -> bool:
        """
        Depth-first search utility for cycle detection in dependency graph.

        Uses two sets to track node states during DFS traversal:
        - visited: nodes that have been completely processed
        - recursion_stack: nodes in the current DFS path being explored

        A cycle is detected when we encounter a node that's already in the
        current recursion stack, indicating a back edge.

        Args:
            node: Current node being explored in the dependency graph.
            visited: Set of nodes that have been completely processed.
            recursion_stack: Set of nodes in the current DFS path.

        Returns:
            True if a cycle is detected starting from this node, False otherwise.
        """
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in self.dependency_graph.get(node, set()):
            if neighbor not in visited:
                if self._dependency_path_contains_cycle(
                    neighbor, visited, recursion_stack
                ):
                    return True
            elif neighbor in recursion_stack:
                return True  # Cycle detected.

        recursion_stack.remove(node)
        return False

    def get_topological_sort(self) -> list[PointKey]:
        """
        Performs a topological sort of the derived points to determine the correct
        calculation order.

        Raises:
            ValueError: If a circular dependency is detected in the graph.
        """
        visited: set[PointKey] = set()
        recursion_stack: set[PointKey] = set()
        nodes: set[PointKey] = set(self.dependency_graph)

        for node in nodes:
            if node not in visited:
                if self._dependency_path_contains_cycle(node, visited, recursion_stack):
                    raise ValueError(
                        "Circular dependency detected in derived point definitions."
                    )

        visited.clear()
        order = []

        def dfs(node: PointKey):
            if node in visited:
                return
            visited.add(node)

            # Recurse on dependencies that are also derived points.
            for dep in self.dependency_graph.get(node, set()):
                if dep in self.spec.functions:
                    dfs(dep)

            order.append(node)

        for point_id in self.spec.functions:
            if point_id not in visited:
                dfs(point_id)

        return order

    def update_in_place(self, positions: dict[PointKey, _V]) -> None:
        """
        Compute derived points and add them to positions dict in-place.

        Args:
            positions: Dictionary to mutate in-place by adding derived points.
                       Works with both Point3 (normal solve) and DualVec3 (autodiff).
        """
        for point_id in self.update_order:
            update_func = self.spec.functions[point_id]
            update_positions = cast(dict[PointKey, PositionValue], positions)
            positions[point_id] = cast(_V, update_func(update_positions))

    def _get_computation_plan(
        self, point_id: PointKey
    ) -> tuple[tuple[PointKey, ...], tuple[PointKey, ...]]:
        """
        Return the minimal plan needed to compute a single derived point.

        The plan is the transitive dependency closure of point_id, split into
        the derived functions that must be re-evaluated and the base (non-
        derived) points those functions read. This lets callers recompute one
        derived point without evaluating the full derived-point set.

        Args:
            point_id: The derived point to plan for.

        Returns:
            Tuple of (chain, base_dependencies) where:
            - chain: Derived points to evaluate, in dependency order, ending
              with point_id itself.
            - base_dependencies: Non-derived points the chain reads as inputs
              (free points and fixed hardpoints).

        Raises:
            KeyError: If point_id is not a derived point in this spec.
        """
        plan = self._computation_plans.get(point_id)
        if plan is not None:
            return plan

        if point_id not in self.spec.functions:
            raise KeyError(f"Point '{point_id}' is not a derived point in this spec.")

        # Walk the dependency graph from point_id, partitioning the closure
        # into derived nodes (which need evaluation) and base inputs.
        derived_needed: set[PointKey] = set()
        base_dependencies: set[PointKey] = set()
        stack = [point_id]
        while stack:
            node = stack.pop()
            if node in self.spec.functions:
                if node in derived_needed:
                    continue
                derived_needed.add(node)
                stack.extend(self.dependency_graph[node])
            else:
                base_dependencies.add(node)

        # Restrict the global topological order to the needed subset so the
        # chain evaluates dependencies before their dependents.
        chain = tuple(p for p in self.update_order if p in derived_needed)

        plan = (chain, tuple(base_dependencies))
        self._computation_plans[point_id] = plan
        return plan

    def _update_chain_in_place(
        self, positions: dict[PointKey, _V], chain: tuple[PointKey, ...]
    ) -> None:
        """
        Evaluate a pre-planned subset of derived points in-place.

        Args:
            positions: Dictionary to mutate in-place. Must contain the base
                       dependencies reported by _get_computation_plan().
            chain: Derived points in dependency order, as returned by
                   _get_computation_plan().
        """
        update_positions = cast(dict[PointKey, PositionValue], positions)
        for point_id in chain:
            positions[point_id] = cast(
                _V, self.spec.functions[point_id](update_positions)
            )

    def compute_point_jacobian(
        self,
        point_id: PointKey,
        positions: Mapping[PointKey, Point3],
        variable_points: Container[PointKey],
    ) -> dict[PointKey, np.ndarray]:
        """
        Compute a derived point's Jacobian with respect to relevant variables.

        Only the point's transitive dependency chain is evaluated. Dual-number
        inputs are created once, then their derivative seeds are reused for all
        input coordinates.

        Args:
            point_id: Derived point whose position is differentiated.
            positions: Current base and derived point positions.
            variable_points: Points whose coordinates are solver variables.

        Returns:
            Mapping from each relevant variable point to a 3x3 block. Column d
            contains d(point_id) / d(variable_point[d]).

        Raises:
            KeyError: If point_id is not a derived point in this spec.
        """
        chain, base_dependencies = self._get_computation_plan(point_id)

        # The values alias the input arrays, which remain constant throughout
        # this Jacobian evaluation. Fixed dependencies must be present because
        # derived functions read them, even though they are never seeded.
        dual_positions = {
            dependency: DualVec3(positions[dependency].data)
            for dependency in base_dependencies
        }

        jacobian_blocks: dict[PointKey, np.ndarray] = {}
        for dependency in base_dependencies:
            if dependency not in variable_points:
                continue

            block = np.empty((3, 3), dtype=np.float64)
            seed_derivative = dual_positions[dependency].deriv
            for dimension in range(3):
                # Seed d(input) / d(input[dimension]) with basis vector e_d.
                seed_derivative[:] = 0.0
                seed_derivative[dimension] = 1.0
                self._update_chain_in_place(dual_positions, chain)
                block[:, dimension] = dual_positions[point_id].deriv

            # Clear the seed before differentiating with respect to another point.
            seed_derivative[:] = 0.0
            jacobian_blocks[dependency] = block

        return jacobian_blocks
