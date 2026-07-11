"""
Kinematics solver using damped least squares.

This module provides functions to solve suspension kinematics by
satisfying geometric constraints and position targets using Levenberg-
Marquardt.
"""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from kinematics.constraints import (
    AngleConstraint,
    Constraint,
    CoplanarPointsConstraint,
    DistanceConstraint,
    EqualDistanceConstraint,
    FixedAxisConstraint,
    PointOnLineConstraint,
    PointOnPlaneConstraint,
    ScalarTripleProductConstraint,
    SphericalJointConstraint,
    ThreePointAngleConstraint,
    VectorsParallelConstraint,
    VectorsPerpendicularConstraint,
)
from kinematics.core.constants import (
    SOLVE_ACCEPT_RESIDUAL,
    SOLVE_TOLERANCE_GRAD,
    SOLVE_TOLERANCE_STEP,
    SOLVE_TOLERANCE_VALUE,
)
from kinematics.core.dual import seed_positions
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey
from kinematics.core.types import PointTarget, SweepConfig, TargetPositionMode
from kinematics.core.vector_utils.generic import project_coordinate
from kinematics.jacobians import (
    jac_angle,
    jac_coplanar,
    jac_distance,
    jac_equal_distance,
    jac_point_on_line,
    jac_three_point_angle,
    jac_vectors_parallel,
    jac_vectors_perpendicular,
)
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.state import SuspensionState
from kinematics.targets import resolve_target

# Levenberg-Marquardt: damped least squares that can deal with the system being
# overdetermined (m > n), as may be the case with any redundant (but consistent)
# constraints.
SOLVE_METHOD: str = "lm"


class SolverConfig(NamedTuple):
    """
    Configuration parameters for the kinematic solver.

    Attributes:
        ftol (float): Tolerance for the function value convergence.
        xtol (float): Tolerance for the solution vector convergence.
        gtol (float): Tolerance for the gradient convergence.
        verbose (int): Verbosity level for the solver output.
    """

    ftol: float = SOLVE_TOLERANCE_VALUE
    xtol: float = SOLVE_TOLERANCE_STEP
    gtol: float = SOLVE_TOLERANCE_GRAD
    verbose: int = 0
    residual_tolerance: float = SOLVE_ACCEPT_RESIDUAL


@dataclass
class SolverInfo:
    """
    Information about the solver's execution for a single solve step.

    Attributes:
        converged (bool): Whether the solver converged to a solution.
        nfev (int): Number of function evaluations performed.
        max_residual (float): Maximum residual value in the final solution.
    """

    converged: bool
    nfev: int
    max_residual: float


def validate_least_squares_dimensions(
    n_vars: int,
    n_residuals: int,
    *,
    method: str = SOLVE_METHOD,
) -> None:
    """
    Validate that the chosen least-squares method is compatible with the system size.

    Args:
        n_vars: Number of solver variables.
        n_residuals: Number of residual equations.
        method: Least-squares method passed to SciPy.

    Raises:
        ValueError: If the requested method cannot solve the given system size.
    """
    if method == "lm" and n_vars > n_residuals:
        raise ValueError(
            f"System is underdetermined (n_vars={n_vars} > m_res={n_residuals}). "
            "The solve method (Levenberg-Marquardt) requires at least as "
            "many residuals as variables."
        )


def solve_least_squares_problem(
    *,
    residual_function: Callable[..., np.ndarray],
    x_0: np.ndarray,
    args: tuple[Any, ...] = (),
    solver_config: SolverConfig = SolverConfig(),
    n_residuals: int | None = None,
    jacobian_function: Callable[..., np.ndarray] | str | None = None,
    method: str = SOLVE_METHOD,
) -> OptimizeResult:
    """
    Run a single least-squares solve with shared validation and tolerance handling.

    Args:
        residual_function: Residual callback passed to SciPy.
        x_0: Initial guess.
        args: Extra arguments forwarded to the residual function.
        solver_config: Shared solver tolerance and verbosity settings.
        n_residuals: Residual count for method-size validation. If omitted, no
            up-front dimension validation is performed.
        jacobian_function: Optional analytical Jacobian callback or SciPy Jacobian
            mode string.
        method: Least-squares method passed to SciPy.

    Returns:
        The raw SciPy optimize result.
    """
    if n_residuals is not None:
        validate_least_squares_dimensions(
            int(x_0.size),
            n_residuals,
            method=method,
        )

    least_squares_kwargs: dict[str, Any] = {
        "args": args,
        "method": method,
        "ftol": solver_config.ftol,
        "xtol": solver_config.xtol,
        "gtol": solver_config.gtol,
        "verbose": solver_config.verbose,
    }
    if jacobian_function is not None:
        least_squares_kwargs["jac"] = jacobian_function

    return least_squares(residual_function, x_0, **least_squares_kwargs)


class ResidualComputer:
    """
    Computes residuals for kinematic constraints and targets.

    This class holds buffers that are reused across multiple solver evaluations
    to minimize allocations. A single instance is used for an entire sweep,
    with different targets passed to compute() for each step.

    Attributes:
        constraints: Geometric constraints to evaluate.
        derived_manager: Manager for computing derived points in-place.
        state_buffer: Suspension state that is mutated during computation.
        residuals_buffer: Pre-allocated array for residuals (reused across calls).
    """

    def __init__(
        self,
        constraints: list[Constraint],
        derived_manager: DerivedPointsManager,
        state_buffer: SuspensionState,
        n_target_variables: int,
    ):
        self.constraints = constraints
        self.derived_manager = derived_manager
        self.state_buffer = state_buffer

        # Pre-allocate residuals buffer.
        # This is our constraint residuals + target residuals. Each target adds one
        # element to the residuals vector, so if we're sweeping wheel center Z and
        # steering at the same time, we need two extra slots.
        self.n_constraints = len(constraints)
        self.n_target_variables = n_target_variables
        self.n_residuals = self.n_constraints + self.n_target_variables
        self.residuals_buffer = np.empty(self.n_residuals, dtype=np.float64)

        # Jacobian infrastructure: map each free PointID to its column offset
        # in the flattened free_array, and pre-compute per-constraint plans.
        self.point_var_offsets = {
            pid: 3 * k for k, pid in enumerate(state_buffer.free_points_order)
        }
        self.n_vars = len(state_buffer.free_points_order) * 3
        self.jac_buffer = np.zeros((self.n_residuals, self.n_vars), dtype=np.float64)
        self.jac_plan = [self.build_jac_plan(c) for c in constraints]

    def validate_target_count(self, step_targets: list[PointTarget]) -> None:
        """
        Ensure each evaluation uses the fixed target count configured at init time.
        """
        if len(step_targets) != self.n_target_variables:
            raise ValueError(
                "ResidualComputer requires a fixed number of targets per evaluation "
                f"(expected {self.n_target_variables}, got {len(step_targets)})."
            )

    def compute(
        self,
        free_array: np.ndarray,
        step_targets: list[PointTarget],
    ) -> np.ndarray:
        """
        Compute residuals for the given free point positions and targets.

        This method mutates state_buffer in-place and reuses residuals_buffer
        for performance. The number of targets is fixed for the lifetime of
        the residual computer, so the residual vector shape is constant across
        evaluations.

        Args:
            free_array: Flattened array of free point coordinates.
            step_targets: Target constraints for this solve step. Must match
                the fixed target count configured at initialization.

        Returns:
            Copy of the fixed-size residual array containing
            [constraint_residuals, target_residuals].

        Note:
            The returned array is a copy because SciPy's least-squares solver
            keeps references to past evaluations.
        """
        self.validate_target_count(step_targets)

        # Update state buffer in-place with current guess.
        self.state_buffer.update_from_array(free_array)

        # Compute derived points in-place.
        self.derived_manager.update_in_place(self.state_buffer.positions)

        # Fill constraint residuals section: residuals[0:n_constraints].
        for i, constraint in enumerate(self.constraints):
            self.residuals_buffer[i] = constraint.residual(self.state_buffer.positions)

        # Fill target residuals: residuals[n_constraints:n_constraints+n_targets].
        offset = self.n_constraints
        for i, target in enumerate(step_targets):
            direction = resolve_target(target.direction)
            current_pos = self.state_buffer.positions[target.point_id]
            current_coordinate = project_coordinate(current_pos, direction)
            self.residuals_buffer[offset + i] = current_coordinate - target.value

        # Return a copy of the fixed-size residual vector. Note that we must return a
        # copy here because Scipy's least squares keeps references to the evaluated
        # arrays, so subsequent calls would overwrite previous values.
        return self.residuals_buffer.copy()

    # ------------------------------------------------------------------
    # Analytical Jacobian
    # ------------------------------------------------------------------

    def build_jac_plan(self, constraint: Constraint):
        """
        Pre-compute the Jacobian function and distribution mapping for constraint.

        Returns `(compute_fn, distribution)` where:
        - `compute_fn(positions_dict)` returns the partial-derivative array
          for all involved point coordinates (3 entries per point, in point order).
        - `distribution` is a list of `(deriv_offset, jac_col)` tuples indicating
          where to copy each 3-wide block into the Jacobian row.  Only free
          points appear in the distribution list.

        Jacobian functions operate on raw ndarrays, so Point3 positions are
        unwrapped via .data before being passed.
        """
        offsets = self.point_var_offsets
        compute_fn: Callable[[dict[PointKey, Point3]], np.ndarray]

        if isinstance(constraint, (DistanceConstraint, SphericalJointConstraint)):
            point_ids = (constraint.p1, constraint.p2)
            p1 = constraint.p1
            p2 = constraint.p2

            def compute_distance(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_distance(pos[p1].data, pos[p2].data)

            compute_fn = compute_distance

        elif isinstance(constraint, AngleConstraint):
            point_ids = (
                constraint.v1_start,
                constraint.v1_end,
                constraint.v2_start,
                constraint.v2_end,
            )
            v1_start = constraint.v1_start
            v1_end = constraint.v1_end
            v2_start = constraint.v2_start
            v2_end = constraint.v2_end

            def compute_angle(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_angle(
                    pos[v1_start].data,
                    pos[v1_end].data,
                    pos[v2_start].data,
                    pos[v2_end].data,
                )

            compute_fn = compute_angle

        elif isinstance(constraint, ThreePointAngleConstraint):
            point_ids = (constraint.p1, constraint.p2, constraint.p3)
            p1 = constraint.p1
            p2 = constraint.p2
            p3 = constraint.p3

            def compute_three_point_angle(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_three_point_angle(pos[p1].data, pos[p2].data, pos[p3].data)

            compute_fn = compute_three_point_angle

        elif isinstance(constraint, VectorsParallelConstraint):
            point_ids = (
                constraint.v1_start,
                constraint.v1_end,
                constraint.v2_start,
                constraint.v2_end,
            )
            v1_start = constraint.v1_start
            v1_end = constraint.v1_end
            v2_start = constraint.v2_start
            v2_end = constraint.v2_end

            def compute_vectors_parallel(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_vectors_parallel(
                    pos[v1_start].data,
                    pos[v1_end].data,
                    pos[v2_start].data,
                    pos[v2_end].data,
                )

            compute_fn = compute_vectors_parallel

        elif isinstance(constraint, VectorsPerpendicularConstraint):
            point_ids = (
                constraint.v1_start,
                constraint.v1_end,
                constraint.v2_start,
                constraint.v2_end,
            )
            v1_start = constraint.v1_start
            v1_end = constraint.v1_end
            v2_start = constraint.v2_start
            v2_end = constraint.v2_end

            def compute_vectors_perpendicular(
                pos: dict[PointKey, Point3],
            ) -> np.ndarray:
                return jac_vectors_perpendicular(
                    pos[v1_start].data,
                    pos[v1_end].data,
                    pos[v2_start].data,
                    pos[v2_end].data,
                )

            compute_fn = compute_vectors_perpendicular

        elif isinstance(constraint, EqualDistanceConstraint):
            point_ids = (
                constraint.p1,
                constraint.p2,
                constraint.p3,
                constraint.p4,
            )
            p1 = constraint.p1
            p2 = constraint.p2
            p3 = constraint.p3
            p4 = constraint.p4

            def compute_equal_distance(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_equal_distance(
                    pos[p1].data, pos[p2].data, pos[p3].data, pos[p4].data
                )

            compute_fn = compute_equal_distance

        elif isinstance(constraint, FixedAxisConstraint):
            point_ids = (constraint.point_id,)
            fixed = np.zeros(3)
            fixed[constraint.axis.value] = 1.0

            def compute_fixed_axis(pos: dict[PointKey, Point3]) -> np.ndarray:
                _ = pos
                return fixed

            compute_fn = compute_fixed_axis

        elif isinstance(constraint, PointOnLineConstraint):
            point_ids = (constraint.point_id,)
            point_id = constraint.point_id
            line_point = constraint.line_point.data
            line_direction = constraint.line_direction.data

            def compute_point_on_line(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_point_on_line(pos[point_id].data, line_point, line_direction)

            compute_fn = compute_point_on_line

        elif isinstance(constraint, PointOnPlaneConstraint):
            point_ids = (constraint.point_id,)
            normal = constraint.plane_normal.data.copy()

            def compute_point_on_plane(pos: dict[PointKey, Point3]) -> np.ndarray:
                _ = pos
                return normal

            compute_fn = compute_point_on_plane

        elif isinstance(constraint, ScalarTripleProductConstraint):
            point_ids = (
                constraint.p1,
                constraint.p2,
                constraint.p3,
                constraint.p4,
            )
            p1 = constraint.p1
            p2 = constraint.p2
            p3 = constraint.p3
            p4 = constraint.p4
            scale = constraint.scale

            def compute_scalar_triple(pos: dict[PointKey, Point3]) -> np.ndarray:
                return (
                    jac_coplanar(
                        pos[p1].data,
                        pos[p2].data,
                        pos[p3].data,
                        pos[p4].data,
                    )
                    / scale
                )

            compute_fn = compute_scalar_triple

        elif isinstance(constraint, CoplanarPointsConstraint):
            point_ids = (
                constraint.p1,
                constraint.p2,
                constraint.p3,
                constraint.p4,
            )
            p1 = constraint.p1
            p2 = constraint.p2
            p3 = constraint.p3
            p4 = constraint.p4

            def compute_coplanar(pos: dict[PointKey, Point3]) -> np.ndarray:
                return jac_coplanar(
                    pos[p1].data, pos[p2].data, pos[p3].data, pos[p4].data
                )

            compute_fn = compute_coplanar

        else:
            raise TypeError(
                f"No Jacobian implementation for {type(constraint).__name__}"
            )

        # Map each involved point's 3-wide derivative block to its Jacobian
        # column offset.  Points that are not free are omitted (their partials
        # are structurally zero in the Jacobian).
        distribution = []
        for i, pid in enumerate(point_ids):
            if pid in offsets:
                distribution.append((3 * i, offsets[pid]))

        return compute_fn, distribution

    def compute_jacobian(
        self,
        free_array: np.ndarray,
        step_targets: list[PointTarget],
    ) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix.

        Has the same calling convention as `compute` so it can be passed
        directly as `jac=` to `scipy.optimize.least_squares`.
        """
        self.validate_target_count(step_targets)

        self.state_buffer.update_from_array(free_array)
        self.derived_manager.update_in_place(self.state_buffer.positions)

        jac = self.jac_buffer
        jac[:] = 0.0

        positions = self.state_buffer.positions

        # Constraint rows.
        for i, (compute_fn, distribution) in enumerate(self.jac_plan):
            try:
                derivs = compute_fn(positions)
            except ZeroDivisionError:
                # Singularity at an exactly-satisfied constraint (e.g. point
                # already on the line).  The residual is zero so this row's
                # contribution to J^T r is zero -- safe to leave as zeros.
                continue
            for d_start, j_col in distribution:
                jac[i, j_col : j_col + 3] = derivs[d_start : d_start + 3]

        # Target rows: residual = dot(position, direction) - value.
        offset = self.n_constraints
        for i, target in enumerate(step_targets):
            row = offset + i
            direction = resolve_target(target.direction)

            if target.point_id in self.point_var_offsets:
                # Free-point target: derivative w.r.t. its own coords is
                # just the direction vector.
                col = self.point_var_offsets[target.point_id]
                jac[row, col : col + 3] = direction.data
            else:
                # Derived-point target (e.g. WHEEL_CENTER): the point is
                # computed from free points via a nonlinear function, so the
                # Jacobian row needs the chain rule through that computation.
                # We use forward-mode autodiff with dual numbers to get exact
                # derivatives: seed one input coordinate at a time and read
                # the derivative of the output.
                for pid in self.state_buffer.free_points_order:
                    col = self.point_var_offsets[pid]
                    for d in range(3):
                        dual_pos = seed_positions(positions, pid, d)
                        self.derived_manager.update_in_place(dual_pos)
                        target_dual = dual_pos[target.point_id]
                        # d(dot(pos, direction)) / d(pid[d]) =
                        #     dot(direction, d(pos)/d(pid[d]))
                        jac[row, col + d] = float(
                            np.dot(direction.data, target_dual.deriv)
                        )

        return jac.copy()


def convert_targets_to_absolute(
    targets: list[PointTarget],
    initial_state: SuspensionState,
) -> list[PointTarget]:
    """
    Convert all targets to absolute coordinates.

    This function implements the "convert early" pattern: all mode-specific
    logic is handled here, once, before solving begins. The solver then works
    exclusively with absolute coordinates.

    Args:
        targets: List of targets in mixed modes (RELATIVE or ABSOLUTE).
        initial_state: Reference state for resolving RELATIVE targets.

    Returns:
        List of targets with all modes converted to ABSOLUTE.
    """
    resolved: list[PointTarget] = []

    for target in targets:
        if target.mode == TargetPositionMode.ABSOLUTE:
            resolved.append(target)
            continue

        # Convert a relative displacement to an absolute scalar coordinate along the
        # target direction. Project the initial position onto the (unit) direction to
        # get the initial coordinate, then add the displacement.
        direction = resolve_target(target.direction)
        initial_pos = initial_state.positions[target.point_id]
        initial_coord = project_coordinate(initial_pos, direction)
        absolute_value = initial_coord + target.value

        # Create new target with absolute value and mode.
        resolved.append(
            PointTarget(
                point_id=target.point_id,
                direction=target.direction,
                value=absolute_value,
                mode=TargetPositionMode.ABSOLUTE,
            )
        )

    return resolved


def describe_constraint(constraint: Constraint) -> str:
    """Return a compact description of a constraint and its points."""
    point_names = ", ".join(
        sorted(
            getattr(point, "name", str(point)) for point in constraint.involved_points
        )
    )
    return f"{type(constraint).__name__}({point_names})"


def describe_worst_residual(
    residuals: np.ndarray,
    constraints: list[Constraint],
    step_targets: list[PointTarget],
) -> str:
    """Describe the constraint or target owning the largest residual row."""
    worst_index = int(np.argmax(np.abs(residuals)))
    if worst_index < len(constraints):
        return f"constraint {describe_constraint(constraints[worst_index])}"
    target = step_targets[worst_index - len(constraints)]
    point_name = getattr(target.point_id, "name", str(target.point_id))
    return f"target on point '{point_name}' (direction {target.direction})"


def solve_suspension_sweep(
    initial_state: SuspensionState,
    constraints: list[Constraint],
    sweep_config: SweepConfig,
    derived_manager: DerivedPointsManager,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[list[SuspensionState], list[SolverInfo]]:
    """
    Solves a series of kinematic states by sweeping through target
    configurations using damped non-linear least squares. This function
    performs a sweep where each step in the sweep corresponds to a set of
    targets, solving sequentially from the initial state. All state and
    residual buffers are reused across evaluations to minimize allocations.

    Args:
        initial_state: The initial suspension state to start the sweep from.
        constraints: List of geometric constraints to satisfy.
        sweep_config: Configuration for the sweep, including step count and targets.
        derived_manager: Manager to compute derived points in-place.
        solver_config: Configuration parameters for the solver.

    Returns:
        Tuple of (solved_states, solver_stats) where:
        - solved_states: List of converged suspension states for each sweep step
        - solver_stats: List of solver diagnostics for each step

    Raises:
        ValueError: If the system is underdetermined (more variables than residuals).
        RuntimeError: If the solver fails to converge at any step.
    """
    # Convert all targets to absolute coordinates once before solving.
    sweep_targets = [
        convert_targets_to_absolute(
            [sweep[i] for sweep in sweep_config.target_sweeps], initial_state
        )
        for i in range(sweep_config.n_steps)
    ]

    # Working state reused across the sweep; mutated in-place for performance.
    working_state = initial_state.copy()

    # For each step in our sweep, we will keep a copy of the solved state. This is
    # our result dataset.
    solution_states: list[SuspensionState] = []
    solver_stats: list[SolverInfo] = []

    # Create residual computer with pre-allocated buffers.
    # This single instance is reused across all solver evaluations in the sweep.
    residual_computer = ResidualComputer(
        constraints=constraints,
        derived_manager=derived_manager,
        state_buffer=working_state,
        n_target_variables=len(sweep_config.target_sweeps),
    )

    # Initial guess built from the working state's free points.
    x_0 = working_state.get_free_array()

    # Both counts are fixed for the entire sweep, so pass the residual count through
    # the shared least-squares helper for LM dimension validation.
    n_residuals = residual_computer.n_residuals

    for step_index, step_targets in enumerate(sweep_targets):
        result = solve_least_squares_problem(
            residual_function=residual_computer.compute,
            x_0=x_0,
            args=(step_targets,),
            solver_config=solver_config,
            n_residuals=n_residuals,
            jacobian_function=residual_computer.compute_jacobian,  # pyright: ignore[reportArgumentType]
        )

        if not result.success:
            raise RuntimeError(
                f"Solver failed to converge for targets: {step_targets}."
                f"\nMessage: {result.message}"
            )

        # Optimizer convergence does not prove feasibility: least squares can
        # converge to a compromise when the requested mechanism state is
        # unreachable. Reject that state before it enters the sweep history.
        step_max_residual = (
            float(np.max(np.abs(result.fun))) if len(result.fun) > 0 else 0.0
        )
        if step_max_residual > solver_config.residual_tolerance:
            worst = describe_worst_residual(result.fun, constraints, step_targets)
            raise RuntimeError(
                f"Solve at sweep step {step_index} did not reach an acceptable "
                f"residual: worst residual {step_max_residual:.6g} exceeds the "
                f"acceptance tolerance {solver_config.residual_tolerance:.6g}. "
                f"Worst residual row: {worst}. The mechanism likely cannot reach "
                "the requested targets (kinematic lock-out / infeasible target "
                "combination)."
            )

        # Synchronize working_state with the accepted solution. The solver
        # evaluates residuals at many candidate positions during the solve,
        # mutating working_state each time. When it terminates, working_state
        # may be left at a position from gradient estimation (e.g., x* + epsilon)
        # rather than the actual solution x*. We must explicitly restore it to
        # result.x to ensure the stored state matches the returned solution.
        #
        # This synchronisation is necessary because we reuse working_state across all
        # residual evaluations for performance (avoiding dict allocations on each call).
        # The tradeoff is this explicit sync requirement.
        working_state.update_from_array(result.x)
        derived_manager.update_in_place(working_state.positions)

        # Store finalized state for this step.
        solution_states.append(working_state.copy())

        # Collect solver information for this step.
        solver_info = SolverInfo(
            converged=result.success,
            nfev=result.nfev,
            max_residual=step_max_residual,
        )
        solver_stats.append(solver_info)

        # The result becomes our local first guess for the next step.
        x_0 = result.x

    return solution_states, solver_stats
