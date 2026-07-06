"""
Solution-manifold sensitivities via the implicit function theorem.

A solved sweep step satisfies r(q, t) = 0, where q stacks the free point
coordinates and t stacks the sweep target values (e.g. the commanded wheel
center Z or rack Y). The solution q(t) traces a manifold as the targets vary.
This module computes the exact tangent of that manifold, dq/dt_j, for each
sweep target j -- no finite differencing across sweep steps.

Differentiating r(q(t), t) = 0 with respect to t_j gives:

    J * (dq/dt_j) + dr/dt_j = 0

where J = dr/dq is the residual Jacobian already computed analytically by
ResidualComputer. Constraint rows do not depend on t. Each target row has the
form residual = dot(position, direction) - t_j, so dr/dt_j is -1 on that
single row and zero elsewhere. The tangent is therefore the least-squares
solution of:

    J * (dq/dt_j) = e_j

with e_j the unit vector selecting target row j. For a consistent
(over)determined system with full column rank this solve is exact; near a
kinematic lock-out the tangents grow without bound, which is the physically
correct signal (an infinite motion ratio at the travel limit).

Free-point velocities come straight from the linear solve. Velocities of
derived points (wheel center, contact patch, ...) are obtained by one
forward-mode dual-number pass: seed every free point with its velocity and
propagate through the derived-point chain (a Jacobian-vector product).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from kinematics.constraints import Constraint, PointOnLineConstraint
from kinematics.core.dual import seed_positions_with_tangent
from kinematics.core.point_ref import PointKey
from kinematics.core.types import PointTarget
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import ResidualComputer
from kinematics.state import SuspensionState


@dataclass(frozen=True)
class TangentField:
    """
    First-order response of every point position to one sweep target.

    Attributes:
        target_index: Index of the driving target within the step's target
            list (matches the sweep dimension ordering).
        target: The driving PointTarget itself, kept so consumers can
            classify the driver (which point, which direction).
        velocities: Mapping of every point key to d(position)/d(target
            value), a length-3 array. Units are mm of point motion per unit
            of target value (itself mm along the target direction). Fixed
            hardpoints carry zero velocity; derived points carry the
            chain-rule propagated velocity.
    """

    target_index: int
    target: PointTarget
    velocities: dict[PointKey, np.ndarray]

    def velocity(self, point_id: PointKey) -> np.ndarray:
        """
        Velocity of a single point, zeros if the point is unknown.
        """
        vel = self.velocities.get(point_id)
        if vel is None:
            return np.zeros(3, dtype=np.float64)
        return vel


def compute_state_tangents(
    state: SuspensionState,
    constraints: list[Constraint],
    derived_manager: DerivedPointsManager,
    step_targets: Sequence[PointTarget],
) -> list[TangentField]:
    """
    Compute one tangent field per sweep target for a solved state.

    Args:
        state: A converged suspension state (residuals ~ 0 at its positions).
        constraints: The constraint list the state was solved against.
        derived_manager: Derived-points manager matching the suspension.
        step_targets: The targets active at this sweep step, in sweep
            dimension order. Target values are irrelevant here (the Jacobian
            does not depend on them); identities and directions matter.

    Returns:
        A list of TangentField, one per target, in target order.
    """
    if not step_targets:
        return []

    # Work on a scratch copy: ResidualComputer mutates its state buffer.
    scratch = state.copy()
    computer = ResidualComputer(
        constraints=constraints,
        derived_manager=derived_manager,
        state_buffer=scratch,
        n_target_variables=len(step_targets),
    )

    free_array = scratch.get_free_array()
    jacobian = computer.compute_jacobian(free_array, list(step_targets))

    # Some scalar residuals are norms that vanish quadratically at the
    # solution (e.g. PointOnLineConstraint: distance from the line). Their
    # Jacobian row is structurally zero exactly where we evaluate it, so it
    # carries no directional information and the linear system would lose
    # that constraint entirely. Append equivalent smooth rows that pin the
    # first-order motion instead.
    pin_rows = _degenerate_constraint_pins(constraints, computer)
    if pin_rows:
        jacobian = np.vstack([jacobian, np.asarray(pin_rows)])

    # Right-hand sides: differentiating target row j's residual
    # (dot(position, direction) - t_j) by t_j gives -1, so moving it to the
    # right-hand side selects unit vector e_j. Solve all targets at once.
    # Appended pin rows are true constraints, so their rate is zero.
    n_targets = len(step_targets)
    rhs = np.zeros((jacobian.shape[0], n_targets), dtype=np.float64)
    for j in range(n_targets):
        rhs[computer.n_constraints + j, j] = 1.0

    # Least-squares solve of J * dq = e_j. Exact when the constraint system
    # is consistent and J has full column rank (the same condition the LM
    # solver itself needs); rank-deficient configurations yield the
    # minimum-norm tangent.
    dq_all, _residuals, _rank, _sv = np.linalg.lstsq(jacobian, rhs, rcond=None)

    fields: list[TangentField] = []
    for j, target in enumerate(step_targets):
        # Scatter the flat solution back into per-point velocity vectors.
        free_velocities: dict[PointKey, np.ndarray] = {}
        for point_id, offset in computer.point_var_offsets.items():
            free_velocities[point_id] = dq_all[offset : offset + 3, j].copy()

        # One dual-number pass propagates the free-point velocities through
        # the derived-point chain (Jacobian-vector product): every derived
        # point picks up its chain-rule velocity, every fixed point stays at
        # zero.
        dual_positions = seed_positions_with_tangent(scratch.positions, free_velocities)
        derived_manager.update_in_place(dual_positions)

        velocities: dict[PointKey, np.ndarray] = {
            point_id: dual.deriv.copy() for point_id, dual in dual_positions.items()
        }
        fields.append(
            TangentField(target_index=j, target=target, velocities=velocities)
        )

    return fields


def _degenerate_constraint_pins(
    constraints: list[Constraint],
    computer: ResidualComputer,
) -> list[np.ndarray]:
    """
    Smooth replacement rows for constraints whose Jacobian vanishes at the
    solution.

    PointOnLineConstraint measures the perpendicular distance from the line.
    On the line that distance is zero and its gradient is undefined (the
    residual is a norm), so the solver zeroes the row. To first order the
    constraint is equivalent to two linear pins: the point may not move
    along either of two directions n1, n2 spanning the plane perpendicular
    to the line. Those pin rows are full-rank and exact.

    Args:
        constraints: The constraint list.
        computer: The residual computer providing variable offsets.

    Returns:
        List of Jacobian rows (length n_vars) to append, possibly empty.
    """
    rows: list[np.ndarray] = []
    for constraint in constraints:
        if not isinstance(constraint, PointOnLineConstraint):
            continue
        offset = computer.point_var_offsets.get(constraint.point_id)
        if offset is None:
            # The constrained point is not a solver variable; nothing to pin.
            continue

        direction = constraint.line_direction.data
        direction = direction / np.linalg.norm(direction)

        # Build an orthonormal basis {n1, n2} of the plane perpendicular to
        # the line: cross the direction with its least-aligned world axis.
        least_aligned = np.zeros(3)
        least_aligned[int(np.argmin(np.abs(direction)))] = 1.0
        n1 = np.cross(direction, least_aligned)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(direction, n1)

        for normal in (n1, n2):
            row = np.zeros(computer.n_vars, dtype=np.float64)
            row[offset : offset + 3] = normal
            rows.append(row)
    return rows


def combine_tangents(
    fields: Sequence[TangentField],
    coefficients: Sequence[float],
) -> dict[PointKey, np.ndarray]:
    """
    Linearly combine tangent fields into a modal velocity field.

    Sensitivities are linear, so the response to a coordinated input (e.g.
    a roll mode commanding left wheel up and right wheel down) is the same
    linear combination of the per-target tangents:

        v_mode = sum_j coefficient_j * v_target_j

    Args:
        fields: Tangent fields to combine.
        coefficients: One weight per field.

    Returns:
        Mapping of point key to combined velocity (length-3 arrays).
    """
    if len(fields) != len(coefficients):
        raise ValueError(
            f"Field/coefficient count mismatch: {len(fields)} fields, "
            f"{len(coefficients)} coefficients."
        )

    combined: dict[PointKey, np.ndarray] = {}
    for field, coefficient in zip(fields, coefficients):
        for point_id, velocity in field.velocities.items():
            accumulated = combined.get(point_id)
            if accumulated is None:
                combined[point_id] = coefficient * velocity
            else:
                accumulated += coefficient * velocity
    return combined


def tangent_positions(
    state: SuspensionState,
    velocities: Mapping[PointKey, np.ndarray],
):
    """
    Seed a state's positions with a velocity field for dual evaluation.

    Convenience wrapper: metric kernels evaluated on the returned mapping
    produce DualScalar results whose .deriv is the metric's rate along the
    tangent (directional derivative).

    Args:
        state: The solved state providing point positions.
        velocities: Velocity field, e.g. from a TangentField or
            combine_tangents().

    Returns:
        Mapping of point key to DualVec3 seeded with the velocity field.
    """
    return seed_positions_with_tangent(state.positions, velocities)
