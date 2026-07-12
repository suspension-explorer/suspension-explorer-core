"""
Solution-manifold sensitivities via the implicit function theorem.

A solved step satisfies r(q, t) = 0. Differentiating gives
J * dq/dt_j = e_j, where J is the analytical residual Jacobian and e_j
selects the target residual row. Derived-point velocities are propagated by
one forward dual-number pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from kinematics.constraints import Constraint, PointOnLineConstraint
from kinematics.core.dual import DualVec3, seed_positions_with_tangent
from kinematics.core.point_ref import PointKey
from kinematics.core.types import PointTarget
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import ResidualComputer
from kinematics.state import SuspensionState


@dataclass(frozen=True)
class TangentField:
    """First-order response of every point position to one sweep target."""

    target_index: int
    target: PointTarget
    velocities: dict[PointKey, np.ndarray]

    def velocity(self, point_id: PointKey) -> np.ndarray:
        """Return a point velocity, or zeros when the point is unknown."""
        velocity = self.velocities.get(point_id)
        if velocity is None:
            return np.zeros(3, dtype=np.float64)
        return velocity


@dataclass(frozen=True)
class TangentSolveInfo:
    """Numerical health of one state's tangent least-squares solve."""

    n_variables: int
    rank: int
    smallest_singular_value: float
    condition_number: float

    @property
    def rank_deficient(self) -> bool:
        """Whether the tangent system does not pin every variable."""
        return self.rank < self.n_variables


def compute_state_tangents(
    state: SuspensionState,
    constraints: list[Constraint],
    derived_manager: DerivedPointsManager,
    step_targets: Sequence[PointTarget],
) -> tuple[list[TangentField], TangentSolveInfo]:
    """Compute one tangent field per target and report solve health."""
    if not step_targets:
        return [], TangentSolveInfo(
            n_variables=0,
            rank=0,
            smallest_singular_value=0.0,
            condition_number=1.0,
        )

    # ResidualComputer mutates its state buffer, so use a scratch state.
    scratch = state.copy()
    computer = ResidualComputer(
        constraints=constraints,
        derived_manager=derived_manager,
        state_buffer=scratch,
        n_target_variables=len(step_targets),
    )
    free_array = scratch.get_free_array()
    jacobian = computer.compute_jacobian(free_array, list(step_targets))

    # Norm residuals such as point-on-line have a zero row at the solution.
    # Add equivalent smooth first-order pins so the tangent retains them.
    pin_rows = _degenerate_constraint_pins(constraints, computer)
    if pin_rows:
        jacobian = np.vstack([jacobian, np.asarray(pin_rows)])

    n_targets = len(step_targets)
    right_hand_sides = np.zeros(
        (jacobian.shape[0], n_targets),
        dtype=np.float64,
    )
    for target_index in range(n_targets):
        right_hand_sides[computer.n_constraints + target_index, target_index] = 1.0

    tangent_arrays, _residuals, rank, singular_values = np.linalg.lstsq(
        jacobian,
        right_hand_sides,
        rcond=None,
    )
    smallest_singular_value = (
        float(singular_values[-1]) if singular_values.size else 0.0
    )
    largest_singular_value = float(singular_values[0]) if singular_values.size else 0.0
    solve_info = TangentSolveInfo(
        n_variables=int(jacobian.shape[1]),
        rank=int(rank),
        smallest_singular_value=smallest_singular_value,
        condition_number=(
            largest_singular_value / smallest_singular_value
            if smallest_singular_value > 0.0
            else np.inf
        ),
    )

    fields: list[TangentField] = []
    for target_index, target in enumerate(step_targets):
        free_velocities: dict[PointKey, np.ndarray] = {}
        for point_id, offset in computer.point_var_offsets.items():
            free_velocities[point_id] = tangent_arrays[
                offset : offset + 3,
                target_index,
            ].copy()

        dual_positions = seed_positions_with_tangent(
            scratch.positions,
            free_velocities,
        )
        derived_manager.update_in_place(dual_positions)
        velocities = {
            point_id: dual_position.deriv.copy()
            for point_id, dual_position in dual_positions.items()
        }
        fields.append(
            TangentField(
                target_index=target_index,
                target=target,
                velocities=velocities,
            )
        )

    return fields, solve_info


def _degenerate_constraint_pins(
    constraints: list[Constraint],
    computer: ResidualComputer,
) -> list[np.ndarray]:
    """Build smooth first-order rows for zero-gradient norm constraints."""
    rows: list[np.ndarray] = []
    for constraint in constraints:
        if not isinstance(constraint, PointOnLineConstraint):
            continue
        offset = computer.point_var_offsets.get(constraint.point_id)
        if offset is None:
            continue

        direction = constraint.line_direction.data
        direction = direction / np.linalg.norm(direction)

        # Cross with the least-aligned world axis to obtain a stable basis of
        # the plane perpendicular to the line.
        least_aligned = np.zeros(3)
        least_aligned[int(np.argmin(np.abs(direction)))] = 1.0
        normal_1 = np.cross(direction, least_aligned)
        normal_1 /= np.linalg.norm(normal_1)
        normal_2 = np.cross(direction, normal_1)

        for normal in (normal_1, normal_2):
            row = np.zeros(computer.n_vars, dtype=np.float64)
            row[offset : offset + 3] = normal
            rows.append(row)
    return rows


def combine_tangents(
    fields: Sequence[TangentField],
    coefficients: Sequence[float],
) -> dict[PointKey, np.ndarray]:
    """Linearly combine tangent fields into one velocity field."""
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
) -> dict[PointKey, DualVec3]:
    """Seed a state's positions with a velocity field for dual evaluation."""
    return seed_positions_with_tangent(state.positions, velocities)
