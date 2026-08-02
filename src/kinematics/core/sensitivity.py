"""Analytical tangent fields on a solved suspension configuration.

Let ``q`` contain the free point coordinates, ``C(q)`` the permanent
kinematic constraints, and ``g(q) - t`` the residuals for a supplied basis of
scalar targets.  At a solved state, differentiating

``r(q, t) = [C(q), g(q) - t] = 0``

with respect to target ``t_j`` gives the implicit-function tangent equation

``r_q dq/dt_j = [0, e_j]``.

Consequently, each :class:`TangentField` is a *partial* response to one scalar
target: its selected target has unit rate and every other target in the same
basis has zero rate.  Changing that target basis changes which coordinates are
held at zero rate and can therefore produce a different partial derivative at
the same solved position.  The basis is part of the question being asked, not
just a numerical implementation detail.

The least-squares solve acts on free points.  Their velocities are then
propagated through the derived-point dependency graph with dual numbers, and
an optional post-derived closure applies the same implicit closure used by the
state solver.  Thus derived and closure-owned outputs participate in the
reported velocity field without becoming independent solve variables.

:class:`TangentSolveInfo` reports column rank, remaining nullity, singular
values and conditioning for the common tangent matrix.  Each requested
response also records permanent-constraint and full target-basis rate
residuals.  Rate consistency uses an absolute floor plus a relative term
scaled by ``||J||_inf ||dq||_inf + ||b||_inf``.  This is essential for an
overdetermined system: full column rank can make a least-squares answer unique
while still making it an inconsistent compromise that satisfies neither the
constraints nor the requested target rates.

This module deliberately assigns no physical meaning such as steering, bump,
roll, or sweep-path motion to a field.  Callers establish that meaning by
choosing a target basis and selecting a response semantically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from kinematics.core.constraints import Constraint, PointOnLineConstraint
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.dual import DualVec3, seed_positions_with_tangent
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.solver import ResidualComputer
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import ScalarCoordinateTarget

_RATE_RESIDUAL_ABSOLUTE_TOLERANCE = 1e-12
_RATE_RESIDUAL_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class TangentField:
    """First-order response of every point position to one sweep target."""

    target_index: int
    target: ScalarCoordinateTarget
    velocities: dict[PointKey, np.ndarray]

    def velocity(self, point_id: PointKey) -> np.ndarray:
        """Return a point velocity, or zeros when the point is unknown."""
        velocity = self.velocities.get(point_id)
        if velocity is None:
            return np.zeros(3, dtype=np.float64)
        return velocity


@dataclass(frozen=True)
class TangentResponseInfo:
    """Rate consistency of one requested target response.

    Residual tuples preserve the row order supplied to
    :func:`compute_state_tangents`.  Constraint residuals include any smooth
    first-order pins added for degenerate norm constraints.  Target residuals
    are measured against the complete identity-column request: the selected
    target rate is one and all other target rates are zero.
    """

    target_index: int
    constraint_rate_residuals: tuple[float, ...]
    target_rate_residuals: tuple[float, ...]
    consistency_tolerance: float
    full_column_rank: bool
    finite: bool

    @property
    def max_constraint_rate_residual(self) -> float:
        """Maximum absolute permanent-constraint rate error."""
        return _max_abs(self.constraint_rate_residuals)

    @property
    def selected_target_rate_residual(self) -> float:
        """Absolute error from the requested unit rate."""
        if self.target_index >= len(self.target_rate_residuals):
            return np.inf
        return abs(self.target_rate_residuals[self.target_index])

    @property
    def max_other_target_rate_residual(self) -> float:
        """Maximum absolute rate error among targets meant to remain fixed."""
        return _max_abs(
            residual
            for index, residual in enumerate(self.target_rate_residuals)
            if index != self.target_index
        )

    @property
    def max_rate_residual(self) -> float:
        """Maximum absolute constraint or target rate error."""
        return max(
            self.max_constraint_rate_residual,
            _max_abs(self.target_rate_residuals),
        )

    @property
    def rate_consistent(self) -> bool:
        """Whether all requested rates are met within the scaled tolerance."""
        return self.finite and self.max_rate_residual <= self.consistency_tolerance

    @property
    def unique(self) -> bool:
        """Whether this is a finite, consistent, uniquely determined response."""
        return self.full_column_rank and self.rate_consistent


@dataclass(frozen=True)
class TangentSolveInfo:
    """Numerical health of one state's common tangent least-squares solve."""

    n_equations: int
    n_variables: int
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float
    responses: tuple[TangentResponseInfo, ...]

    @property
    def nullity(self) -> int:
        """Number of solve-variable directions left unconstrained."""
        return max(self.n_variables - self.rank, 0)

    @property
    def full_column_rank(self) -> bool:
        """Whether the system determines every solve variable."""
        return self.nullity == 0

    @property
    def rank_deficient(self) -> bool:
        """Whether the tangent system does not pin every variable."""
        return not self.full_column_rank

    @property
    def smallest_singular_value(self) -> float:
        """Smallest reported singular value, or zero for an empty matrix."""
        return self.singular_values[-1] if self.singular_values else 0.0

    @property
    def finite(self) -> bool:
        """Whether the solved tangents, rates and singular values are finite.

        An infinite condition number is the valid representation of a singular
        matrix and is reported separately through rank and nullity; it does not
        by itself mean the computed minimum-norm values contain non-finite data.
        """
        return bool(
            all(np.isfinite(value) for value in self.singular_values)
            and all(response.finite for response in self.responses)
        )

    @property
    def rate_consistent(self) -> bool:
        """Whether every requested target response is rate-consistent."""
        return all(response.rate_consistent for response in self.responses)

    def response_for_target(self, target_index: int) -> TangentResponseInfo:
        """Return diagnostics for a target index without relying on tuple order."""
        matching = [
            response
            for response in self.responses
            if response.target_index == target_index
        ]
        if len(matching) != 1:
            raise KeyError(
                f"No unique tangent response for target index {target_index}."
            )
        return matching[0]


def compute_state_tangents(
    state: SuspensionState,
    constraints: list[Constraint],
    derived_manager: DerivedPointsManager,
    step_targets: Sequence[ScalarCoordinateTarget],
    post_derived_update: Callable[[dict], float | None] | None = None,
) -> tuple[list[TangentField], TangentSolveInfo]:
    """Compute one tangent field per target and report solve health.

    ``post_derived_update`` mirrors the sweep's post-solve ground closure: it
    is applied to the dual position map after the derived-point update so
    closure outputs (the coupled wheel contact centres) carry their implicit
    derivatives into the tangent field instead of the zero seed.
    """
    if not step_targets:
        return [], TangentSolveInfo(
            n_equations=0,
            n_variables=0,
            rank=0,
            singular_values=(),
            condition_number=1.0,
            responses=(),
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
    full_column_rank = int(rank) == int(jacobian.shape[1])
    # An already ill-conditioned solve can overflow while reconstructing its
    # rates. Preserve those infinities in the diagnostic result without
    # leaking NumPy runtime warnings from an advisory calculation.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rate_residuals = jacobian @ tangent_arrays - right_hand_sides
    constraint_row_indices = [*range(computer.n_constraints)]
    constraint_row_indices.extend(
        range(computer.n_residuals, computer.n_residuals + len(pin_rows))
    )
    target_slice = slice(
        computer.n_constraints,
        computer.n_constraints + n_targets,
    )
    responses = tuple(
        _response_info(
            target_index=target_index,
            jacobian=jacobian,
            tangent=tangent_arrays[:, target_index],
            right_hand_side=right_hand_sides[:, target_index],
            constraint_rate_residuals=rate_residuals[
                constraint_row_indices, target_index
            ],
            target_rate_residuals=rate_residuals[target_slice, target_index],
            full_column_rank=full_column_rank,
        )
        for target_index in range(n_targets)
    )
    solve_info = TangentSolveInfo(
        n_equations=int(jacobian.shape[0]),
        n_variables=int(jacobian.shape[1]),
        rank=int(rank),
        singular_values=tuple(float(value) for value in singular_values),
        condition_number=(
            largest_singular_value / smallest_singular_value
            if smallest_singular_value > 0.0
            else np.inf
        ),
        responses=responses,
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
        if post_derived_update is not None:
            post_derived_update(dual_positions)
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


def _response_info(
    *,
    target_index: int,
    jacobian: np.ndarray,
    tangent: np.ndarray,
    right_hand_side: np.ndarray,
    constraint_rate_residuals: np.ndarray,
    target_rate_residuals: np.ndarray,
    full_column_rank: bool,
) -> TangentResponseInfo:
    """Build scale-aware consistency diagnostics for one response column."""
    jacobian_scale = _infinity_norm(jacobian)
    tangent_scale = _infinity_norm(tangent)
    request_scale = _infinity_norm(right_hand_side)
    consistency_scale = jacobian_scale * tangent_scale + request_scale
    tolerance = (
        _RATE_RESIDUAL_ABSOLUTE_TOLERANCE
        + _RATE_RESIDUAL_RELATIVE_TOLERANCE * consistency_scale
    )
    values = np.concatenate(
        (
            tangent,
            constraint_rate_residuals,
            target_rate_residuals,
        )
    )
    return TangentResponseInfo(
        target_index=target_index,
        constraint_rate_residuals=tuple(
            float(value) for value in constraint_rate_residuals
        ),
        target_rate_residuals=tuple(float(value) for value in target_rate_residuals),
        consistency_tolerance=tolerance,
        full_column_rank=full_column_rank,
        finite=bool(np.all(np.isfinite(values)) and np.isfinite(tolerance)),
    )


def _max_abs(values: Iterable[float]) -> float:
    """Return the largest absolute value, with zero for an empty sequence."""
    return max((abs(value) for value in values), default=0.0)


def _infinity_norm(values: np.ndarray) -> float:
    """Return an infinity norm that is defined as zero for an empty array."""
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, ord=np.inf))


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

        # Cross with the least-aligned chassis axis to obtain a stable basis of
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
