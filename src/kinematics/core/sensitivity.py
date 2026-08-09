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

The calculation first obtains an orthonormal basis ``N`` for the local
mechanism tangent space ``ker(C_q)`` using an SVD of the permanent-constraint
Jacobian.  Its column count is the mechanism's local mobility.  Target
gradients are then restricted to that space and the small system

``(g_q N) alpha_j = e_j`` and ``dq/dt_j = N alpha_j``

is solved for every requested coordinate.  This separation makes the number of
remaining mechanism modes explicit, prevents target rows from compromising
permanent-constraint rates, and permits a well-posed zero-hold response when
the mechanism already has only one local degree of freedom.

Rows are normalised only for the reduced solve and conditioning diagnostics;
the physical derivatives are unchanged because the corresponding right-hand
side rows receive the same scaling.  A scale-independent projection ratio
``||g_i N|| / ||g_i||`` additionally reports when a coordinate is losing
coupling to the mechanism tangent space.  Raw constraint and target rates are
still checked afterwards with scale-aware tolerances.  This remains essential
for redundant or conflicting target bases: extra rows may be consistent, while
an incompatible overdetermined request must not become a least-squares
compromise.

Free-point velocities are propagated through the derived-point dependency
graph with dual numbers, and an optional post-derived closure applies the same
implicit closure used by the state solver.  Thus derived and closure-owned
outputs participate in the reported velocity field without becoming
independent solve variables.

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
_RANK_RELATIVE_TOLERANCE = np.finfo(np.float64).eps


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
    """Numerical health of one state's reduced coordinate-response solve."""

    n_equations: int
    n_variables: int
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float
    constraint_rank: int
    constraint_singular_values: tuple[float, ...]
    mobility: int
    target_rank: int
    target_projection_ratios: tuple[float, ...]
    responses: tuple[TangentResponseInfo, ...]

    @property
    def nullity(self) -> int:
        """Number of local mechanism directions left by the target basis."""
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
        """Smallest normalised target-space singular value, or zero if deficient."""
        if self.target_rank < self.mobility:
            return 0.0
        return self.singular_values[-1] if self.singular_values else 0.0

    @property
    def minimum_target_projection_ratio(self) -> float:
        """Smallest scale-independent target coupling to mechanism motion."""
        return min(self.target_projection_ratios, default=1.0)

    @property
    def finite(self) -> bool:
        """Whether the solved tangents, rates and singular values are finite.

        An infinite condition number is the valid representation of a singular
        matrix and is reported separately through rank and nullity; it does not
        by itself mean the computed minimum-norm values contain non-finite data.
        """
        return bool(
            all(np.isfinite(value) for value in self.singular_values)
            and all(np.isfinite(value) for value in self.constraint_singular_values)
            and all(np.isfinite(value) for value in self.target_projection_ratios)
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


@dataclass(frozen=True)
class _ConstraintTangentSpace:
    """Orthonormal local-motion basis obtained from permanent constraints."""

    basis: np.ndarray
    rank: int
    singular_values: tuple[float, ...]

    @property
    def mobility(self) -> int:
        """Dimension of the permanent-constraint null space."""
        return int(self.basis.shape[1])


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
    constraint_jacobian = jacobian[: computer.n_constraints]
    target_jacobian = jacobian[
        computer.n_constraints : computer.n_constraints + len(step_targets)
    ]

    # Norm residuals such as point-on-line have a zero row at the solution.
    # Add equivalent smooth first-order pins so the tangent retains them.
    pin_rows = _degenerate_constraint_pins(constraints, computer)
    if pin_rows:
        constraint_jacobian = np.vstack([constraint_jacobian, np.asarray(pin_rows)])

    n_targets = len(step_targets)
    constraint_space = _constraint_tangent_space(
        constraint_jacobian,
        computer.n_vars,
    )
    restricted_targets = target_jacobian @ constraint_space.basis
    normalised_targets, row_scales = _normalise_rows(restricted_targets)
    target_right_hand_sides = np.eye(n_targets, dtype=np.float64)
    normalised_right_hand_sides = row_scales[:, None] * target_right_hand_sides
    reduced_tangents, target_rank, target_singular_values = _solve_reduced_responses(
        normalised_targets,
        normalised_right_hand_sides,
        constraint_space.mobility,
        n_targets,
    )
    tangent_arrays = constraint_space.basis @ reduced_tangents
    combined_rank = constraint_space.rank + target_rank
    full_column_rank = combined_rank == computer.n_vars

    # An already ill-conditioned solve can overflow while reconstructing its
    # rates. Preserve those infinities in the diagnostic result without
    # leaking NumPy runtime warnings from an advisory calculation.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        constraint_rate_residuals = constraint_jacobian @ tangent_arrays
        target_rate_residuals = (
            target_jacobian @ tangent_arrays - target_right_hand_sides
        )
    complete_jacobian = np.vstack([constraint_jacobian, target_jacobian])
    responses = tuple(
        _response_info(
            target_index=target_index,
            jacobian=complete_jacobian,
            tangent=tangent_arrays[:, target_index],
            right_hand_side=np.concatenate(
                (
                    np.zeros(constraint_jacobian.shape[0], dtype=np.float64),
                    target_right_hand_sides[:, target_index],
                )
            ),
            constraint_rate_residuals=constraint_rate_residuals[:, target_index],
            target_rate_residuals=target_rate_residuals[:, target_index],
            full_column_rank=full_column_rank,
        )
        for target_index in range(n_targets)
    )
    target_singular_tuple = tuple(float(value) for value in target_singular_values)
    solve_info = TangentSolveInfo(
        n_equations=int(complete_jacobian.shape[0]),
        n_variables=computer.n_vars,
        rank=combined_rank,
        singular_values=target_singular_tuple,
        condition_number=_target_condition_number(
            target_singular_values,
            target_rank,
            constraint_space.mobility,
        ),
        constraint_rank=constraint_space.rank,
        constraint_singular_values=constraint_space.singular_values,
        mobility=constraint_space.mobility,
        target_rank=target_rank,
        target_projection_ratios=_target_projection_ratios(
            target_jacobian,
            restricted_targets,
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


def _constraint_tangent_space(
    constraint_jacobian: np.ndarray,
    n_variables: int,
) -> _ConstraintTangentSpace:
    """Return an orthonormal basis for the permanent-constraint null space."""
    if constraint_jacobian.shape != (constraint_jacobian.shape[0], n_variables):
        raise ValueError("Constraint Jacobian has an unexpected variable count")

    normalised, _row_scales = _normalise_rows(constraint_jacobian)
    if n_variables == 0:
        return _ConstraintTangentSpace(
            basis=np.empty((0, 0), dtype=np.float64),
            rank=0,
            singular_values=(),
        )
    if normalised.shape[0] == 0:
        return _ConstraintTangentSpace(
            basis=np.eye(n_variables, dtype=np.float64),
            rank=0,
            singular_values=(),
        )

    _left, singular_values, right_transpose = np.linalg.svd(
        normalised,
        full_matrices=True,
    )
    rank = _singular_value_rank(singular_values, normalised.shape)
    return _ConstraintTangentSpace(
        basis=right_transpose[rank:].T.copy(),
        rank=rank,
        singular_values=tuple(float(value) for value in singular_values),
    )


def _solve_reduced_responses(
    normalised_targets: np.ndarray,
    normalised_right_hand_sides: np.ndarray,
    mobility: int,
    n_targets: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve target responses within the permanent-constraint tangent space."""
    if mobility == 0:
        return (
            np.empty((0, n_targets), dtype=np.float64),
            0,
            np.empty(0, dtype=np.float64),
        )
    if n_targets == 0:
        return (
            np.empty((mobility, 0), dtype=np.float64),
            0,
            np.empty(0, dtype=np.float64),
        )

    responses, _residuals, rank, singular_values = np.linalg.lstsq(
        normalised_targets,
        normalised_right_hand_sides,
        rcond=None,
    )
    return responses, int(rank), singular_values


def _normalise_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-norm meaningful rows and the applied inverse row scales.

    Jacobians can contain rows whose analytical gradient is zero at the solved
    configuration but whose floating-point evaluation leaves machine-scale
    noise.  Amplifying such a row to unit length invents a constraint and
    reduces the reported mobility.  Use the same dimension-scaled relative
    threshold as the SVD rank test before applying any row scaling.
    """
    if matrix.shape[0] == 0:
        return matrix.copy(), np.empty(0, dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1)
    scales = np.ones(matrix.shape[0], dtype=np.float64)
    largest_norm = float(np.max(row_norms, initial=0.0))
    threshold = max(matrix.shape) * _RANK_RELATIVE_TOLERANCE * largest_norm
    meaningful = row_norms > threshold
    scales[meaningful] = 1.0 / row_norms[meaningful]
    normalised = matrix * scales[:, None]
    normalised[~meaningful] = 0.0
    return normalised, scales


def _singular_value_rank(
    singular_values: np.ndarray,
    matrix_shape: tuple[int, int],
) -> int:
    """Apply NumPy's conventional dimension-scaled relative rank threshold."""
    if singular_values.size == 0:
        return 0
    tolerance = max(matrix_shape) * _RANK_RELATIVE_TOLERANCE * float(singular_values[0])
    return int(np.count_nonzero(singular_values > tolerance))


def _target_condition_number(
    singular_values: np.ndarray,
    target_rank: int,
    mobility: int,
) -> float:
    """Return normalised hold transversality conditioning."""
    if mobility == 0:
        return 1.0
    if target_rank < mobility or singular_values.size == 0:
        return np.inf
    smallest = float(singular_values[-1])
    return float(singular_values[0]) / smallest if smallest > 0.0 else np.inf


def _target_projection_ratios(
    target_jacobian: np.ndarray,
    restricted_targets: np.ndarray,
) -> tuple[float, ...]:
    """Return scale-independent coupling of each target to mechanism motion."""
    full_norms = np.linalg.norm(target_jacobian, axis=1)
    restricted_norms = np.linalg.norm(restricted_targets, axis=1)
    ratios = np.zeros(target_jacobian.shape[0], dtype=np.float64)
    nonzero = full_norms > 0.0
    ratios[nonzero] = restricted_norms[nonzero] / full_norms[nonzero]
    return tuple(float(min(max(value, 0.0), 1.0)) for value in ratios)


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
    """Build forward rate-space consistency diagnostics for one response.

    Scaling by ``||J|| ||dq||`` is inappropriate here: an ill-conditioned
    basis can make ``dq`` enormous and thereby excuse an order-one failure to
    meet the requested rates.  Judge the result in the output space instead,
    relative to the rates actually produced and requested.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        actual_rates = jacobian @ tangent
    actual_rate_scale = _infinity_norm(actual_rates)
    request_scale = _infinity_norm(right_hand_side)
    consistency_scale = actual_rate_scale + request_scale
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
