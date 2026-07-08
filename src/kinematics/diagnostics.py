"""
Post-sweep diagnostics for suspension solves.

The solver raises hard on non-convergence and on an unacceptable residual (see
:func:`~kinematics.solver.solve_suspension_sweep`). This module provides a
*softer*, sweep-wide health report on top of that: it inspects a completed sweep
(states + solver stats) and surfaces issues a caller may want to warn about even
when every step technically solved -- branch snaps mid-sweep, a rocker that has
inverted its handedness, or a link/lever pair creeping toward its transmission
(toggle) singularity where it is about to top out.

The public entry point is :func:`diagnose_sweep`. Each check is a small,
well-named private function returning a list of :class:`DiagnosticIssue`.

Key-type handling: the module works for both plain single-corner models
(``PointID`` keys) and the axle model (``PointRef`` keys) by iterating over
"rocker corners" through :func:`_iter_rocker_corners`, which yields a per-corner
key-mapping function. Axle-specific types are imported lazily inside that helper
to avoid an import cycle (matching the deferred-import pattern used elsewhere in
the codebase).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from statistics import median
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np

from kinematics.core.constants import SOLVE_ACCEPT_RESIDUAL
from kinematics.core.enums import PointID
from kinematics.core.point_ref import PointKey, PointRef, Side

if TYPE_CHECKING:
    from kinematics.solver import SolverInfo
    from kinematics.state import SuspensionState
    from kinematics.suspensions.base import Suspension
    from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension


# Continuity check: a step displacement of a single free point is flagged when it
# exceeds both an absolute floor and a multiple of that point's own median step
# displacement. The floor prevents flagging tiny numerical jitter; the median
# multiple adapts to sweeps that legitimately move points in large, smooth steps.
CONTINUITY_ABS_FLOOR_MM: float = 5.0
CONTINUITY_MEDIAN_FACTOR: float = 4.0

# Transmission (toggle) margin: for a link driving a lever that rotates about an
# axis, the margin is |cos(theta)| between the link direction and the tangent to
# the lever's circular path at the pickup. margin = 1 drives rotation ideally;
# margin -> 0 is a dead-centre / toggle singularity (the link is purely radial and
# can no longer produce torque -- the joint is topping out). Warn below this
# value, which corresponds to the link being within ~8.6 degrees of the radial
# (toggle) direction.
TRANSMISSION_MARGIN_WARN: float = 0.15


@dataclass(frozen=True)
class DiagnosticIssue:
    """
    A single diagnostic finding about a solved sweep.

    Attributes:
        step: Sweep step index the issue is about, or ``None`` for a sweep-wide
            issue.
        category: One of ``"convergence"``, ``"residual"``, ``"jump"``,
            ``"chirality"``, ``"transmission"``.
        severity: ``"error"`` or ``"warning"``.
        message: Human-readable, self-contained description.
        value: The salient numeric value (residual, displacement, margin, ...),
            or ``None`` when not applicable.
    """

    step: int | None
    category: str
    severity: str
    message: str
    value: float | None


@dataclass
class SweepDiagnostics:
    """
    The collected diagnostics for one solved sweep.

    Attributes:
        issues: All findings, in the order the checks produced them.
    """

    issues: list[DiagnosticIssue]

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues (warnings are allowed)."""
        return not self.errors

    @property
    def warnings(self) -> list[DiagnosticIssue]:
        """All warning-severity issues."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def errors(self) -> list[DiagnosticIssue]:
        """All error-severity issues."""
        return [i for i in self.issues if i.severity == "error"]


def diagnose_sweep(
    suspension: "Suspension",
    states: list["SuspensionState"],
    stats: list["SolverInfo"],
) -> SweepDiagnostics:
    """
    Run all diagnostic checks over a completed sweep.

    Args:
        suspension: The suspension that was solved (for topology / design geometry).
        states: The solved states, one per sweep step.
        stats: The solver stats, one per sweep step (aligned with ``states``).

    Returns:
        A :class:`SweepDiagnostics` aggregating every finding.
    """
    issues: list[DiagnosticIssue] = []
    issues.extend(_check_convergence_and_residual(stats))
    issues.extend(_check_continuity(suspension, states))
    issues.extend(_check_chirality(suspension, states))
    issues.extend(_check_transmission(suspension, states))
    return SweepDiagnostics(issues=issues)


# ----------------------------------------------------------------------
# Convergence / residual
# ----------------------------------------------------------------------


def _check_convergence_and_residual(
    stats: list["SolverInfo"],
) -> list[DiagnosticIssue]:
    """
    Flag any step that did not converge or whose residual is unacceptable.

    Both are error-severity. The residual check is belt-and-braces: the solver
    already raises above :data:`SOLVE_ACCEPT_RESIDUAL`, but a programmatic caller
    may loosen ``SolverConfig.residual_tolerance``, so re-checking here keeps the
    report honest against the canonical acceptance threshold.
    """
    issues: list[DiagnosticIssue] = []
    for step, info in enumerate(stats):
        if not info.converged:
            issues.append(
                DiagnosticIssue(
                    step=step,
                    category="convergence",
                    severity="error",
                    message=f"Step {step} did not converge.",
                    value=None,
                )
            )
        if info.max_residual > SOLVE_ACCEPT_RESIDUAL:
            issues.append(
                DiagnosticIssue(
                    step=step,
                    category="residual",
                    severity="error",
                    message=(
                        f"Step {step} residual {info.max_residual:.6g} exceeds the "
                        f"acceptance tolerance {SOLVE_ACCEPT_RESIDUAL:.6g} "
                        "(infeasible target / kinematic lock-out)."
                    ),
                    value=info.max_residual,
                )
            )
    return issues


# ----------------------------------------------------------------------
# Continuity / branch snaps
# ----------------------------------------------------------------------


def _check_continuity(
    suspension: "Suspension",
    states: list["SuspensionState"],
) -> list[DiagnosticIssue]:
    """
    Flag discontinuous per-point jumps between consecutive steps.

    For each free point, the per-step displacement is compared against
    ``max(CONTINUITY_ABS_FLOOR_MM, CONTINUITY_MEDIAN_FACTOR * median_nonzero)``
    where the median is over that point's own nonzero step displacements. A step
    exceeding the threshold is a warning: it usually signals the solver snapped
    onto a different assembly branch mid-sweep.
    """
    issues: list[DiagnosticIssue] = []
    if len(states) < 2:
        return issues

    free_keys = list(suspension.free_points())
    for key in free_keys:
        displacements = _point_step_displacements(states, key)
        if not displacements:
            continue
        nonzero = [d for d in displacements if d > 0.0]
        med = median(nonzero) if nonzero else 0.0
        threshold = max(CONTINUITY_ABS_FLOOR_MM, CONTINUITY_MEDIAN_FACTOR * med)
        for i, disp in enumerate(displacements):
            # displacements[i] is the move from state i to state i+1.
            step = i + 1
            if disp > threshold:
                name = getattr(key, "name", str(key))
                issues.append(
                    DiagnosticIssue(
                        step=step,
                        category="jump",
                        severity="warning",
                        message=(
                            f"Point '{name}' jumped {disp:.3g} mm from step {i} to "
                            f"step {step} (threshold {threshold:.3g} mm); possible "
                            "branch snap."
                        ),
                        value=disp,
                    )
                )
    return issues


def _point_step_displacements(
    states: list["SuspensionState"],
    key: PointKey,
) -> list[float]:
    """Per-step Euclidean displacement of one point across the sweep."""
    displacements: list[float] = []
    prev = states[0].positions.get(key)
    for state in states[1:]:
        cur = state.positions.get(key)
        if prev is None or cur is None:
            displacements.append(0.0)
        else:
            displacements.append(float(np.linalg.norm(cur.data - prev.data)))
        prev = cur
    return displacements


# ----------------------------------------------------------------------
# Chirality
# ----------------------------------------------------------------------


def _check_chirality(
    suspension: "Suspension",
    states: list["SuspensionState"],
) -> list[DiagnosticIssue]:
    """
    Flag any state whose rocker triple product has flipped sign from design.

    The rocker droplink is fixed by distances alone, which admit a mirror-image
    (reflected) branch. Recompute the signed scalar triple product
    ``(axis_front, axis_rear, pushrod_inboard, droplink_rocker)`` per state; a
    sign different from the design sign means the rigid rocker body has inverted.
    Error severity. This is belt-and-braces on top of the
    ``ScalarTripleProductConstraint`` now built into the rocker.
    """
    issues: list[DiagnosticIssue] = []
    design = suspension.initial_state()

    for label, key_of, corner in _iter_rocker_corners(suspension):
        if not corner.has_droplink:
            continue
        design_triple = _rocker_triple(design.positions, key_of)
        design_sign = np.sign(design_triple)
        if design_sign == 0.0:
            continue
        prefix = f"{label} " if label else ""
        for step, state in enumerate(states):
            triple = _rocker_triple(state.positions, key_of)
            if np.sign(triple) != design_sign and triple != 0.0:
                issues.append(
                    DiagnosticIssue(
                        step=step,
                        category="chirality",
                        severity="error",
                        message=(
                            f"{prefix}rocker handedness inverted at step {step}: "
                            f"triple product {triple:.6g} has the opposite sign to "
                            f"the design value {design_triple:.6g} (rocker folded "
                            "onto its mirror branch)."
                        ),
                        value=triple,
                    )
                )
    return issues


def _rocker_triple(
    positions: dict,
    key_of: Callable[[PointID], PointKey],
) -> float:
    """Signed scalar triple product of the rocker's four defining points."""
    from kinematics.core.vector_utils.geometric import compute_scalar_triple_product

    axis_front = positions[key_of(PointID.ROCKER_AXIS_FRONT)]
    axis_rear = positions[key_of(PointID.ROCKER_AXIS_REAR)]
    pushrod_in = positions[key_of(PointID.PUSHROD_INBOARD)]
    droplink = positions[key_of(PointID.DROPLINK_ROCKER)]
    return compute_scalar_triple_product(
        axis_rear - axis_front,
        pushrod_in - axis_front,
        droplink - axis_front,
    )


# ----------------------------------------------------------------------
# Transmission (toggle) margin
# ----------------------------------------------------------------------


def _check_transmission(
    suspension: "Suspension",
    states: list["SuspensionState"],
) -> list[DiagnosticIssue]:
    """
    Flag link/lever pairs approaching their transmission (toggle) singularity.

    For each driven circle joint, the margin is the absolute cosine between the
    driving link direction and the tangent to the driven point's circular path.
    As the margin drops toward zero the link becomes radial and the joint tops
    out. The checks are:

    - pushrod vs the tangent at ``PUSHROD_INBOARD`` about the rocker axis
      (whenever a rocker is present);
    - the rocker->ARB droplink vs the tangent at ``DROPLINK_ROCKER`` about the
      rocker axis (axle with ARB only);
    - the rocker->ARB droplink vs the tangent at ``DROPLINK_ARB`` about the ARB
      axis (axle with ARB only).

    Each sub-threshold state is a warning naming the joint, side, step, and
    margin.
    """
    issues: list[DiagnosticIssue] = []
    has_arb = _axle_has_arb(suspension)
    arb_axis_keys = _arb_axis_keys(suspension) if has_arb else None

    for label, key_of, corner in _iter_rocker_corners(suspension):
        if not corner.has_rocker:
            continue
        prefix = f"{label} " if label else ""

        for step, state in enumerate(states):
            pos = state.positions
            axis_front = pos[key_of(PointID.ROCKER_AXIS_FRONT)]
            axis_rear = pos[key_of(PointID.ROCKER_AXIS_REAR)]
            axis_dir = axis_rear.data - axis_front.data

            # (a) Pushrod driving the rocker at PUSHROD_INBOARD.
            pushrod = (
                pos[key_of(PointID.PUSHROD_OUTBOARD)].data
                - pos[key_of(PointID.PUSHROD_INBOARD)].data
            )
            margin_a = _transmission_margin(
                pos[key_of(PointID.PUSHROD_INBOARD)].data,
                axis_front.data,
                axis_dir,
                pushrod,
            )
            issues.extend(
                _maybe_transmission_issue(
                    margin_a, step, f"{prefix}pushrod @ PUSHROD_INBOARD"
                )
            )

            if has_arb and corner.has_droplink and arb_axis_keys is not None:
                droplink = (
                    pos[key_of(PointID.DROPLINK_ARB)].data
                    - pos[key_of(PointID.DROPLINK_ROCKER)].data
                )
                # (b) Droplink vs tangent at DROPLINK_ROCKER about the rocker axis.
                margin_b = _transmission_margin(
                    pos[key_of(PointID.DROPLINK_ROCKER)].data,
                    axis_front.data,
                    axis_dir,
                    droplink,
                )
                issues.extend(
                    _maybe_transmission_issue(
                        margin_b, step, f"{prefix}droplink @ DROPLINK_ROCKER"
                    )
                )

                # (c) Droplink vs tangent at DROPLINK_ARB about the ARB axis.
                arb_a_key, arb_b_key = arb_axis_keys
                arb_axis_a = pos[arb_a_key].data
                arb_axis_dir = pos[arb_b_key].data - arb_axis_a
                margin_c = _transmission_margin(
                    pos[key_of(PointID.DROPLINK_ARB)].data,
                    arb_axis_a,
                    arb_axis_dir,
                    droplink,
                )
                issues.extend(
                    _maybe_transmission_issue(
                        margin_c, step, f"{prefix}droplink @ DROPLINK_ARB"
                    )
                )
    return issues


def _maybe_transmission_issue(
    margin: float | None,
    step: int,
    joint: str,
) -> list[DiagnosticIssue]:
    """Build a transmission warning when the margin is below threshold."""
    if margin is None or margin >= TRANSMISSION_MARGIN_WARN:
        return []
    angle_from_toggle = degrees(acos(min(1.0, max(0.0, margin))))
    return [
        DiagnosticIssue(
            step=step,
            category="transmission",
            severity="warning",
            message=(
                f"{joint} near transmission singularity at step {step}: margin "
                f"{margin:.3g} below {TRANSMISSION_MARGIN_WARN:.3g} "
                f"(~{90.0 - angle_from_toggle:.1f} deg from toggle); the joint is "
                "topping out."
            ),
            value=margin,
        )
    ]


def _transmission_margin(
    driven_point: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    link: np.ndarray,
) -> float | None:
    """
    |cos| between a driving link and the tangent to a driven circular path.

    ``tangent = normalize(axis_dir x radius_perp)`` where ``radius_perp`` is the
    component of ``driven_point - axis_point`` perpendicular to ``axis_dir``.
    Returns ``None`` if any vector degenerates (zero-length), which is itself a
    singular configuration but has no defined margin.
    """
    axis_norm = float(np.linalg.norm(axis_dir))
    link_norm = float(np.linalg.norm(link))
    if axis_norm == 0.0 or link_norm == 0.0:
        return None

    axis_unit = axis_dir / axis_norm
    radius = driven_point - axis_point
    radius_perp = radius - axis_unit * float(np.dot(radius, axis_unit))
    if float(np.linalg.norm(radius_perp)) == 0.0:
        return None

    tangent = np.cross(axis_unit, radius_perp)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm == 0.0:
        return None

    tangent_unit = tangent / tangent_norm
    link_unit = link / link_norm
    return float(abs(np.dot(link_unit, tangent_unit)))


# ----------------------------------------------------------------------
# Corner iteration (key-type agnostic)
# ----------------------------------------------------------------------


def _iter_rocker_corners(
    suspension: "Suspension",
) -> Iterator[tuple[str, Callable[[PointID], PointKey], "DoubleWishboneSuspension"]]:
    """
    Yield ``(label, key_of, corner)`` for each rocker-bearing corner.

    ``key_of(pid)`` maps a plain ``PointID`` to the concrete key used in this
    model's state -- identity for a single corner, ``PointRef(side, pid)`` for the
    axle. ``label`` is the side name for the axle, empty for a single corner.
    ``corner`` is the corner suspension exposing ``has_rocker`` /
    ``has_droplink``.

    Axle types are imported lazily here to avoid an import cycle.
    """
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

    if isinstance(suspension, DoubleWishboneAxleSuspension):
        for side, corner in suspension.corners.items():
            yield (
                side.name.lower(),
                lambda pid, s=side: PointRef(s, pid),
                corner,
            )
    else:
        from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension

        if isinstance(suspension, DoubleWishboneSuspension):
            yield ("", lambda pid: pid, suspension)


def _axle_has_arb(suspension: "Suspension") -> bool:
    """True when the suspension is an axle with a complete ARB group."""
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension

    return isinstance(suspension, DoubleWishboneAxleSuspension) and suspension.has_arb


def _arb_axis_keys(
    suspension: "Suspension",
) -> tuple[PointKey, PointKey]:
    """The two CENTER-side ARB axis point keys (axle only)."""
    return (
        PointRef(Side.CENTER, PointID.ARB_AXIS_A),
        PointRef(Side.CENTER, PointID.ARB_AXIS_B),
    )
