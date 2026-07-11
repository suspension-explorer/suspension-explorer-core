"""Corner derivative metrics computed from solution-manifold tangents."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from kinematics.core.dual import DualScalar, seed_positions_with_tangent
from kinematics.core.enums import Axis, PointID
from kinematics.core.point_ref import PointKey
from kinematics.metrics import kernels
from kinematics.sensitivity import TangentField
from kinematics.targets import resolve_target

if TYPE_CHECKING:
    from kinematics.state import SuspensionState
    from kinematics.suspensions.base import Suspension

RateRow = OrderedDict[str, float | None]

# A target must be within about 2.5 degrees of the expected world axis.
_AXIS_ALIGNMENT_TOLERANCE = 0.999


def _axis_alignment(target, axis: Axis) -> float | None:
    """Return signed target alignment when it is parallel to an axis."""
    direction = resolve_target(target.direction).data
    unit = np.zeros(3)
    unit[int(axis)] = 1.0
    alignment = float(np.dot(direction, unit))
    if abs(alignment) < _AXIS_ALIGNMENT_TOLERANCE:
        return None
    return alignment


def _find_bump_driver(
    tangents: Sequence[TangentField],
) -> tuple[TangentField, float] | None:
    """Find a wheel-center target aligned with the world Z axis."""
    for field in tangents:
        if field.target.point_id != PointID.WHEEL_CENTER:
            continue
        alignment = _axis_alignment(field.target, Axis.Z)
        if alignment is not None:
            return field, alignment
    return None


def _find_rack_driver(
    tangents: Sequence[TangentField],
) -> tuple[TangentField, float] | None:
    """Find a trackrod-inboard target aligned with the world Y axis."""
    for field in tangents:
        if field.target.point_id != PointID.TRACKROD_INBOARD:
            continue
        alignment = _axis_alignment(field.target, Axis.Y)
        if alignment is not None:
            return field, alignment
    return None


def _rate(kernel_result: object) -> float:
    """Extract a scalar derivative from a dual-safe metric kernel."""
    assert isinstance(kernel_result, DualScalar)
    return float(kernel_result.deriv)


def _scaled(
    velocities: Mapping[PointKey, np.ndarray], scale: float
) -> dict[PointKey, np.ndarray]:
    """Fold the target direction sign into a tangent velocity field."""
    return {point: scale * velocity for point, velocity in velocities.items()}


def _corner_bump_rates(
    positions: Mapping[PointKey, object],
    velocities: Mapping[PointKey, np.ndarray],
    side_sign: float,
    suspension: "Suspension",
) -> RateRow:
    """Compute rates per +1 mm of upward wheel-center travel."""
    duals = seed_positions_with_tangent(positions, velocities)
    row: RateRow = OrderedDict()

    row["camber_gain_deg_per_mm"] = _rate(kernels.camber_deg(duals, side_sign))
    row["bump_steer_deg_per_mm"] = _rate(kernels.toe_deg(duals, side_sign))
    row["caster_gain_deg_per_mm"] = _rate(kernels.caster_deg(duals))
    row["kpi_gain_deg_per_mm"] = _rate(kernels.kpi_deg(duals, side_sign))

    lateral_speed = _rate(
        kernels.coordinate(duals, PointID.CONTACT_PATCH_CENTER, Axis.Y)
    )
    row["half_track_rate_mm_per_mm"] = side_sign * lateral_speed

    longitudinal_speed = _rate(
        kernels.coordinate(duals, PointID.CONTACT_PATCH_CENTER, Axis.X)
    )
    row["wheel_recession_rate_mm_per_mm"] = -longitudinal_speed

    if suspension.has_strut:
        # Installation ratio is damper compression per millimetre of bump.
        row["damper_motion_ratio"] = -_rate(kernels.strut_length_mm(duals))

    return row


def _corner_rack_rates(
    positions: Mapping[PointKey, object],
    velocities: Mapping[PointKey, np.ndarray],
    side_sign: float,
) -> RateRow:
    """Compute rates per +1 mm of world-Y rack travel."""
    duals = seed_positions_with_tangent(positions, velocities)
    row: RateRow = OrderedDict()
    row["toe_vs_rack_deg_per_mm"] = _rate(kernels.toe_deg(duals, side_sign))
    row["camber_vs_rack_deg_per_mm"] = _rate(kernels.camber_deg(duals, side_sign))
    return row


def compute_corner_rate_metrics(
    state: "SuspensionState",
    suspension: "Suspension",
    tangents: Sequence[TangentField],
) -> RateRow:
    """Compute derivative metrics for the drivers present in a corner sweep."""
    side_sign = (
        -1.0 if float(state.positions[PointID.AXLE_OUTBOARD][Axis.Y]) < 0 else 1.0
    )
    row: RateRow = OrderedDict()

    bump = _find_bump_driver(tangents)
    if bump is not None:
        field, alignment = bump
        row.update(
            _corner_bump_rates(
                state.positions,
                _scaled(field.velocities, alignment),
                side_sign,
                suspension,
            )
        )

    rack = _find_rack_driver(tangents)
    if rack is not None:
        field, alignment = rack
        row.update(
            _corner_rack_rates(
                state.positions,
                _scaled(field.velocities, alignment),
                side_sign,
            )
        )

    return row


__all__ = ["RateRow", "compute_corner_rate_metrics"]
