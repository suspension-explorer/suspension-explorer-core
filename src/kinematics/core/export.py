"""
Pure position flattening at transport boundaries.
"""

from collections.abc import Mapping, Sequence
from enum import Enum

from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey, point_key_name


class StandardColumn(Enum):
    """Standard non-metric columns written at flat result boundaries."""

    STEP_INDEX = "step_index"
    SOLVER_CONVERGED = "solver_converged"
    SOLVER_NFEV = "solver_nfev"
    SOLVER_MAX_RESIDUAL = "solver_max_residual"


def flatten_positions(
    positions: Mapping[PointKey, object],
    output_points: Sequence[PointKey],
) -> dict[str, tuple[float, float, float]]:
    """Flatten selected typed positions to public point names and tuples."""
    flattened: dict[str, tuple[float, float, float]] = {}
    for point in output_points:
        position = positions.get(point)
        if position is None:
            continue
        raw = extract_array(position)
        flattened[point_key_name(point)] = (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]),
        )
    return flattened
