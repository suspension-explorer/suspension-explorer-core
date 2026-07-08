"""
Structured sweep specification.

``SweepSpec`` is the validated, transport-agnostic description of a sweep: the
same model backs YAML files (via ``kinematics.io``) and structured API
requests. ``build_sweep_config`` expands a validated spec into the executable
``SweepConfig`` consumed by the solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from kinematics.core.enums import Axis, TargetPositionMode
from kinematics.core.geometry import Direction3, extract_array
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    SweepConfig,
    WorldAxisSystem,
)
from kinematics.schema.coercion import (
    CIAxis,
    CIPointID,
    CISide,
    CITargetPositionMode,
)

if TYPE_CHECKING:
    from kinematics.suspensions.base import Suspension

# Shared axis <-> unit vector mapping.
AXIS_VECTORS: dict[Axis, np.ndarray] = {
    Axis.X: WorldAxisSystem.X.data,
    Axis.Y: WorldAxisSystem.Y.data,
    Axis.Z: WorldAxisSystem.Z.data,
}


def vector_to_axis(vec: np.ndarray) -> Axis | None:
    """
    Return the Axis if vec matches a principal axis, else None.
    """
    vec_data = extract_array(vec)
    for axis, axis_vec in AXIS_VECTORS.items():
        if np.allclose(vec_data, axis_vec):
            return axis
    return None


class DirectionSpec(BaseModel):
    """Specification for a target direction (either axis or custom vector)."""

    model_config = ConfigDict(frozen=True)

    axis: CIAxis | None = None
    vector: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_exactly_one(self) -> "DirectionSpec":
        if (self.axis is None) == (self.vector is None):
            raise ValueError("Specify exactly one of 'axis' or 'vector'")
        return self

    def to_unit_vector(self) -> np.ndarray:
        """Convert to a normalized 3D unit vector."""
        if self.axis is not None:
            return AXIS_VECTORS[self.axis]

        vec = np.asarray(self.vector, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(f"Vector must be 3D, got shape {vec.shape}")

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero")
        return vec / norm


class TargetSpec(BaseModel):
    """Specification for a single sweep target dimension."""

    model_config = ConfigDict(frozen=True)

    point: CIPointID
    direction: DirectionSpec
    name: str | None = None
    side: CISide | None = None
    mode: CITargetPositionMode = TargetPositionMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None

    def expand_values(self, default_steps: int | None) -> list[float]:
        """Expand this target into concrete values."""
        if self.values is not None:
            return [float(v) for v in self.values]

        if self.start is None or self.stop is None:
            raise ValueError(
                f"Target '{self.name or self.point.name}': "
                "must specify either 'values' or both 'start' and 'stop'"
            )

        if default_steps is None:
            raise ValueError(
                f"Target '{self.name or self.point.name}': "
                "no 'steps' count available (specify at target or file level)"
            )

        return list(
            np.linspace(float(self.start), float(self.stop), int(default_steps))
        )


class SweepSpec(BaseModel):
    """
    Validated sweep specification.

    This is the structured wire/file format for sweeps: a shared step count and
    a list of target dimensions. Expand it into an executable ``SweepConfig``
    with :func:`build_sweep_config`.
    """

    model_config = ConfigDict(frozen=True)

    version: int = 1
    steps: int | None = None
    targets: list[TargetSpec]

    @model_validator(mode="after")
    def check_version(self) -> "SweepSpec":
        if self.version != 1:
            raise ValueError(f"Unsupported sweep version: {self.version}")
        return self

    @property
    def n_steps(self) -> int:
        """
        Number of solved steps this spec expands to.

        Each target expands independently (explicit values win over the
        shared step count); the sweep runs for the longest expansion.
        """
        lengths = [len(target.expand_values(self.steps)) for target in self.targets]
        return max(lengths) if lengths else 0


def build_sweep_config(
    spec: SweepSpec,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """
    Expand a validated SweepSpec into an executable SweepConfig.

    Args:
        spec: A validated sweep specification.
        suspension: Optional suspension used to resolve each target's
            ``(point, side)`` into a concrete point key. When omitted, targets
            must not specify a ``side``.

    Returns:
        SweepConfig ready for use with the solver.

    Raises:
        ValueError: If the targets expand to unequal lengths, or a target sets
            ``side`` without a suspension context.
    """
    # Expand values and verify equal lengths.
    target_sequences = [t.expand_values(spec.steps) for t in spec.targets]
    lengths = {len(seq) for seq in target_sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"All targets must have the same length, got: {sorted(lengths)}"
        )

    # Build per-dimension target lists.
    sweep_dimensions: list[list[PointTarget]] = []
    for target_spec, values in zip(spec.targets, target_sequences):
        unit_vec = target_spec.direction.to_unit_vector()
        axis = vector_to_axis(unit_vec)
        direction: PointTargetAxis | PointTargetVector
        if axis:
            direction = PointTargetAxis(axis)
        else:
            direction = PointTargetVector(Direction3(unit_vec))

        # Resolve the concrete point key through the suspension when provided.
        if suspension is not None:
            point_key = suspension.resolve_target_key(
                target_spec.point, target_spec.side
            )
        else:
            if target_spec.side is not None:
                raise ValueError(
                    f"Sweep target for '{target_spec.point.name}' specifies a "
                    "'side', which requires a suspension context to resolve."
                )
            point_key = target_spec.point

        targets = [
            PointTarget(
                point_id=point_key,
                direction=direction,
                value=val,
                mode=target_spec.mode,
            )
            for val in values
        ]
        sweep_dimensions.append(targets)

    return SweepConfig(sweep_dimensions)
