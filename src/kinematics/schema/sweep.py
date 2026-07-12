"""Validated, transport-independent sweep specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from kinematics.core.enums import Axis, TargetPositionMode
from kinematics.core.geometry import Direction3, extract_array
from kinematics.core.point_ref import Side
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

AXIS_VECTORS: dict[Axis, np.ndarray] = {
    Axis.X: WorldAxisSystem.X.data,
    Axis.Y: WorldAxisSystem.Y.data,
    Axis.Z: WorldAxisSystem.Z.data,
}


def vector_to_axis(vector: np.ndarray) -> Axis | None:
    """Return the principal axis represented by a vector, if any."""
    vector_data = extract_array(vector)
    for axis, axis_vector in AXIS_VECTORS.items():
        if np.allclose(vector_data, axis_vector):
            return axis
    return None


class DirectionSpec(BaseModel):
    """Target direction specified by either an axis or a custom vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: CIAxis | None = None
    vector: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_exactly_one(self) -> "DirectionSpec":
        if (self.axis is None) == (self.vector is None):
            raise ValueError("Specify exactly one of 'axis' or 'vector'")
        return self

    def to_unit_vector(self) -> np.ndarray:
        """Convert this specification to a normalized three-dimensional vector."""
        if self.axis is not None:
            return AXIS_VECTORS[self.axis]
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(f"Vector must be 3D, got shape {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero")
        return vector / norm


class TargetSpec(BaseModel):
    """One target dimension in a suspension sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: CIPointID
    direction: DirectionSpec
    name: str | None = None
    side: CISide | None = None
    mode: CITargetPositionMode = TargetPositionMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None

    @model_validator(mode="after")
    def check_side(self) -> "TargetSpec":
        if self.side == Side.CENTER:
            raise ValueError("Sweep target side must be 'left' or 'right'.")
        return self

    def expand_values(self, default_steps: int | None) -> list[float]:
        """Expand explicit values or a start-stop range."""
        if self.values is not None:
            return [float(value) for value in self.values]
        if self.start is None or self.stop is None:
            raise ValueError(
                f"Target '{self.name or self.point.name}': must specify either "
                "'values' or both 'start' and 'stop'"
            )
        if default_steps is None:
            raise ValueError(
                f"Target '{self.name or self.point.name}': no 'steps' count "
                "available (specify at target or file level)"
            )
        return list(np.linspace(float(self.start), float(self.stop), default_steps))


class SweepSpec(BaseModel):
    """Validated sweep file or API specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    steps: int | None = None
    targets: list[TargetSpec]

    @model_validator(mode="after")
    def check_version(self) -> "SweepSpec":
        if self.version != 1:
            raise ValueError(f"Unsupported sweep version: {self.version}")
        return self


def build_sweep_config(
    spec: SweepSpec,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """Expand a validated sweep and resolve optional side-qualified targets."""
    target_sequences = [target.expand_values(spec.steps) for target in spec.targets]
    lengths = {len(sequence) for sequence in target_sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"All targets must have the same length, got: {sorted(lengths)}"
        )

    dimensions: list[list[PointTarget]] = []
    for target_spec, values in zip(spec.targets, target_sequences):
        unit_vector = target_spec.direction.to_unit_vector()
        axis = vector_to_axis(unit_vector)
        direction: PointTargetAxis | PointTargetVector
        if axis is not None:
            direction = PointTargetAxis(axis)
        else:
            direction = PointTargetVector(Direction3(unit_vector))

        if suspension is not None:
            point_key = suspension.resolve_target_key(
                target_spec.point, target_spec.side
            )
            if point_key not in suspension.initial_state().positions:
                raise ValueError(
                    f"Sweep target point '{point_key.name}' is not present in "
                    f"suspension type '{suspension.TYPE_KEY}'."
                )
        else:
            if target_spec.side is not None:
                raise ValueError(
                    f"Sweep target for '{target_spec.point.name}' specifies a "
                    "'side', which requires a suspension context to resolve."
                )
            point_key = target_spec.point

        dimensions.append(
            [
                PointTarget(
                    point_id=point_key,
                    direction=direction,
                    value=value,
                    mode=target_spec.mode,
                )
                for value in values
            ]
        )
    return SweepConfig(dimensions)
