"""Parse sweep configuration files into executable sweep specifications."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kinematics.core.enums import Axis, TargetPositionMode
from kinematics.core.point_ref import Side
from kinematics.core.types import (
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    SweepConfig,
    WorldAxisSystem,
)
from kinematics.io.validation import (
    CIAxis,
    CIPointID,
    CITargetPositionMode,
    coerce_enum,
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
    from kinematics.core.geometry import extract_array

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
    side: Side | None = None
    mode: CITargetPositionMode = TargetPositionMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None

    @field_validator("side", mode="before")
    @classmethod
    def coerce_side(cls, v: object) -> Side | None:
        """Coerce a case-insensitive 'left'/'right' string to a Side."""
        if v is None:
            return None
        return coerce_enum(Side, v)  # type: ignore[arg-type]

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


class SweepFile(BaseModel):
    """Schema for sweep configuration files."""

    model_config = ConfigDict(frozen=True)

    version: Annotated[int, "Schema version (currently only 1 supported)"]
    steps: int | None = None
    targets: list[TargetSpec]

    @field_validator("version")
    @classmethod
    def check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported sweep version: {v}")
        return v


def parse_sweep_file(
    path: Path,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """
    Parse a sweep configuration file into a SweepConfig.

    Args:
        path: Path to the YAML sweep configuration file.
        suspension: Optional suspension used to resolve each target's
            ``(point, side)`` into a concrete point key. When omitted, targets
            must not specify a ``side`` (single-corner behaviour).

    Returns:
        SweepConfig ready for use with the solver.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid or contains errors.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sweep file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}") from e

    if not isinstance(raw_data, dict):
        raise ValueError("Sweep file must contain a YAML mapping")

    try:
        file_spec = SweepFile.model_validate(raw_data)
    except Exception as e:
        raise ValueError(f"Invalid sweep specification: {e}") from e

    return build_sweep_config(file_spec, suspension)


def build_sweep_config(
    file_spec: SweepFile,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """
    Expand a validated SweepFile spec into an executable SweepConfig.

    This is the value-expansion half of sweep loading, split out so callers that
    already hold a validated spec (e.g. a structured API request) can build a
    SweepConfig without going through a YAML file.

    Args:
        file_spec: A validated sweep specification.
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
    target_sequences = [t.expand_values(file_spec.steps) for t in file_spec.targets]
    lengths = {len(seq) for seq in target_sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"All targets must have the same length, got: {sorted(lengths)}"
        )

    # Build per-dimension target lists.
    sweep_dimensions: list[list[PointTarget]] = []
    for target_spec, values in zip(file_spec.targets, target_sequences):
        unit_vec = target_spec.direction.to_unit_vector()
        axis = vector_to_axis(unit_vec)
        if axis:
            direction = PointTargetAxis(axis)
        else:
            from kinematics.core.geometry import Direction3

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
