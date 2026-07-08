"""
YAML file loading.

Thin wrappers that read a YAML file, validate it into a structured spec
(``kinematics.schema``), and hand off to the core construction/expansion
functions. All file-format concern lives here; nothing inboard of this module
touches the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kinematics.core.types import SweepConfig
from kinematics.schema.geometry import parse_geometry_spec
from kinematics.schema.sweep import SweepSpec, build_sweep_config
from kinematics.suspensions.build import build_suspension

if TYPE_CHECKING:
    from kinematics.suspensions.base import Suspension


def _read_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{kind} file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing {kind} file: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"{kind} file must contain a YAML mapping: {path}")
    return data


def load_geometry(path: Path) -> "Suspension":
    """
    Load a suspension from a YAML geometry file.

    The file is validated into a ``GeometrySpec`` (dispatched on its ``type``
    key) and built into the matching suspension class.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not valid YAML or fails spec validation.
    """
    data = _read_yaml_mapping(path, "Geometry")
    spec = parse_geometry_spec(data)
    return build_suspension(spec)


def load_sweep(
    path: Path,
    suspension: "Suspension | None" = None,
) -> SweepConfig:
    """
    Load a sweep from a YAML sweep file.

    The file is validated into a ``SweepSpec`` and expanded into an executable
    ``SweepConfig``.

    Args:
        path: Path to the YAML sweep file.
        suspension: Optional suspension used to resolve each target's
            ``(point, side)`` into a concrete point key. When omitted, targets
            must not specify a ``side`` (single-corner behavior).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not valid YAML or fails spec validation.
    """
    data = _read_yaml_mapping(path, "Sweep")
    try:
        spec = SweepSpec.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid sweep specification: {e}") from e
    return build_sweep_config(spec, suspension)
