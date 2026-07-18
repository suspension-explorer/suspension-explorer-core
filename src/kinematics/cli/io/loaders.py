"""YAML file loading for validated suspension geometry."""

from pathlib import Path
from typing import Any

import yaml

from kinematics.core.input import build_suspension
from kinematics.core.suspensions.base import Suspension


def _read_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{kind} file not found: {path}")
    except yaml.YAMLError as error:
        raise ValueError(f"Error parsing {kind.lower()} file: {error}") from error

    if data is None:
        raise ValueError(f"{kind} file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"{kind} file must contain a YAML mapping: {path}")
    return data


def load_geometry(path: Path) -> Suspension:
    """Load, validate, and build a suspension from a YAML geometry file."""
    data = _read_yaml_mapping(path, "Geometry")
    return build_suspension(data)
