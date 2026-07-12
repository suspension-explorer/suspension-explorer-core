"""YAML adapter for validated sweep specifications."""

from pathlib import Path

import yaml

from kinematics.core.types import SweepConfig
from kinematics.schema.sweep import SweepSpec, build_sweep_config
from kinematics.suspensions.base import Suspension


def parse_sweep_file(
    path: Path,
    suspension: Suspension | None = None,
) -> SweepConfig:
    """Load, validate, and expand a sweep YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sweep file not found: {path}")
    except yaml.YAMLError as error:
        raise ValueError(f"Error parsing YAML: {error}") from error

    if not isinstance(raw_data, dict):
        raise ValueError("Sweep file must contain a YAML mapping")
    try:
        spec = SweepSpec.model_validate(raw_data)
    except Exception as error:
        raise ValueError(f"Invalid sweep specification: {error}") from error
    return build_sweep_config(spec, suspension)
