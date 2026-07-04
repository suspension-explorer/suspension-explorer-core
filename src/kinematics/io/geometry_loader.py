"""
YAML geometry loader.

This module loads suspension geometry from YAML files and returns instantiated
Suspension subclass instances directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from kinematics.core.enums import PointID, ShimType, Units
from kinematics.core.geometry import Point3
from kinematics.io.validation import coerce_enum, coerce_point3
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.config.settings import SuspensionConfig
from kinematics.suspensions.registry import get_suspension_class, list_supported_types


def load_geometry(file_path: Path) -> Suspension:
    """
    Load suspension geometry from a YAML file.

    Returns the appropriate Suspension subclass instance directly.

    The YAML format is:

        type: double_wishbone
        name: "My Suspension"
        version: "1.0.0"
        units: MILLIMETERS

        hardpoints:
          LOWER_WISHBONE_INBOARD_FRONT: [x, y, z]
          LOWER_WISHBONE_INBOARD_REAR: [x, y, z]
          ...

        config:
          steered: true
          wheel:
            offset: 0  # ET convention: positive is inboard.
            tire: {...}
          camber_shim:
            shim_face_point_a: {x: ..., y: ..., z: ...}
            shim_face_point_b: {x: ..., y: ..., z: ...}
            shim_face_normal: {x: ..., y: ..., z: ...}
            design_thickness: 30.0
            setup_thickness: 30.0

    Args:
        file_path: Path to the YAML geometry file.

    Returns:
        Instantiated Suspension subclass (e.g., DoubleWishboneSuspension).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If validation fails or type is unsupported.
    """
    try:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Geometry file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing geometry file: {e}") from e

    if yaml_data is None:
        raise ValueError("Geometry file is empty.")

    if "type" not in yaml_data:
        raise ValueError("Geometry type not specified in file")

    type_key = yaml_data.pop("type").lower()

    suspension_class = get_suspension_class(type_key)
    if suspension_class is None:
        available = list_supported_types()
        raise ValueError(
            f"Unsupported geometry type: '{type_key}'. "
            f"Supported types: {', '.join(available)}"
        )

    return suspension_class.from_yaml_data(yaml_data)


def load_suspension(
    yaml_data: dict[str, Any],
    suspension_class: type[Suspension],
) -> Suspension:
    """
    Load and instantiate a suspension from parsed YAML data.

    Args:
        yaml_data: Parsed YAML data (with 'type' already removed).
        suspension_class: The Suspension subclass to instantiate.

    Returns:
        Instantiated suspension.

    Raises:
        ValueError: If validation fails.
    """
    # Validate and parse hardpoints.
    raw_hardpoints = yaml_data.get("hardpoints", {})
    hardpoints, errors = parse_hardpoints(raw_hardpoints, suspension_class)
    if errors:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(errors))

    # Load configuration using Pydantic.
    config_data = yaml_data.get("config", yaml_data.get("configuration", {}))
    try:
        config = SuspensionConfig.model_validate(config_data)
    except PydanticValidationError as e:
        raise ValueError(f"Configuration validation error: {e}") from e

    # Validate shim config against suspension class.
    if config.camber_shim is not None:
        if ShimType.OUTBOARD_CAMBER not in suspension_class.SUPPORTED_SHIMS:
            raise ValueError(
                f"Suspension type '{suspension_class.__name__}' does not support "
                "outboard camber shims"
            )

    # Parse units (case-insensitive).
    units_str = yaml_data.get("units", "MILLIMETERS")
    try:
        units = coerce_enum(Units, units_str)
    except ValueError:
        raise ValueError(f"Unknown units: {units_str}")

    return suspension_class(
        name=yaml_data.get("name", "unnamed"),
        version=yaml_data.get("version", "0.0.0"),
        units=units,
        hardpoints=hardpoints,
        config=config,
    )


def parse_hardpoints(
    raw_hardpoints: dict[str, Any],
    suspension_class: type[Suspension],
) -> tuple[dict[PointID, Point3], list[str]]:
    """
    Validate and parse hardpoints from YAML data.

    Args:
        raw_hardpoints: Raw hardpoints dict from YAML.
        suspension_class: The suspension class to validate against.

    Returns:
        Tuple of (parsed hardpoints dict, list of error messages).
    """
    errors: list[str] = []
    valid_points = suspension_class.all_valid_points()
    hardpoints: dict[PointID, Point3] = {}

    for key, value in raw_hardpoints.items():
        # Case-insensitive point ID lookup.
        try:
            point_id = coerce_enum(PointID, key)
        except ValueError:
            errors.append(f"Unknown hardpoint key '{key}'")
            continue

        if point_id not in valid_points:
            errors.append(f"Unknown hardpoint key '{key}'")
            continue

        # Parse vec3 value.
        try:
            hardpoints[point_id] = coerce_point3(value)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"'{key}': {e}")

    # Check for missing required points.
    missing = suspension_class.REQUIRED_POINTS - set(hardpoints.keys())
    if missing:
        missing_names = sorted(p.name for p in missing)
        errors.append(f"Missing required hardpoints: {', '.join(missing_names)}")

    return hardpoints, errors
