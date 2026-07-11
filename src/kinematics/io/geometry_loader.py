"""Compatibility adapters for geometry loading moved to ``kinematics.io``."""

from typing import Any

from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.io.loaders import load_geometry
from kinematics.schema.coercion import coerce_point3
from kinematics.schema.geometry import parse_geometry_spec
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.build import build_suspension


def load_suspension(
    yaml_data: dict[str, Any], suspension_class: type[Suspension]
) -> Suspension:
    """Build legacy parsed geometry through the structured schema pipeline."""
    data = dict(yaml_data)
    data["type"] = suspension_class.TYPE_KEY
    if "config" not in data and "configuration" in data:
        data["config"] = data.pop("configuration")
    return build_suspension(parse_geometry_spec(data))


def parse_hardpoints(
    raw_hardpoints: dict[str, Any], suspension_class: type[Suspension]
) -> tuple[dict[PointID, Point3], list[str]]:
    """Parse hardpoints using the legacy error-collecting helper contract."""
    hardpoints: dict[PointID, Point3] = {}
    errors: list[str] = []
    for key, value in raw_hardpoints.items():
        try:
            point = PointID[key.upper()]
        except KeyError:
            errors.append(f"Unknown hardpoint key '{key}'")
            continue
        if point not in suspension_class.all_valid_points():
            errors.append(f"Unknown hardpoint key '{key}'")
            continue
        try:
            hardpoints[point] = coerce_point3(value)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"'{key}': {error}")

    missing = suspension_class.REQUIRED_POINTS - set(hardpoints)
    if missing:
        names = ", ".join(sorted(point.name for point in missing))
        errors.append(f"Missing required hardpoints: {names}")
    return hardpoints, errors

__all__ = ["load_geometry", "load_suspension", "parse_hardpoints"]
