"""Transport-neutral facade for decoded suspension and sweep inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from kinematics.core.enums import Scope
from kinematics.core.schema.geometry import GeometrySpecBase
from kinematics.core.schema.sweep import SweepSpec, build_sweep_config
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.registry import (
    SuspensionDefinition,
    get_suspension_definition,
)
from kinematics.core.targeting import SweepConfig


def _validate_geometry(
    data: Mapping[str, Any],
) -> tuple[GeometrySpecBase, SuspensionDefinition]:
    """Resolve and validate one decoded geometry mapping."""
    if "type" not in data:
        raise ValueError("Geometry type not specified")

    type_key = data["type"]
    raw_scope = data.get("scope", Scope.CORNER)
    try:
        scope = Scope(raw_scope)
    except ValueError as error:
        raise ValueError(f"Unsupported geometry scope: '{raw_scope}'") from error

    definition = get_suspension_definition(type_key, scope)
    if definition is None:
        raise ValueError(
            f"Unsupported geometry type: '{type_key}' with scope '{scope}'"
        )

    normalized = dict(data)
    normalized["type"] = definition.type_key
    normalized["scope"] = scope
    try:
        spec = definition.spec_type.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"Invalid geometry specification: {error}") from error
    return spec, definition


def parse_geometry_spec(data: Mapping[str, Any]) -> GeometrySpecBase:
    """Validate a decoded geometry mapping as its selected architecture."""
    spec, _ = _validate_geometry(data)
    return spec


def build_suspension(data: Mapping[str, Any]) -> Suspension:
    """Decode, validate, dispatch, and build one suspension mapping."""
    spec, definition = _validate_geometry(data)
    return definition.build(spec)


def parse_sweep_spec(data: Mapping[str, Any]) -> SweepSpec:
    """Validate a decoded sweep mapping."""
    try:
        return SweepSpec.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"Invalid sweep specification: {error}") from error


def build_sweep(
    data: Mapping[str, Any],
    suspension: Suspension | None = None,
) -> SweepConfig:
    """Decode, validate, and expand one sweep mapping."""
    return build_sweep_config(parse_sweep_spec(data), suspension)
