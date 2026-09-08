"""Enumerate supported selectors and bind them to packaged reference geometry.

The reference dimensions only allow the real builders and metric declarations
to run. They are not performance recommendations, numerical sweep results, or
an alternative source of compatibility rules. Unsupported selector combinations
are rejected by production schema/actuation validation; a failure to build a
validated combination is a manifest error, never silently omitted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from importlib.resources import files
from itertools import product
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from kinematics.core.enums import (
    ActuationType,
    ArbType,
    CornerDamperType,
    CornerSpringType,
    HeaveLinkType,
    MountBody,
    Scope,
    SteeringType,
    SuspensionType,
)
from kinematics.core.schema.geometry import (
    ActuationSpec,
    CornerDamperSpec,
    CornerSpringSpec,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.build import (
    build_actuation,
    build_corner_damper,
    build_corner_spring,
)
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.suspensions.corner.mechanisms import ActuationDirect
from kinematics.core.suspensions.registry import (
    SUSPENSION_DEFINITIONS,
    SuspensionDefinition,
    get_suspension_definition,
)


class Selectors(BaseModel):
    """One complete selection of the finite mechanism options.

    Null means the architecture has no such selector. A literal 'none' means
    it has a selector and explicitly supports omitting that mechanism.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: SuspensionType
    scope: Scope
    actuation: ActuationType | None
    mount: MountBody | None
    spring: CornerSpringType | None
    damper: CornerDamperType | None
    steering: SteeringType
    anti_roll: ArbType | None
    heave_link: HeaveLinkType | None


def _corner_class(definition: SuspensionDefinition) -> type[CornerSuspension]:
    corner = get_suspension_definition(definition.type_key, Scope.CORNER)
    if corner is None:
        raise ValueError(f"No corner definition for {definition.type_key}")
    return cast("type[CornerSuspension]", corner.suspension_type)


def selector_candidates(definition: SuspensionDefinition) -> Iterator[Selectors]:
    """Enumerate enums only where the registered schema exposes a selector."""
    corner = get_suspension_definition(definition.type_key, Scope.CORNER)
    if corner is None:
        raise ValueError(f"No corner definition for {definition.type_key}")
    fields = corner.spec_type.model_fields
    composed = "actuation" in fields
    axle = definition.scope is Scope.AXLE
    for values in product(
        tuple(ActuationType) if composed else (None,),
        tuple(MountBody) if composed else (None,),
        tuple(CornerSpringType) if "spring" in fields else (None,),
        tuple(CornerDamperType) if "damper" in fields else (None,),
        tuple(SteeringType),
        tuple(ArbType) if axle else (None,),
        tuple(HeaveLinkType) if axle else (None,),
    ):
        yield Selectors(
            architecture=definition.type_key,
            scope=definition.scope,
            **dict(
                zip(
                    (
                        "actuation",
                        "mount",
                        "spring",
                        "damper",
                        "steering",
                        "anti_roll",
                        "heave_link",
                    ),
                    values,
                    strict=True,
                )
            ),
        )


def _geometry_mapping(selectors: Selectors, seed: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(seed[selectors.architecture]["config"])
    config["steering"] = {"type": selectors.steering}
    mechanisms = {
        key: {"type": value}
        for key in ("spring", "damper", "anti_roll", "heave_link")
        if (value := getattr(selectors, key)) is not None
    }
    if selectors.actuation is not None:
        mechanisms["actuation"] = {
            "type": selectors.actuation,
            "mount": selectors.mount,
        }
    common = {"type": selectors.architecture, "scope": selectors.scope}
    if selectors.scope is Scope.CORNER:
        return {**common, **mechanisms, "config": config, "hardpoints": {}}
    return {
        **common,
        "vehicle_config": {key: config[key] for key in ("cg_position", "wheelbase")},
        "axle_config": {
            **mechanisms,
            "steering": config["steering"],
            "wheel": config["wheel"],
            "axle_position": config.get("axle_position", "front"),
        },
        "hardpoints": {"left": {}},
    }


def _reference_points(
    selectors: Selectors, cls: type[CornerSuspension], seed: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    available = deepcopy(seed[selectors.architecture]["hardpoints"])
    required = {point.name.lower() for point in cls.REQUIRED_POINTS}
    center: dict[str, Any] = {}
    if selectors.architecture is not SuspensionType.TRAILING_ARM:
        prefix = "trackrod" if selectors.steering is SteeringType.RACK else "toe_link"
        for end in ("inboard", "outboard"):
            available[f"{prefix}_{end}"] = available[f"trackrod_{end}"]
            required.add(f"{prefix}_{end}")
    if selectors.actuation is not None and selectors.mount is not None:
        actuation = build_actuation(
            ActuationSpec(type=selectors.actuation, mount=selectors.mount),
            mount_bodies=getattr(cls, "MOUNT_BODIES", {}),
        )
        required.update(point.name.lower() for point in actuation.required_points)
        if isinstance(actuation, ActuationDirect) and actuation.derive_pickup_on_link:
            endpoints = [
                available[p.name.lower()] for p in actuation.spring_pickup_body
            ]
            available["strut_bottom"] = {
                axis: sum(point[axis] for point in endpoints) / 2
                for axis in ("x", "y", "z")
            }
        if selectors.spring is not None:
            spring = build_corner_spring(CornerSpringSpec(type=selectors.spring))
            required.update(point.name.lower() for point in spring.required_points)
        if selectors.damper is not None:
            damper = build_corner_damper(CornerDamperSpec(type=selectors.damper))
            required.update(point.name.lower() for point in damper.required_points)
    elif selectors.architecture is SuspensionType.TRAILING_ARM:
        points = getattr(
            cls,
            "COILOVER_POINTS"
            if selectors.spring is CornerSpringType.COILOVER
            else "TORSION_POINTS",
        )
        required.update(point.name.lower() for point in points)
    if selectors.anti_roll not in (None, ArbType.NONE):
        arb = seed[selectors.anti_roll]
        available.update(arb["hardpoints"])
        required.update(arb["hardpoints"])
        center = deepcopy(arb["center"])
    if selectors.heave_link is HeaveLinkType.ROCKER_TO_ROCKER:
        required.add("heave_link_rocker")
    return {key: available[key] for key in sorted(required)}, center


def reference_configurations() -> Iterator[tuple[Selectors, Suspension]]:
    """Build every schema/actuation-valid selector combination without solving."""
    seed = json.loads(
        files(__package__).joinpath("reference_geometry.json").read_text()
    )
    for definition in SUSPENSION_DEFINITIONS:
        cls = _corner_class(definition)
        for selectors in selector_candidates(definition):
            geometry = _geometry_mapping(selectors, seed)
            try:
                definition.spec_type.model_validate(geometry)
                if selectors.actuation is not None and selectors.mount is not None:
                    build_actuation(
                        ActuationSpec(type=selectors.actuation, mount=selectors.mount),
                        mount_bodies=getattr(cls, "MOUNT_BODIES", {}),
                    )
            except (ValidationError, ValueError):
                continue
            try:
                points, center = _reference_points(selectors, cls, seed)
                geometry["hardpoints"] = (
                    {"left": points, "center": center}
                    if selectors.scope is Scope.AXLE
                    else points
                )
                yield (
                    selectors,
                    definition.build(definition.spec_type.model_validate(geometry)),
                )
            except (ValueError, KeyError) as error:
                raise ValueError(
                    f"Manifest reference failed for {selectors.model_dump()}: {error}"
                ) from error
