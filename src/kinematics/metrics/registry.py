"""Canonical metric identity, scope, and unit metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Sequence, cast

from kinematics.metrics.catalog import (
    get_default_corner_derivative_metrics,
    get_default_corner_metrics,
)
from kinematics.metrics.derivatives import DerivativeMetricDefinition
from kinematics.metrics.units import MetricUnit, MetricUnitQuotient

if TYPE_CHECKING:
    from kinematics.suspensions.axle import DoubleWishboneAxleSuspension
    from kinematics.suspensions.base import Suspension

MetricUnits = MetricUnit | MetricUnitQuotient
MetricScope = Literal["corner", "axle"]
MetricKind = Literal["state", "derivative"]

LOCATIONS: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for one location-independent metric identity."""

    key: str
    label: str
    unit: MetricUnits
    kind: MetricKind
    scope: MetricScope
    component: str | None = None


def flat_key(key: str, location: str | None = None) -> str:
    """Render a location only at a flat export boundary."""
    return key if location is None else f"{key}_{location}"


def split_flat_key(key: str) -> tuple[str, str | None]:
    """Split a canonical flat key into its identity and location."""
    for location in LOCATIONS:
        suffix = f"_{location}"
        if key.endswith(suffix):
            return key[: -len(suffix)], location
    return key, None


_AXLE_SPECS = (
    MetricSpec("heave", "Heave", MetricUnit.MM, "state", "axle"),
    MetricSpec("roll", "Roll", MetricUnit.DEG, "state", "axle"),
    MetricSpec(
        "ride_height_change", "Ride Height Change", MetricUnit.MM, "state", "axle"
    ),
    MetricSpec("track", "Track", MetricUnit.MM, "state", "axle"),
    MetricSpec("roll_center_y", "Roll Center Y", MetricUnit.MM, "state", "axle"),
    MetricSpec("roll_center_z", "Roll Center Z", MetricUnit.MM, "state", "axle"),
    MetricSpec(
        "trackrod_inboard_displacement",
        "Trackrod Inboard Displacement",
        MetricUnit.MM,
        "state",
        "axle",
    ),
    MetricSpec("arb_twist", "ARB Twist", MetricUnit.DEG, "state", "axle", "arb"),
)

_TOPOLOGY_SPECS = (
    MetricSpec(
        "rocker_angle", "Rocker Angle", MetricUnit.DEG, "state", "corner", "rocker"
    ),
    MetricSpec(
        "torsion_bar_twist",
        "Torsion Bar Twist",
        MetricUnit.DEG,
        "state",
        "corner",
        "torsion_bar",
    ),
    MetricSpec(
        "arb_arm_angle", "ARB Arm Angle", MetricUnit.DEG, "state", "corner", "arb"
    ),
)


def derivative_spec(
    definition: DerivativeMetricDefinition,
    *,
    scope: MetricScope = "corner",
) -> MetricSpec:
    """Derive metadata from a declarative derivative definition."""
    response = definition.response.name.replace("_", " ").title()
    driver = definition.driver.name.replace("_", " ").title()
    return MetricSpec(
        definition.column_name,
        f"{response} wrt {driver}",
        definition.unit,
        "derivative",
        scope,
    )


def all_static_metric_specs() -> tuple[MetricSpec, ...]:
    """Return every statically declared state metric."""
    corner = tuple(
        MetricSpec(metric.column_name, metric.label, metric.unit, "state", "corner")
        for metric in get_default_corner_metrics()
    )
    specs = corner + _TOPOLOGY_SPECS + _AXLE_SPECS
    _validate_unique(specs)
    return specs


def specs_by_key(
    derivative_definitions: Sequence[DerivativeMetricDefinition] = (),
) -> dict[str, MetricSpec]:
    """Return static and supplied derivative metadata keyed by identity."""
    specs = (
        *all_static_metric_specs(),
        *(derivative_spec(d) for d in derivative_definitions),
    )
    _validate_unique(specs)
    return {spec.key: spec for spec in specs}


def metric_specs_for_suspension(suspension: "Suspension") -> dict[str, MetricSpec]:
    """Return all metadata that the selected topology can emit."""
    derivatives: list[tuple[DerivativeMetricDefinition, MetricScope]] = []
    if suspension.is_axle:
        axle = cast("DoubleWishboneAxleSuspension", suspension)
        representative = next(iter(axle.corners.values()))
        derivatives.extend(
            (definition, "corner")
            for definition in (
                *get_default_corner_derivative_metrics(representative),
                *representative.derivative_metric_definitions(),
            )
        )
        derivatives.extend(
            (definition, "axle") for definition in axle.derivative_metric_definitions()
        )
    else:
        derivatives.extend(
            (definition, "corner")
            for definition in (
                *get_default_corner_derivative_metrics(suspension),
                *suspension.derivative_metric_definitions(),
            )
        )

    result = {spec.key: spec for spec in all_static_metric_specs()}
    for definition, scope in derivatives:
        spec = derivative_spec(definition, scope=scope)
        existing = result.get(spec.key)
        if existing is not None and existing != spec:
            raise ValueError(f"Conflicting metric specification: {spec.key}")
        result[spec.key] = spec
    return result


def flat_specs_for_suspension(suspension: "Suspension") -> dict[str, MetricSpec]:
    """Render topology metadata using the same keys as flat result exports."""
    specs = metric_specs_for_suspension(suspension)
    if not suspension.is_axle:
        return specs

    result: dict[str, MetricSpec] = {}
    for spec in specs.values():
        locations: tuple[str | None, ...] = (
            LOCATIONS if spec.scope == "corner" else (None,)
        )
        for location in locations:
            key = flat_key(spec.key, location)
            result[key] = replace(spec, key=key)
    return result


def _validate_unique(specs: Sequence[MetricSpec]) -> None:
    """Reject ambiguous metadata declarations."""
    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            raise ValueError(f"Duplicate metric specification: {spec.key}")
        seen.add(spec.key)
