"""Canonical metric identity, scope, and unit metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Sequence, cast

from kinematics.core.enums import Scope
from kinematics.core.metrics.catalog import (
    get_default_corner_derivative_metrics,
    get_default_corner_metrics,
)
from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
from kinematics.core.metrics.units import MetricUnit, MetricUnitQuotient

if TYPE_CHECKING:
    from kinematics.core.suspensions.axle import AxleSuspension
    from kinematics.core.suspensions.base import Suspension
    from kinematics.core.suspensions.corner.base import CornerSuspension

MetricUnits = MetricUnit | MetricUnitQuotient


class MetricKind(StrEnum):
    """Whether a metric is a state value or a declared derivative."""

    STATE = "state"
    DERIVATIVE = "derivative"


LOCATIONS: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for one location-independent metric identity."""

    key: str
    label: str
    unit: MetricUnits
    kind: MetricKind
    scope: Scope
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
    MetricSpec("heave", "Heave", MetricUnit.MM, MetricKind.STATE, Scope.AXLE),
    MetricSpec("roll", "Roll", MetricUnit.DEG, MetricKind.STATE, Scope.AXLE),
    MetricSpec(
        "ride_height_change",
        "Ride Height Change",
        MetricUnit.MM,
        MetricKind.STATE,
        Scope.AXLE,
    ),
    MetricSpec("track", "Track", MetricUnit.MM, MetricKind.STATE, Scope.AXLE),
    MetricSpec(
        "roll_center_y",
        "Roll Center Y",
        MetricUnit.MM,
        MetricKind.STATE,
        Scope.AXLE,
    ),
    MetricSpec(
        "roll_center_z",
        "Roll Center Z",
        MetricUnit.MM,
        MetricKind.STATE,
        Scope.AXLE,
    ),
    MetricSpec(
        "rack_displacement",
        "Rack Displacement",
        MetricUnit.MM,
        MetricKind.STATE,
        Scope.AXLE,
    ),
    MetricSpec(
        "arb_twist",
        "ARB Twist",
        MetricUnit.DEG,
        MetricKind.STATE,
        Scope.AXLE,
        "arb",
    ),
    MetricSpec(
        "t_bar_heave_angle",
        "T-Bar Heave Angle",
        MetricUnit.DEG,
        MetricKind.STATE,
        Scope.AXLE,
        "arb",
    ),
    MetricSpec(
        "heave_link_length",
        "Heave Link Length",
        MetricUnit.MM,
        MetricKind.STATE,
        Scope.AXLE,
        "heave_link",
    ),
)

_TOPOLOGY_SPECS = (
    MetricSpec(
        "rocker_angle",
        "Rocker Angle",
        MetricUnit.DEG,
        MetricKind.STATE,
        Scope.CORNER,
        "rocker",
    ),
    MetricSpec(
        "torsion_bar_twist",
        "Torsion Bar Twist",
        MetricUnit.DEG,
        MetricKind.STATE,
        Scope.CORNER,
        "torsion_bar",
    ),
    MetricSpec(
        "arb_arm_angle",
        "ARB Arm Angle",
        MetricUnit.DEG,
        MetricKind.STATE,
        Scope.CORNER,
        "arb",
    ),
)


def derivative_spec(
    definition: DerivativeMetricDefinition,
    *,
    scope: Scope = Scope.CORNER,
) -> MetricSpec:
    """Derive metadata from a declarative derivative definition."""
    response = definition.response.name.replace("_", " ").title()
    driver = definition.driver.name.replace("_", " ").title()
    return MetricSpec(
        definition.column_name,
        f"{response} wrt {driver}",
        definition.unit,
        MetricKind.DERIVATIVE,
        scope,
    )


def all_static_metric_specs() -> tuple[MetricSpec, ...]:
    """Return every statically declared state metric."""
    corner = tuple(
        MetricSpec(
            metric.column_name,
            metric.label,
            metric.unit,
            MetricKind.STATE,
            Scope.CORNER,
        )
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
    derivatives: list[tuple[DerivativeMetricDefinition, Scope]] = []
    if suspension.is_axle:
        axle = cast("AxleSuspension", suspension)
        representative = next(iter(axle.corners.values()))
        derivatives.extend(
            (definition, Scope.CORNER)
            for definition in (
                *get_default_corner_derivative_metrics(representative),
                *representative.derivative_metric_definitions(),
            )
        )
        derivatives.extend(
            (definition, Scope.AXLE)
            for definition in axle.derivative_metric_definitions()
        )
    else:
        corner = cast("CornerSuspension", suspension)
        derivatives.extend(
            (definition, Scope.CORNER)
            for definition in (
                *get_default_corner_derivative_metrics(corner),
                *corner.derivative_metric_definitions(),
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
            LOCATIONS if spec.scope is Scope.CORNER else (None,)
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
