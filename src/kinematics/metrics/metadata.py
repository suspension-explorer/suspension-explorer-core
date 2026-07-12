"""Display metadata derived from canonical metric specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from kinematics.metrics.registry import MetricSpec, split_flat_key


@dataclass(frozen=True)
class MetricDisplay:
    """Consumer-facing metadata for one metric at an optional location."""

    key: str
    label: str
    unit: str
    kind: str
    scope: str
    component: str | None
    location: str | None


def metric_display(key: str, specs: Mapping[str, MetricSpec]) -> MetricDisplay | None:
    """Resolve a base or flat export key without inferring its unit."""
    spec = specs.get(key)
    location: str | None = None
    if spec is None:
        base_key, location = split_flat_key(key)
        spec = specs.get(base_key)
        if spec is None or spec.scope != "corner":
            return None
    prefix = "" if location is None else f"{location.title()} "
    return MetricDisplay(
        key=key,
        label=f"{prefix}{spec.label}",
        unit=spec.unit.symbol,
        kind=spec.kind,
        scope=spec.scope,
        component=spec.component,
        location=location,
    )


def metric_display_for_keys(
    keys: Iterable[str], specs: Mapping[str, MetricSpec]
) -> list[MetricDisplay]:
    """Resolve known keys in input order."""
    displays = (metric_display(key, specs) for key in keys)
    return [display for display in displays if display is not None]
