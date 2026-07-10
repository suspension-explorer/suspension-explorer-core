"""
Display metadata for exported metric columns.

A thin, derived view over the metric registry (kinematics.metrics.registry):
every column family the package can emit is declared exactly once there, and
this module resolves concrete flat export column names -- including the
location-suffixed variants rendered by ``registry.flat_key`` (e.g.
``camber_deg_left``) -- to display metadata, so that any front-end (CLI, API,
plots) labels them identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kinematics.metrics.registry import all_metric_specs, split_flat_key


@dataclass(frozen=True)
class MetricDisplay:
    """
    Display and grouping metadata for one exported metric column.

    Attributes:
        key: The exact flat export column name (e.g. "camber_gain_deg_per_mm_left").
        label: Human-readable name (e.g. "Left Camber Gain").
        unit: Physical unit symbol ("deg", "mm", "deg/mm", "mm/mm", "%", "-").
        kind: "state" or "rate", from the underlying MetricSpec.
        component: The physical component the metric belongs to ("damper",
            "rocker", "arb", ...), or None.
        motion_ratio: True for installation-ratio metrics.
        location: The corner location the column was measured at ("left" /
            "right"), or None for location-less columns.
    """

    key: str
    label: str
    unit: str
    kind: str
    component: str | None
    motion_ratio: bool
    location: str | None


def metric_display(key: str) -> MetricDisplay | None:
    """
    Resolve display metadata for a single flat export column name.

    An exact spec-key match resolves as a location-less column. Otherwise the
    key is split with ``registry.split_flat_key``; a known location suffix on
    a corner-scope spec resolves with the location folded into the label
    ("Left Camber Gain"). Axle-scope specs resolve un-suffixed only.

    Args:
        key: The flat export column name.

    Returns:
        The display metadata, or None for an unknown column.
    """
    specs = {spec.key: spec for spec in all_metric_specs()}

    spec = specs.get(key)
    location: str | None = None
    label_prefix = ""

    if spec is None:
        base_key, location = split_flat_key(key)
        if location is None:
            return None
        spec = specs.get(base_key)
        if spec is None or spec.scope != "corner":
            return None
        label_prefix = f"{location.title()} "

    return MetricDisplay(
        key=key,
        label=label_prefix + spec.label,
        unit=spec.unit,
        kind=spec.kind,
        component=spec.component,
        motion_ratio=spec.motion_ratio,
        location=location,
    )


def metric_display_for_keys(keys: Iterable[str]) -> list[MetricDisplay]:
    """
    Resolve display metadata for a sequence of column names, preserving order.

    Unknown columns are skipped rather than guessed at; the package test suite
    guards that every column actually emitted by the metrics layer resolves.

    Args:
        keys: Flat export column names, in output order.

    Returns:
        Display metadata for every recognized column, in the same order.
    """
    resolved = (metric_display(key) for key in keys)
    return [display for display in resolved if display is not None]
