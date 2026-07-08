"""
Display metadata for exported metric columns.

A thin, derived view over the metric registry (kinematics.metrics.registry):
every column family the package can emit is declared exactly once there, and
this module resolves concrete export column names -- including the axle's
``left_`` / ``right_`` prefixed variants -- to display metadata, so that any
front-end (CLI, API, plots) labels them identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kinematics.metrics.registry import all_metric_specs


@dataclass(frozen=True)
class MetricDisplay:
    """
    Display metadata for one exported metric column.

    Attributes:
        key: The exact export column name (e.g. "left_camber_gain_deg_per_mm").
        label: Human-readable name (e.g. "Left Camber Gain").
        unit: Physical unit symbol ("deg", "mm", "deg/mm", "mm/mm", "%", "-").
    """

    key: str
    label: str
    unit: str


_SIDE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("left_", "Left "),
    ("right_", "Right "),
)


def metric_display(key: str) -> MetricDisplay | None:
    """
    Resolve display metadata for a single exported column name.

    Corner-scope specs resolve both un-prefixed (corner models) and ``left_``
    / ``right_`` prefixed (axle models, side prepended to the label);
    axle-scope specs resolve un-prefixed only.

    Args:
        key: The export column name.

    Returns:
        The display metadata, or None for an unknown column.
    """
    specs = {spec.key: spec for spec in all_metric_specs()}

    spec = specs.get(key)
    if spec is not None:
        return MetricDisplay(key=key, label=spec.label, unit=spec.unit)

    for prefix, side_label in _SIDE_PREFIXES:
        if key.startswith(prefix):
            spec = specs.get(key[len(prefix) :])
            if spec is not None and spec.scope == "corner":
                return MetricDisplay(
                    key=key, label=side_label + spec.label, unit=spec.unit
                )

    return None


def metric_display_for_keys(keys: Iterable[str]) -> list[MetricDisplay]:
    """
    Resolve display metadata for a sequence of column names, preserving order.

    Unknown columns are skipped rather than guessed at; the package test suite
    guards that every column actually emitted by the metrics layer resolves.

    Args:
        keys: Export column names, in output order.

    Returns:
        Display metadata for every recognized column, in the same order.
    """
    resolved = (metric_display(key) for key in keys)
    return [display for display in resolved if display is not None]
