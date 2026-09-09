"""Create a deterministic catalog from production suspension declarations."""

from __future__ import annotations

from importlib.metadata import version
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kinematics.core.capabilities.reference import Selectors, reference_configurations
from kinematics.core.enums import Scope, ShimType, SuspensionType
from kinematics.core.export import StandardColumn
from kinematics.core.metrics.registry import (
    MetricKind,
    MetricSpec,
    metric_specs_for_suspension,
)
from kinematics.core.primitives.point_ref import point_key_name
from kinematics.core.suspensions.registry import SUSPENSION_DEFINITIONS


class ManifestModel(BaseModel):
    """Strict JSON contract shared by manifest records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Architecture(ManifestModel):
    """Architecture identity and setup capabilities from its registered class."""

    key: SuspensionType
    label: str
    scopes: list[Scope]
    setup_shims: list[ShimType]


class Metric(ManifestModel):
    """One canonical metric identity, without flat location suffixes."""

    key: str
    label: str
    unit: str
    kind: MetricKind
    scope: Scope
    component: str | None

    @classmethod
    def from_spec(cls, spec: MetricSpec) -> Metric:
        """Serialize structured physical units using their export symbols."""
        return cls(
            key=spec.key,
            label=spec.label,
            unit=spec.unit.symbol,
            kind=spec.kind,
            scope=spec.scope,
            component=spec.component,
        )


class Configuration(Selectors):
    """An allowed selector combination and its declared output sets."""

    metric_set: str
    position_set: str


class CapabilityManifest(ManifestModel):
    """Version 1 capability manifest, independent of CLI and website packages.

    A declared metric can still be null for an undefined geometric result or
    absent physical input. Configurations enumerate finite mechanism selectors,
    not numerical hardpoints, setup thicknesses, or custom response definitions.
    """

    schema_version: Literal[1] = 1
    core_version: str
    architectures: list[Architecture]
    metrics: list[Metric]
    metric_sets: dict[str, list[str]]
    position_sets: dict[str, list[str]]
    configurations: list[Configuration]
    standard_columns: list[str]
    position_column_pattern: str = "{point}_{axis}"
    position_unit: str = "mm"
    corner_metric_column_pattern: str = "{metric}_{side}"
    target_column_pattern: str = "target_{coordinate_export_id}{side_suffix}"


ARCHITECTURE_LABELS = {
    SuspensionType.DOUBLE_WISHBONE: "Double wishbone",
    SuspensionType.MACPHERSON: "MacPherson strut",
    SuspensionType.MULTI_LINK: "Multi-link (five-link)",
    SuspensionType.TRAILING_ARM: "Semi-trailing arm",
}


def _intern(keys: list[str], sets: dict[str, list[str]], prefix: str) -> str:
    for name, existing in sets.items():
        if keys == existing:
            return name
    name = f"{prefix}-{len(sets) + 1:03d}"
    sets[name] = keys
    return name


def create_manifest() -> CapabilityManifest:
    """Enumerate every supported built-in selector combination and its outputs.

    Builds reference models and reads declarations without running a solver.
    Output is deterministic for a given package version and source tree; there
    are deliberately no timestamps or environment-specific paths.
    """
    architectures = [
        Architecture(
            key=definition.type_key,
            label=ARCHITECTURE_LABELS[definition.type_key],
            scopes=[
                d.scope
                for d in SUSPENSION_DEFINITIONS
                if d.type_key is definition.type_key
            ],
            setup_shims=sorted(definition.suspension_type.SUPPORTED_SHIMS),
        )
        for definition in SUSPENSION_DEFINITIONS
        if definition.scope is Scope.CORNER
    ]
    metrics: dict[str, Metric] = {}
    metric_sets: dict[str, list[str]] = {}
    position_sets: dict[str, list[str]] = {}
    configurations: list[Configuration] = []
    for selectors, suspension in reference_configurations():
        specs = metric_specs_for_suspension(suspension)
        for key, spec in specs.items():
            metric = Metric.from_spec(spec)
            if key in metrics and metrics[key] != metric:
                raise ValueError(f"Conflicting metric declarations for {key}")
            metrics[key] = metric
        configurations.append(
            Configuration(
                **selectors.model_dump(),
                metric_set=_intern(sorted(specs), metric_sets, "metrics"),
                position_set=_intern(
                    sorted(
                        point_key_name(point) for point in suspension.output_points()
                    ),
                    position_sets,
                    "positions",
                ),
            )
        )
    expected = {(d.type_key, d.scope) for d in SUSPENSION_DEFINITIONS}
    actual = {(c.architecture, c.scope) for c in configurations}
    if expected != actual:
        raise ValueError(f"Missing manifest configurations: {expected - actual}")
    return CapabilityManifest(
        core_version=version("kinematics"),
        architectures=architectures,
        metrics=[metrics[key] for key in sorted(metrics)],
        metric_sets=metric_sets,
        position_sets=position_sets,
        configurations=configurations,
        standard_columns=[column.value for column in StandardColumn],
    )
