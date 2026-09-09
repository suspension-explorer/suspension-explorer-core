"""Contract and coverage checks for the published capability catalog."""

import json
import subprocess
import sys
from enum import Enum
from importlib.metadata import version
from pathlib import Path

import pytest

from kinematics.core.capabilities import CapabilityManifest, create_manifest
from kinematics.core.capabilities.reference import reference_configurations
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
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.metrics.registry import (
    MetricKind,
    flat_specs_for_suspension,
    metric_specs_for_suspension,
)
from kinematics.core.suspensions.registry import SUSPENSION_DEFINITIONS


@pytest.fixture(scope="module")
def manifest() -> CapabilityManifest:
    return create_manifest()


def test_manifest_is_deterministic_and_round_trips(
    manifest: CapabilityManifest,
) -> None:
    serialized = manifest.model_dump_json()
    assert serialized == create_manifest().model_dump_json()
    assert CapabilityManifest.model_validate_json(serialized) == manifest
    assert manifest.core_version == version("kinematics")
    assert {(c.architecture, c.scope) for c in manifest.configurations} == {
        (d.type_key, d.scope) for d in SUSPENSION_DEFINITIONS
    }
    identities = [
        tuple(c.model_dump(exclude={"metric_set", "position_set"}).values())
        for c in manifest.configurations
    ]
    assert len(identities) == len(set(identities))


@pytest.mark.parametrize(
    ("selector", "enum"),
    [
        ("architecture", SuspensionType),
        ("scope", Scope),
        ("actuation", ActuationType),
        ("mount", MountBody),
        ("spring", CornerSpringType),
        ("damper", CornerDamperType),
        ("steering", SteeringType),
        ("anti_roll", ArbType),
        ("heave_link", HeaveLinkType),
    ],
)
def test_manifest_covers_every_public_selector_value(
    manifest: CapabilityManifest, selector: str, enum: type[Enum]
) -> None:
    values = {getattr(c, selector) for c in manifest.configurations} - {None}
    assert values == set(enum)


def test_manifest_preserves_conditional_mechanism_support(
    manifest: CapabilityManifest,
) -> None:
    for configuration in manifest.configurations:
        if configuration.anti_roll not in (None, ArbType.NONE):
            assert configuration.scope is Scope.AXLE
            assert configuration.actuation is ActuationType.PUSHROD_ROCKER
        if configuration.heave_link is HeaveLinkType.ROCKER_TO_ROCKER:
            assert configuration.scope is Scope.AXLE
            assert configuration.actuation is ActuationType.PUSHROD_ROCKER
        if configuration.damper is CornerDamperType.LINEAR:
            assert configuration.actuation is ActuationType.PUSHROD_ROCKER
            assert configuration.spring is not CornerSpringType.COILOVER
        if configuration.actuation is ActuationType.DIRECT:
            assert configuration.spring is not CornerSpringType.TORSION_BAR
        if (
            configuration.architecture is SuspensionType.MULTI_LINK
            and configuration.actuation is ActuationType.PUSHROD_ROCKER
        ):
            assert configuration.mount is MountBody.UPRIGHT
        if configuration.architecture in (
            SuspensionType.MACPHERSON,
            SuspensionType.TRAILING_ARM,
        ):
            assert configuration.actuation is None
            assert configuration.anti_roll in (None, ArbType.NONE)
            assert configuration.heave_link in (None, HeaveLinkType.NONE)
    assert any(
        c.architecture is SuspensionType.MULTI_LINK
        and c.anti_roll is ArbType.T_BAR
        and c.heave_link is HeaveLinkType.ROCKER_TO_ROCKER
        for c in manifest.configurations
    )


def test_manifest_matches_real_state_outputs_and_flat_export_names(
    manifest: CapabilityManifest,
) -> None:
    declared = {m.key: m for m in manifest.metrics}
    for configuration, (selectors, suspension) in zip(
        manifest.configurations, reference_configurations(), strict=True
    ):
        assert configuration.architecture is selectors.architecture
        specs = metric_specs_for_suspension(suspension)
        assert set(manifest.metric_sets[configuration.metric_set]) == set(specs)
        row = suspension.compute_state_metrics(suspension.initial_state())
        if isinstance(row, AxleMetricRows):
            emitted = set(row.axle)
            for corner_row in row.corners.values():
                emitted.update(corner_row)
        else:
            emitted = set(row)
        assert emitted == {
            key for key, spec in specs.items() if spec.kind is MetricKind.STATE
        }
        flat_keys = {
            f"{key}_{side}"
            if suspension.is_axle and declared[key].scope is Scope.CORNER
            else key
            for key in specs
            for side in (
                ("left", "right")
                if suspension.is_axle and declared[key].scope is Scope.CORNER
                else (None,)
            )
        }
        assert flat_keys == set(flat_specs_for_suspension(suspension))
    assert declared["deriv_t_bar_center_x_wrt_hub_z_left"].unit == "mm/mm"
    assert declared["heave_link_length"].scope is Scope.AXLE
    assert declared["caster_virtual"].unit == "deg"


def test_manifest_cli_needs_no_optional_dependencies(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.json"
    schema = tmp_path / "schema.json"
    script = (
        "import runpy, sys\n"
        "for name in ('matplotlib', 'pyarrow', 'typer', 'yaml'):\n"
        "    sys.modules[name] = None\n"
        f"sys.argv = ['manifest', '--out', {str(output)!r}, "
        f"'--schema-out', {str(schema)!r}]\n"
        "runpy.run_module('kinematics.core.capabilities', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert (
        CapabilityManifest.model_validate_json(output.read_text()).schema_version == 1
    )
    assert json.loads(schema.read_text())["properties"]["schema_version"]["const"] == 1


def test_manifest_cli_rejects_colliding_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("existing")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kinematics.core.capabilities",
            "--out",
            str(output),
            "--schema-out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert output.read_text() == "existing"
