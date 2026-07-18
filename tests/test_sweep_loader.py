"""Tests for the sweep YAML adapter boundary."""

from pathlib import Path
from typing import Any, cast

import pytest

import kinematics.cli.io.sweep_loader as sweep_loader
from kinematics.core.suspensions.base import Suspension
from kinematics.core.targeting import SweepConfig


def test_load_sweep_delegates_decoded_mapping_to_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text("version: 1\ntargets: []\n", encoding="utf-8")
    suspension = cast(Suspension, object())
    sentinel = cast(SweepConfig, object())
    captured: dict[str, Any] = {}

    def fake_build_sweep(
        data: dict[str, Any],
        received_suspension: Suspension | None,
    ) -> SweepConfig:
        captured.update(data)
        assert received_suspension is suspension
        return sentinel

    monkeypatch.setattr(sweep_loader, "build_sweep", fake_build_sweep)

    result = sweep_loader.load_sweep(path, suspension)

    assert result is sentinel
    assert captured == {"version": 1, "targets": []}
