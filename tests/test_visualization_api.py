"""Tests for visualization's reconstructed contact-plane check."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest

from kinematics.cli.visualization import api
from kinematics.core.suspensions.base import Suspension


def test_visualization_api_import_does_not_require_matplotlib() -> None:
    """Non-rendering helpers remain usable without the optional viz extra."""
    script = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "from kinematics.cli.visualization import api\n"
        "assert api.GeometryVisualizationResult is not None\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "contact_centers",
    [
        ((25.0, 800.0, 125.0),),
        ((25.0, 800.0, 125.0), (25.0, -800.0, 225.0)),
    ],
    ids=("translated_corner", "translated_and_rolled_axle"),
)
def test_geometry_visualization_checks_reconstructed_road_plane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    contact_centers: tuple[tuple[float, float, float], ...],
) -> None:
    """Chassis-coordinate Z is not a valid road-contact check."""
    references = tuple(
        SimpleNamespace(wheel_contact_center=f"contact_center_{index}")
        for index in range(len(contact_centers))
    )
    positions = {
        reference.wheel_contact_center: contact_center
        for reference, contact_center in zip(references, contact_centers, strict=True)
    }
    render_model = SimpleNamespace(
        visualizer=SimpleNamespace(wheel_references=references),
        positions=lambda state: positions,
    )
    suspension = SimpleNamespace(initial_state=lambda: object())

    monkeypatch.setattr(api, "build_render_model", lambda _: render_model)
    monkeypatch.setattr(api, "create_four_view_plot", lambda **_: None)

    result = api.visualize_geometry(
        cast("Suspension", suspension), tmp_path / "geometry.png"
    )

    assert result.wheel_contact_center_z == tuple(
        contact_center[2] for contact_center in contact_centers
    )
    assert result.wheel_contact_center_road_distance_mm == pytest.approx(
        (0.0,) * len(contact_centers)
    )
    assert result.wheel_contact_centers_on_road
