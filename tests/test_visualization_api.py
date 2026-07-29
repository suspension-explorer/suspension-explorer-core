"""Tests for visualization's reconstructed contact-plane check."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from kinematics.cli.visualization import api
from kinematics.core.suspensions.base import Suspension


@pytest.mark.parametrize(
    "contact_centres",
    [
        ((25.0, 800.0, 125.0),),
        ((25.0, 800.0, 125.0), (25.0, -800.0, 225.0)),
    ],
    ids=("translated_corner", "translated_and_rolled_axle"),
)
def test_geometry_visualization_checks_reconstructed_road_plane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    contact_centres: tuple[tuple[float, float, float], ...],
) -> None:
    """Chassis-coordinate Z is not a valid road-contact check."""
    references = tuple(
        SimpleNamespace(wheel_contact_centre=f"contact_centre_{index}")
        for index in range(len(contact_centres))
    )
    positions = {
        reference.wheel_contact_centre: contact_centre
        for reference, contact_centre in zip(references, contact_centres, strict=True)
    }
    render_model = SimpleNamespace(
        visualizer=SimpleNamespace(wheel_references=references),
        positions=lambda state: positions,
    )
    suspension = SimpleNamespace(initial_state=lambda: object())

    monkeypatch.setattr(api, "build_render_model", lambda _: render_model)
    monkeypatch.setattr(api, "create_four_view_plot", lambda **_: None)

    result = api.visualize_geometry(
        cast(Suspension, suspension), tmp_path / "geometry.png"
    )

    assert result.wheel_contact_centre_z == tuple(
        contact_centre[2] for contact_centre in contact_centres
    )
    assert result.wheel_contact_centre_road_distance_mm == pytest.approx(
        (0.0,) * len(contact_centres)
    )
    assert result.wheel_contact_centres_on_road
