"""Tests for canonical metric identities and display metadata."""

from kinematics.core.metrics.metadata import metric_display
from kinematics.core.metrics.registry import flat_key, specs_by_key, split_flat_key
from kinematics.core.metrics.units import MetricUnit


def test_static_metric_specs_use_unit_free_identities() -> None:
    specs = specs_by_key()

    assert specs["camber"].unit is MetricUnit.DEG
    assert specs["camber"].scope == "corner"
    assert specs["track"].unit is MetricUnit.MM
    assert specs["track"].scope == "axle"
    assert "camber_deg" not in specs
    assert "track_mm" not in specs


def test_flat_location_is_separate_from_metric_identity() -> None:
    assert flat_key("camber", "left") == "camber_left"
    assert split_flat_key("camber_left") == ("camber", "left")
    assert split_flat_key("track") == ("track", None)


def test_display_metadata_resolves_corner_location_without_changing_units() -> None:
    specs = specs_by_key()

    display = metric_display("camber_right", specs)

    assert display is not None
    assert display.key == "camber_right"
    assert display.label == "Right Camber"
    assert display.unit == "deg"
    assert display.location == "right"


def test_axle_metric_rejects_corner_location() -> None:
    assert metric_display("track_left", specs_by_key()) is None
