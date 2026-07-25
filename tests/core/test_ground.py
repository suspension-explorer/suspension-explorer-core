"""Tests for the axle-level ground-line primitive."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from kinematics.core.metrics.ground import AxleGroundLine
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3


def _line(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> AxleGroundLine:
    ground_line = AxleGroundLine.from_contact_patches(Point3(left), Point3(right))
    assert ground_line is not None
    return ground_line


def test_flat_ground_line_has_upward_normal_and_expected_height():
    ground_line = _line((100.0, 500.0, 20.0), (-100.0, -500.0, 20.0))

    assert ground_line.tangent_y == pytest.approx(1.0)
    assert ground_line.tangent_z == pytest.approx(0.0)
    assert ground_line.normal_y == pytest.approx(0.0)
    assert ground_line.normal_z == pytest.approx(1.0)
    assert ground_line.offset_mm == pytest.approx(-20.0)
    assert ground_line.c == pytest.approx(-20.0)
    assert ground_line.angle_deg == pytest.approx(0.0)
    assert ground_line.z_at(-250.0) == pytest.approx(20.0)
    assert ground_line.plane_normal == pytest.approx((0.0, 0.0, 1.0))


def test_signed_roll_is_positive_when_left_contact_patch_is_higher():
    ground_line = _line((0.0, 500.0, 100.0), (0.0, -500.0, 0.0))

    assert ground_line.angle_deg == pytest.approx(np.degrees(np.arctan(0.1)))
    assert ground_line.normal_z > 0.0
    assert ground_line.z_at(0.0) == pytest.approx(50.0)


def test_equal_z_translation_preserves_orientation_and_shifts_offset():
    original = _line((0.0, 500.0, 100.0), (0.0, -500.0, 0.0))
    translated = _line((0.0, 500.0, 137.5), (0.0, -500.0, 37.5))

    assert (translated.tangent_y, translated.tangent_z) == pytest.approx(
        (original.tangent_y, original.tangent_z)
    )
    assert translated.offset_mm == pytest.approx(
        original.offset_mm - 37.5 * original.normal_z
    )
    original_z = original.z_at(0.0)
    translated_z = translated.z_at(0.0)
    assert original_z is not None
    assert translated_z is not None
    assert translated_z == pytest.approx(original_z + 37.5)


def test_argument_order_and_x_coordinates_do_not_change_line():
    left = Point3((9999.0, 500.0, 100.0))
    right = Point3((-9999.0, -500.0, 0.0))
    ordered = AxleGroundLine.from_contact_patches(left, right)
    reversed_order = AxleGroundLine.from_contact_patches(right, left)
    changed_x = AxleGroundLine.from_contact_patches(
        Point3((-1.0, 500.0, 100.0)), Point3((1.0, -500.0, 0.0))
    )

    assert ordered is not None
    assert reversed_order == ordered
    assert changed_x == ordered


def test_line_equation_contains_both_contact_patches_and_distance_is_signed():
    left = Point3((0.0, 500.0, 100.0))
    right = Point3((0.0, -500.0, 0.0))
    ground_line = AxleGroundLine.from_contact_patches(left, right)
    assert ground_line is not None

    assert ground_line.signed_distance_yz(left) == pytest.approx(0.0)
    assert ground_line.signed_distance_yz(right) == pytest.approx(0.0)
    assert ground_line.signed_distance_yz(Point3((123.0, 0.0, 60.0))) > 0.0
    assert ground_line.signed_distance_yz(Point3((123.0, 0.0, 40.0))) < 0.0


def test_factory_rejects_collapsed_lateral_track_despite_height_difference():
    ground_line = AxleGroundLine.from_contact_patches(
        Point3((0.0, 0.0, 10.0)), Point3((0.0, 0.0, 0.0))
    )

    assert ground_line is None


def test_direct_vertical_line_has_no_z_at_value():
    ground_line = AxleGroundLine(0.0, 1.0, -1.0, 0.0, 0.0)

    assert ground_line.z_at(0.0) is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, EPS_GEOMETRIC, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, np.nan, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 1.0, np.inf), (1.0, -1.0, 0.0)),
    ],
)
def test_degenerate_or_nonfinite_contact_patches_return_none(left, right):
    assert AxleGroundLine.from_contact_patches(Point3(left), Point3(right)) is None


def test_ground_line_is_immutable():
    ground_line = _line((0.0, 500.0, 0.0), (0.0, -500.0, 0.0))

    with pytest.raises(FrozenInstanceError):
        setattr(ground_line, "offset_mm", 1.0)


def test_direct_construction_accepts_a_canonical_line():
    ground_line = AxleGroundLine(0.6, 0.8, -0.8, 0.6, -12.0)

    assert ground_line.offset_mm == -12.0


@pytest.mark.parametrize(
    "values",
    [
        (np.nan, 0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 1.0, np.inf),
        (2.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 2.0, 0.0),
        (0.6, 0.8, 0.6, 0.8, 0.0),
        (0.6, 0.8, 0.8, -0.6, 0.0),
        (-1.0, 0.0, 0.0, -1.0, 0.0),
        (0.0, -1.0, 1.0, 0.0, 0.0),
    ],
)
def test_direct_construction_rejects_invalid_line_invariants(values):
    with pytest.raises(ValueError):
        AxleGroundLine(*values)
