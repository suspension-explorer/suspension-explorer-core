"""Tests for affine geometry primitives and their numpy interoperability."""

import numpy as np
import pytest

from kinematics.core.geometry import Point3, extract_array, try_extract_array


def test_extract_array_returns_plain_array_unchanged():
    array = np.array([1.0, 2.0, 3.0])

    assert extract_array(array) is array


def test_extract_array_returns_geometry_backing_array():
    point = Point3([1.0, 2.0, 3.0])

    assert extract_array(point) is point.data


def test_try_extract_array_rejects_unsupported_operand():
    assert try_extract_array(1.0) is None


def test_extract_array_rejects_unsupported_value():
    with pytest.raises(TypeError, match="Expected an array-backed value, got float"):
        extract_array(1.0)
