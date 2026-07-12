from pathlib import Path

import pytest
import yaml

from kinematics.core.point_ref import Side
from kinematics.io import load_geometry
from kinematics.suspensions.base import Suspension


@pytest.fixture
def empty_geometry_file(tmp_path: Path):
    empty_file = tmp_path / "empty_geometry.yaml"
    empty_file.touch()
    return empty_file


@pytest.fixture
def invalid_yaml_geometry_file(tmp_path: Path):
    file_path = tmp_path / "invalid_geometry.yaml"
    file_path.write_text('""')
    return file_path


@pytest.fixture
def invalid_geometry_file(tmp_path: Path):
    data = {
        "type": "double_wishbone",
        "side": "LEFT",
        "hardpoints": {
            # Missing most required hardpoints
            "LOWER_WISHBONE_INBOARD_FRONT": [0, 0, 0],
        },
        "config": {
            "steered": True,
            "wheel": {
                "offset": 0,
                "tire": {
                    "aspect_ratio": 0.55,
                    "section_width": 270,
                    "rim_diameter": 13,
                },
            },
            "cg_position": {"x": 0, "y": 0, "z": 0},
            "wheelbase": 2500.0,
        },
    }  # Valid type but missing required hardpoints
    file_path = tmp_path / "invalid_geometry.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


def test_load_geometry_valid(double_wishbone_geometry_file):
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, Suspension)
    # Test that the suspension has required attributes and methods.
    assert suspension.hardpoints is not None
    assert suspension.config is not None
    assert suspension.side is Side.LEFT
    assert suspension.initial_state() is not None


def test_load_geometry_empty(empty_geometry_file):
    with pytest.raises(ValueError, match="Geometry file is empty"):
        load_geometry(empty_geometry_file)


def test_load_geometry_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Geometry file not found"):
        load_geometry(tmp_path / "file_not_found.yaml")


def test_load_geometry_invalid(invalid_geometry_file):
    with pytest.raises(ValueError, match="Missing required hardpoints"):
        load_geometry(invalid_geometry_file)


def test_load_geometry_unsupported_type(tmp_path: Path):
    """
    Test handling of unsupported geometry types.
    """
    data = {"type": "UNSUPPORTED_TYPE", "name": "test"}
    file_path = tmp_path / "unsupported.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)

    with pytest.raises(ValueError, match="Unsupported geometry type"):
        load_geometry(file_path)


def test_load_geometry_missing_type(tmp_path: Path):
    """
    Test handling of missing geometry type.
    """
    data = {"name": "test"}  # Missing 'type' field
    file_path = tmp_path / "missing_type.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)

    with pytest.raises(ValueError, match="Geometry type not specified"):
        load_geometry(file_path)


def test_load_geometry_yaml_error(tmp_path: Path):
    """
    Test handling of malformed YAML.
    """
    file_path = tmp_path / "malformed.yaml"
    file_path.write_text("invalid: yaml: content: [")

    with pytest.raises(ValueError, match="Error parsing geometry file"):
        load_geometry(file_path)
