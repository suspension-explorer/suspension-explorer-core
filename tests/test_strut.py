from pathlib import Path
from typing import cast

import pytest
import yaml

from kinematics.constraints import ScalarTripleProductConstraint
from kinematics.core.enums import PointID
from kinematics.io import load_geometry
from kinematics.suspensions.corner import (
    DoubleWishboneCoiloverSuspension,
    DoubleWishboneSuspension,
)
from kinematics.suspensions.registry import (
    SUSPENSION_DEFINITIONS,
    get_suspension_class,
    get_suspension_definition,
    list_supported_types,
)

TEST_DATA = Path(__file__).parent / "data"


def _write_geometry(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _read_geometry(name: str) -> dict[str, object]:
    data = yaml.safe_load((TEST_DATA / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _mirror_hardpoint_y(data: dict[str, object]) -> None:
    """Mirror every geometry hardpoint through the vehicle XZ plane."""
    hardpoints = cast(dict[str, object], data["hardpoints"])
    for position in hardpoints.values():
        coordinates = cast(dict[str, float], position)
        coordinates["y"] = -coordinates["y"]


def test_explicit_corner_types_select_distinct_models() -> None:
    basic = load_geometry(TEST_DATA / "geometry.yaml")
    coilover = load_geometry(TEST_DATA / "corner_strut_geometry.yaml")

    assert type(basic) is DoubleWishboneSuspension
    assert type(coilover) is DoubleWishboneCoiloverSuspension
    assert not basic.has_strut
    assert coilover.has_strut


def test_type_selection_is_case_insensitive(tmp_path: Path) -> None:
    data = _read_geometry("corner_strut_geometry.yaml")
    data["type"] = "DOUBLE_WISHBONE_COILOVER"

    suspension = load_geometry(_write_geometry(tmp_path, data))

    assert type(suspension) is DoubleWishboneCoiloverSuspension


@pytest.mark.parametrize(
    ("declared_side", "axle_outboard_y", "expected_message"),
    [
        ("right", 950.0, "Side 'right' requires AXLE_OUTBOARD Y < 0"),
        ("left", -950.0, "Side 'left' requires AXLE_OUTBOARD Y > 0"),
    ],
)
def test_corner_rejects_declared_side_that_conflicts_with_hardpoints(
    tmp_path: Path,
    declared_side: str,
    axle_outboard_y: float,
    expected_message: str,
) -> None:
    data = _read_geometry("geometry.yaml")
    data["side"] = declared_side
    hardpoints = cast(dict[str, object], data["hardpoints"])
    axle_outboard = cast(dict[str, float], hardpoints["axle_outboard"])
    axle_outboard["y"] = axle_outboard_y

    with pytest.raises(ValueError, match=expected_message):
        load_geometry(_write_geometry(tmp_path, data))


@pytest.mark.parametrize("fixture", ["geometry.yaml", "corner_strut_geometry.yaml"])
def test_corner_accepts_hardpoints_matching_each_declared_side(
    tmp_path: Path, fixture: str
) -> None:
    left_data = _read_geometry(fixture)
    left_suspension = load_geometry(_write_geometry(tmp_path, left_data))
    assert left_suspension.side.name == "LEFT"

    right_data = _read_geometry(fixture)
    right_data["side"] = "right"
    _mirror_hardpoint_y(right_data)
    right_suspension = load_geometry(_write_geometry(tmp_path, right_data))
    assert right_suspension.side.name == "RIGHT"


@pytest.mark.parametrize("missing", ["strut_top", "strut_bottom"])
def test_coilover_rejects_missing_required_hardpoint(
    tmp_path: Path, missing: str
) -> None:
    data = _read_geometry("corner_strut_geometry.yaml")
    hardpoints = cast(dict[str, object], data["hardpoints"])
    assert isinstance(hardpoints, dict)
    del hardpoints[missing]

    with pytest.raises(ValueError, match="Missing required hardpoints"):
        load_geometry(_write_geometry(tmp_path, data))


def test_basic_type_rejects_coilover_hardpoints(tmp_path: Path) -> None:
    data = _read_geometry("corner_strut_geometry.yaml")
    data["type"] = "double_wishbone"

    with pytest.raises(ValueError, match="Invalid hardpoints"):
        load_geometry(_write_geometry(tmp_path, data))


def test_basic_type_does_not_accept_variant_points() -> None:
    assert PointID.STRUT_TOP not in DoubleWishboneSuspension.all_valid_points()
    assert PointID.STRUT_BOTTOM not in DoubleWishboneSuspension.all_valid_points()


def test_corner_registry_has_one_complete_definition_per_type() -> None:
    expected_types = {
        "double_wishbone",
        "double_wishbone_front",
        "double_wishbone_rear",
        "double_wishbone_coilover",
        "double_wishbone_axle",
        "double_wishbone_pushrod_rocker",
        "double_wishbone_pushrod_rocker_arb",
        "double_wishbone_pushrod_rocker_axle",
    }
    canonical_types = {
        "double_wishbone",
        "double_wishbone_coilover",
        "double_wishbone_axle",
        "double_wishbone_pushrod_rocker",
        "double_wishbone_pushrod_rocker_arb",
        "double_wishbone_pushrod_rocker_axle",
    }

    assert set(list_supported_types()) == expected_types
    assert {definition.type_key for definition in SUSPENSION_DEFINITIONS} == (
        canonical_types
    )
    for type_key in canonical_types:
        definition = get_suspension_definition(type_key)
        assert definition is not None
        assert definition.suspension_type is get_suspension_class(type_key)
        assert definition.spec_type.model_fields["type"].default == type_key


def test_coilover_topology_adds_moving_pickup_and_attachment_constraints() -> None:
    suspension = load_geometry(TEST_DATA / "corner_strut_geometry.yaml")

    assert PointID.STRUT_BOTTOM in suspension.free_points()
    assert PointID.STRUT_TOP not in suspension.free_points()
    assert PointID.STRUT_TOP in suspension.OUTPUT_POINTS
    assert PointID.STRUT_BOTTOM in suspension.OUTPUT_POINTS
    constraints = suspension.constraints()
    assert len(constraints) == 21
    assert (
        sum(
            isinstance(constraint, ScalarTripleProductConstraint)
            for constraint in constraints
        )
        == 1
    )


def test_legacy_configuration_key_remains_loadable(tmp_path: Path) -> None:
    data = _read_geometry("geometry.yaml")
    data["configuration"] = data.pop("config")

    suspension = load_geometry(_write_geometry(tmp_path, data))

    assert type(suspension) is DoubleWishboneSuspension
