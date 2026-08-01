"""Geometry-independent sweep vocabulary and deferred compatibility errors."""

import json
from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import PointID
from kinematics.core.input import build_sweep, parse_sweep_spec
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.schema.sweep import ElementLengthTargetSpec
from kinematics.core.targeting import (
    ACTUATOR_POSITION_TARGET_IDS,
    ELEMENT_LENGTH_TARGET_IDS,
    POINT_TARGET_IDS,
    TargetKind,
    sweep_target_vocabulary,
)

EXPECTED_POINT_IDS = (
    "lower_wishbone_inboard_front",
    "lower_wishbone_inboard_rear",
    "lower_wishbone_outboard",
    "upper_wishbone_inboard_front",
    "upper_wishbone_inboard_rear",
    "upper_wishbone_outboard",
    "pushrod_inboard",
    "pushrod_outboard",
    "trackrod_inboard",
    "trackrod_outboard",
    "toe_link_inboard",
    "toe_link_outboard",
    "axle_inboard",
    "axle_outboard",
    "axle_midpoint",
    "strut_top",
    "strut_bottom",
    "wheel_center",
    "wheel_inboard",
    "wheel_outboard",
    "wheel_contact_centre",
    "camber_shim_face_point_a",
    "camber_shim_face_point_b",
    "camber_shim_face_normal",
    "rocker_axis_a",
    "rocker_axis_b",
    "droplink_rocker",
    "droplink_u_bar",
    "arb_u_bar_axis_a",
    "arb_u_bar_axis_b",
    "heave_link_rocker",
    "arb_t_bar_pivot",
    "droplink_t_bar",
    "trailing_arm_pivot_a",
    "trailing_arm_pivot_b",
    "trailing_arm_outboard",
    "torsion_bar_axis_a",
    "torsion_bar_axis_b",
    "damper_chassis",
    "damper_rocker",
)


def _point_target(point: str, *, side: str | None = None) -> dict[str, object]:
    target: dict[str, object] = {
        "kind": "point",
        "point": point,
        "direction": {"axis": "z"},
        "values": [0.0],
    }
    if side is not None:
        target["side"] = side
    return target


def _element_target(element: str, *, side: str | None = None) -> dict[str, object]:
    target: dict[str, object] = {
        "kind": "element_length",
        "element": element,
        "values": [0.0],
    }
    if side is not None:
        target["side"] = side
    return target


def _actuator_target(actuator: str, *, side: str | None = None) -> dict[str, object]:
    target: dict[str, object] = {
        "kind": "actuator_position",
        "actuator": actuator,
        "direction": {"axis": "y"},
        "values": [0.0],
    }
    if side is not None:
        target["side"] = side
    return target


def test_public_sweep_target_vocabulary_is_complete_stable_and_json_native() -> None:
    assert POINT_TARGET_IDS == EXPECTED_POINT_IDS
    assert ACTUATOR_POSITION_TARGET_IDS == ("rack",)
    assert ELEMENT_LENGTH_TARGET_IDS == ("damper", "heave_link")

    vocabulary = sweep_target_vocabulary()
    assert vocabulary["positions"][:2] == [
        {
            "kind": "actuator_position",
            "id": "rack",
            "label": "Rack",
            "featured": True,
            "side_policy": "shared",
        },
        {
            "kind": "point",
            "id": "wheel_center",
            "label": "Wheel Center",
            "featured": True,
            "side_policy": "corner",
        },
    ]
    point_entries = [
        item for item in vocabulary["positions"] if item["kind"] == "point"
    ]
    assert {item["id"] for item in point_entries} == set(EXPECTED_POINT_IDS)
    assert all(
        item["featured"] == (item["id"] == "wheel_center") for item in point_entries
    )
    shared_point_ids = {
        "arb_u_bar_axis_a",
        "arb_u_bar_axis_b",
        "arb_t_bar_pivot",
    }
    assert {
        item["id"] for item in point_entries if item["side_policy"] == "shared"
    } == shared_point_ids
    assert all(
        item["side_policy"]
        == ("shared" if item["id"] in shared_point_ids else "corner")
        for item in point_entries
    )
    assert all(
        next(item for item in point_entries if item["id"] == point_id)["side_policy"]
        == "corner"
        for point_id in (
            "heave_link_rocker",
            "droplink_rocker",
            "droplink_u_bar",
            "droplink_t_bar",
        )
    )
    assert vocabulary["element_lengths"] == [
        {
            "id": "damper",
            "label": "Damper Length",
            "unit": "mm",
            "featured": True,
            "side_policy": "corner",
        },
        {
            "id": "heave_link",
            "label": "Heave Link Length",
            "unit": "mm",
            "featured": False,
            "side_policy": "shared",
        },
    ]
    assert (
        next(
            item["label"]
            for item in vocabulary["positions"]
            if item["id"] == "arb_u_bar_axis_a"
        )
        == "ARB U-Bar Axis A"
    )
    assert "not_assigned" not in json.dumps(vocabulary)
    json.dumps(vocabulary)


def test_schema_rejects_reserved_point_and_unknown_element_ids() -> None:
    with pytest.raises(ValueError, match="'not_assigned' is reserved"):
        parse_sweep_spec({"targets": [_point_target("not_assigned")]})

    with pytest.raises(
        ValueError,
        match=r"Unknown element-length target ID 'pushrod'.*damper, heave_link",
    ):
        parse_sweep_spec({"targets": [_element_target("pushrod")]})

    with pytest.raises(ValueError, match="Unknown actuator-position target ID 'bar'"):
        parse_sweep_spec({"targets": [_actuator_target("bar")]})


@pytest.mark.parametrize(
    "target",
    [
        _actuator_target("rack", side="center"),
        _element_target("heave_link", side="center"),
    ],
)
def test_shared_targets_are_authored_by_omitting_side(
    target: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        parse_sweep_spec({"targets": [target]})


def test_known_element_parses_without_geometry_but_availability_is_deferred(
    test_data_dir: Path,
) -> None:
    raw = {"targets": [_element_target("heave_link")]}
    spec = parse_sweep_spec(raw)
    target = spec.targets[0]
    assert isinstance(target, ElementLengthTargetSpec)
    assert target.element == "heave_link"

    corner = load_geometry(test_data_dir / "corner_strut_geometry.yaml")
    with pytest.raises(
        ValueError,
        match=(
            r"Driveable element-length target 'heave_link' is unavailable"
            r".*Available .*damper"
        ),
    ):
        build_sweep(raw, corner)


def test_known_actuator_parses_without_geometry_but_availability_is_deferred(
    test_data_dir: Path,
) -> None:
    raw = {"targets": [_actuator_target("rack")]}
    spec = parse_sweep_spec(raw)
    assert spec.targets[0].kind == "actuator_position"

    unsteered = load_geometry(test_data_dir / "trailing_arm_torsion_geometry.yaml")
    with pytest.raises(
        ValueError,
        match=r"Driveable actuator-position target 'rack' is unavailable",
    ):
        build_sweep(raw, unsteered)


def test_point_presence_mobility_and_side_rules_fail_actionably(
    test_data_dir: Path,
) -> None:
    corner = load_geometry(test_data_dir / "corner_strut_geometry.yaml")
    unsteered_corner = load_geometry(
        test_data_dir / "trailing_arm_torsion_geometry.yaml"
    )
    axle = load_geometry(test_data_dir / "trailing_arm_axle_geometry.yaml")

    with pytest.raises(ValueError, match=r"'TRAILING_ARM_OUTBOARD' is not present"):
        build_sweep(
            {"targets": [_point_target("trailing_arm_outboard", side="left")]},
            corner,
        )
    with pytest.raises(ValueError, match=r"'LOWER_WISHBONE_INBOARD_FRONT' is fixed"):
        build_sweep(
            {
                "targets": [
                    _point_target("lower_wishbone_inboard_front", side="left")
                ]
            },
            corner,
        )
    sweep = build_sweep(
        {"targets": [_point_target("wheel_center", side="left")]},
        unsteered_corner,
    )
    assert sweep.target_sweeps[0][0].required_points == (PointID.WHEEL_CENTER,)
    with pytest.raises(ValueError, match="requires side left"):
        build_sweep(
            {"targets": [_point_target("wheel_center")]},
            unsteered_corner,
        )
    with pytest.raises(
        ValueError,
        match=r"unavailable on side 'right'.*Available sides: left",
    ):
        build_sweep(
            {"targets": [_point_target("wheel_center", side="right")]},
            unsteered_corner,
        )
    with pytest.raises(ValueError, match="requires side left or right"):
        build_sweep(
            {"targets": [_point_target("wheel_center")]},
            axle,
        )
    with pytest.raises(ValueError, match="requires side left"):
        build_sweep(
            {"targets": [_point_target("trackrod_inboard")]},
            axle,
        )


def test_element_side_rules_remain_topology_specific(test_data_dir: Path) -> None:
    corner = load_geometry(test_data_dir / "corner_strut_geometry.yaml")
    unsteered_corner = load_geometry(
        test_data_dir / "trailing_arm_torsion_geometry.yaml"
    )
    axle = load_geometry(test_data_dir / "trailing_arm_axle_geometry.yaml")

    sweep = build_sweep(
        {"targets": [_element_target("damper", side="left")]},
        unsteered_corner,
    )
    assert sweep.target_sweeps[0][0].coordinate_id == "damper"
    assert next(
        coordinate
        for coordinate in unsteered_corner.drive_coordinates()
        if coordinate.id == "damper"
    ).side is Side.LEFT
    with pytest.raises(ValueError, match="requires side left"):
        build_sweep(
            {"targets": [_element_target("damper")]},
            unsteered_corner,
        )
    with pytest.raises(
        ValueError,
        match=r"unavailable on side 'right'.*Available sides: left",
    ):
        build_sweep(
            {"targets": [_element_target("damper", side="right")]},
            unsteered_corner,
        )
    with pytest.raises(ValueError, match="shared and does not accept a side"):
        corner.resolve_drive_coordinate(
            "rack",
            Side.LEFT,
            TargetKind.ACTUATOR_POSITION,
        )
    with pytest.raises(ValueError, match="requires side left or right"):
        build_sweep(
            {"targets": [_element_target("damper")]},
            axle,
        )

    assert axle.resolve_target_key(
        PointID.ARB_U_BAR_AXIS_A,
        None,
    ) == PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)
    with pytest.raises(ValueError, match="is shared and does not accept a side"):
        axle.resolve_target_key(PointID.ARB_U_BAR_AXIS_A, Side.LEFT)
