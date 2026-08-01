"""Independent inboard damper composition for double-wishbone rockers."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.constraints import DistanceConstraint
from kinematics.core.elements import ElementType, VariableLengthLinkElement
from kinematics.core.enums import CornerDamperType, PointID
from kinematics.core.input import build_suspension, build_sweep, parse_geometry_spec
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.schema.geometry import DoubleWishboneGeometrySpec
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.corner.mechanisms import (
    ActuationPushrodRocker,
    CornerDamperLinear,
    CornerDamperNone,
    CornerSpringTorsionBar,
)
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent / "data"
ROCKER_GEOMETRY = DATA_DIR / "corner_rocker_geometry.yaml"
DAMPER_GEOMETRY = DATA_DIR / "corner_rocker_damper_geometry.yaml"
AXLE_GEOMETRY = DATA_DIR / "axle_geometry_rocker.yaml"


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_omitted_damper_preserves_legacy_pushrod_rocker_geometry() -> None:
    spec = parse_geometry_spec(_read_yaml(ROCKER_GEOMETRY))
    assert isinstance(spec, DoubleWishboneGeometrySpec)
    assert spec.damper.type is CornerDamperType.NONE

    suspension = build_suspension(_read_yaml(ROCKER_GEOMETRY))
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert isinstance(suspension.damper, CornerDamperNone)
    assert suspension.damper_points() is None
    assert [coordinate.id for coordinate in suspension.drive_coordinates()] == ["rack"]


def test_linear_damper_is_fixed_to_chassis_and_rigidly_carried_by_rocker() -> None:
    spec = parse_geometry_spec(_read_yaml(DAMPER_GEOMETRY))
    assert isinstance(spec, DoubleWishboneGeometrySpec)
    assert spec.damper.type is CornerDamperType.LINEAR
    assert spec.damper.model_dump(mode="json") == {"type": "linear"}

    suspension = load_geometry(DAMPER_GEOMETRY)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert isinstance(suspension.actuation, ActuationPushrodRocker)
    assert isinstance(suspension.spring, CornerSpringTorsionBar)
    assert isinstance(suspension.damper, CornerDamperLinear)
    assert PointID.DAMPER_ROCKER not in suspension.actuation.external_point_ids

    assert suspension.damper_points() == (
        PointID.DAMPER_CHASSIS,
        PointID.DAMPER_ROCKER,
    )
    assert PointID.DAMPER_CHASSIS not in suspension.free_points()
    assert PointID.DAMPER_ROCKER in suspension.free_points()
    damper_coordinate = next(
        coordinate
        for coordinate in suspension.drive_coordinates()
        if coordinate.id == "damper"
    )
    assert damper_coordinate.point_keys == suspension.damper_points()
    assert {
        definition.column_name
        for definition in suspension.derivative_metric_definitions()
    } == {
        "deriv_rocker_angle_wrt_hub_z",
        "deriv_torsion_bar_twist_wrt_hub_z",
        "deriv_damper_length_wrt_hub_z",
    }

    damper = next(
        element
        for element in suspension.elements()
        if isinstance(element, VariableLengthLinkElement)
        and element.type is ElementType.DAMPER
    )
    assert (damper.point_a, damper.point_b) == suspension.damper_points()
    assert not any(
        isinstance(constraint, DistanceConstraint)
        and {constraint.p1, constraint.p2}
        == {PointID.DAMPER_CHASSIS, PointID.DAMPER_ROCKER}
        for constraint in suspension.constraints()
    )
    assert (
        sum(
            isinstance(constraint, DistanceConstraint)
            and PointID.DAMPER_ROCKER in constraint.involved_points
            for constraint in suspension.constraints()
        )
        == 3
    )

    metrics = suspension.compute_state_metrics(suspension.initial_state())
    expected = np.linalg.norm(
        suspension.hardpoints[PointID.DAMPER_ROCKER]
        - suspension.hardpoints[PointID.DAMPER_CHASSIS]
    )
    assert metrics["damper_length"] == pytest.approx(expected)

    sweep = build_sweep(
        {
            "targets": [
                {
                    "type": "element_length",
                    "element": "damper",
                    "side": "left",
                    "mode": "relative",
                    "values": [0.0, -1.0],
                },
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "mode": "relative",
                    "values": [0.0, 0.0],
                },
            ]
        },
        suspension,
    )
    states, infos = solve_sweep(suspension, sweep)
    assert all(info.converged for info in infos)
    design, compressed = states
    np.testing.assert_allclose(
        compressed.get(PointID.DAMPER_CHASSIS).data,
        design.get(PointID.DAMPER_CHASSIS).data,
    )
    assert not np.allclose(
        compressed.get(PointID.DAMPER_ROCKER).data,
        design.get(PointID.DAMPER_ROCKER).data,
    )
    for rocker_reference in (
        PointID.ROCKER_AXIS_A,
        PointID.ROCKER_AXIS_B,
        PointID.PUSHROD_INBOARD,
    ):
        design_radius = np.linalg.norm(
            design.get(PointID.DAMPER_ROCKER) - design.get(rocker_reference)
        )
        compressed_radius = np.linalg.norm(
            compressed.get(PointID.DAMPER_ROCKER) - compressed.get(rocker_reference)
        )
        assert compressed_radius == pytest.approx(design_radius, abs=1e-6)


def test_linear_damper_requires_explicit_valid_endpoints() -> None:
    missing = _read_yaml(DAMPER_GEOMETRY)
    missing["hardpoints"].pop("damper_chassis")
    with pytest.raises(ValueError, match="Missing required hardpoints: DAMPER_CHASSIS"):
        build_suspension(missing)

    coincident = _read_yaml(DAMPER_GEOMETRY)
    coincident["hardpoints"]["damper_rocker"] = coincident["hardpoints"][
        "damper_chassis"
    ]
    with pytest.raises(ValueError, match="endpoints must be distinct"):
        build_suspension(coincident)

    on_axis = _read_yaml(DAMPER_GEOMETRY)
    on_axis["hardpoints"]["damper_rocker"] = {"x": 0, "y": 340, "z": 450}
    with pytest.raises(ValueError, match="must not lie on the rocker axis"):
        build_suspension(on_axis)

    implicit = _read_yaml(DAMPER_GEOMETRY)
    implicit.pop("damper")
    with pytest.raises(ValueError, match="Invalid hardpoints: DAMPER_CHASSIS"):
        build_suspension(implicit)


@pytest.mark.parametrize(
    ("actuation_type", "spring_type", "message"),
    [
        ("direct", "torsion_bar", "requires pushrod-rocker"),
        ("pushrod_rocker", "coilover", "cannot be combined with a coilover"),
    ],
)
def test_schema_rejects_incompatible_independent_damper_combinations(
    actuation_type: str,
    spring_type: str,
    message: str,
) -> None:
    raw = _read_yaml(DAMPER_GEOMETRY)
    raw["actuation"]["type"] = actuation_type
    raw["spring"]["type"] = spring_type

    with pytest.raises(ValueError, match=message):
        parse_geometry_spec(raw)


def test_axle_composes_mirrored_independent_dampers_and_sided_coordinates() -> None:
    raw = _read_yaml(AXLE_GEOMETRY)
    raw["axle_config"]["damper"] = {"type": "linear"}
    raw["hardpoints"]["left"].update(
        {
            "damper_chassis": {"x": 0, "y": 150, "z": 500},
            "damper_rocker": {"x": 0, "y": 300, "z": 470},
        }
    )
    suspension = build_suspension(raw)
    assert isinstance(suspension, AxleSuspension)

    coordinates = suspension.drive_coordinates()
    assert [(coordinate.id, coordinate.side) for coordinate in coordinates] == [
        ("rack", None),
        ("damper", Side.LEFT),
        ("damper", Side.RIGHT),
    ]
    assert coordinates[1].point_keys == (
        PointRef(Side.LEFT, PointID.DAMPER_CHASSIS),
        PointRef(Side.LEFT, PointID.DAMPER_ROCKER),
    )
    assert coordinates[2].point_keys == (
        PointRef(Side.RIGHT, PointID.DAMPER_CHASSIS),
        PointRef(Side.RIGHT, PointID.DAMPER_ROCKER),
    )
