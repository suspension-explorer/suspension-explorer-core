"""Simple installed-length setup shims and their solved effects."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from kinematics.core.analysis import initial_pose
from kinematics.core.constraints import DistanceConstraint
from kinematics.core.enums import Axis, PointID, ShimType
from kinematics.core.input import build_suspension, build_sweep
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema import LengthShimConfig
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import (
    DoubleWishboneSuspension,
    MacPhersonSuspension,
    MultiLinkSuspension,
)
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent / "data"


def _geometry_data(filename: str) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        yaml.safe_load((DATA_DIR / filename).read_text(encoding="utf-8")),
    )


def _distance_constraint(
    suspension: DoubleWishboneSuspension,
    point_a: PointID,
    point_b: PointID,
) -> DistanceConstraint:
    endpoints = {point_a, point_b}
    return next(
        constraint
        for constraint in suspension.constraints()
        if isinstance(constraint, DistanceConstraint)
        and constraint.involved_points == endpoints
    )


def _rendered_distance(
    positions: dict[str, tuple[float, float, float]],
    point_a: str,
    point_b: str,
) -> float:
    a = positions[point_a]
    b = positions[point_b]
    return sum((a[index] - b[index]) ** 2 for index in range(3)) ** 0.5


def test_length_shim_adjustment_is_setup_minus_design() -> None:
    shim = LengthShimConfig(design_thickness=2.0, setup_thickness=5.5)

    assert shim.length_adjustment == pytest.approx(3.5)


def test_architectures_declare_supported_simple_shims() -> None:
    assert {
        ShimType.OUTBOARD_CAMBER,
        ShimType.PUSHROD,
        ShimType.TOE,
    } <= DoubleWishboneSuspension.SUPPORTED_SHIMS
    assert {ShimType.PUSHROD, ShimType.TOE} <= MultiLinkSuspension.SUPPORTED_SHIMS
    assert MacPhersonSuspension.SUPPORTED_SHIMS == frozenset({ShimType.TOE})


def test_pushrod_shim_requires_pushrod_rocker_actuation() -> None:
    data = _geometry_data("geometry.yaml")
    config = cast("dict[str, object]", data["config"])
    config["pushrod_shim"] = {"design_thickness": 0.0, "setup_thickness": 2.0}

    with pytest.raises(ValueError, match="requires pushrod-rocker actuation"):
        build_suspension(data)


def test_toe_shim_adjusts_a_fixed_toe_link() -> None:
    data = _geometry_data("geometry.yaml")
    config = cast("dict[str, object]", data["config"])
    hardpoints = cast("dict[str, object]", data["hardpoints"])
    config["steering"] = {"type": "none"}
    config["toe_shim"] = {"design_thickness": 2.0, "setup_thickness": 3.5}
    hardpoints["toe_link_inboard"] = hardpoints.pop("trackrod_inboard")
    hardpoints["toe_link_outboard"] = hardpoints.pop("trackrod_outboard")

    suspension = build_suspension(data)
    assert isinstance(suspension, DoubleWishboneSuspension)
    constraint = _distance_constraint(
        suspension,
        PointID.TOE_LINK_INBOARD,
        PointID.TOE_LINK_OUTBOARD,
    )
    design = suspension.initial_state()
    design_length = float(
        (
            design.get(PointID.TOE_LINK_INBOARD) - design.get(PointID.TOE_LINK_OUTBOARD)
        ).norm()
    )
    assert constraint.target_distance == pytest.approx(design_length + 1.5)


def test_axle_mirrors_simple_side_local_setup() -> None:
    data = _geometry_data("macpherson_axle_geometry.yaml")
    axle_config = cast("dict[str, object]", data["axle_config"])
    axle_config["left_setup"] = {
        "toe_shim": {"design_thickness": 0.5, "setup_thickness": 2.5}
    }

    suspension = build_suspension(data)
    assert isinstance(suspension, AxleSuspension)
    for side in (Side.LEFT, Side.RIGHT):
        corner = suspension.corners[side]
        assert isinstance(corner, MacPhersonSuspension)
        assert corner.config is not None
        assert corner.config.toe_shim is not None
        assert corner.config.toe_shim.length_adjustment == pytest.approx(2.0)
        assert corner.wheel_heading_link.length_adjustment == pytest.approx(2.0)


def test_pushrod_shim_adds_to_solved_pushrod_length_and_changes_ride_height() -> None:
    design_data = _geometry_data("corner_strut_rocker_geometry.yaml")
    setup_data = _geometry_data("corner_strut_rocker_geometry.yaml")
    setup_config = cast("dict[str, object]", setup_data["config"])
    setup_config["pushrod_shim"] = {
        "design_thickness": 1.0,
        "setup_thickness": 6.0,
    }

    design = build_suspension(design_data)
    setup = build_suspension(setup_data)
    assert isinstance(design, DoubleWishboneSuspension)
    assert isinstance(setup, DoubleWishboneSuspension)

    design_constraint = _distance_constraint(
        design,
        PointID.PUSHROD_OUTBOARD,
        PointID.PUSHROD_INBOARD,
    )
    setup_constraint = _distance_constraint(
        setup,
        PointID.PUSHROD_OUTBOARD,
        PointID.PUSHROD_INBOARD,
    )
    assert setup_constraint.target_distance == pytest.approx(
        design_constraint.target_distance + 5.0
    )

    sweep_spec = {
        "version": 1,
        "targets": [
            {
                "type": "element_length",
                "element": "damper",
                "side": "left",
                "mode": "relative",
                "values": [0.0],
            },
            {
                "type": "actuator_position",
                "actuator": "rack",
                "direction": {"axis": "y"},
                "mode": "relative",
                "values": [0.0],
            },
        ],
    }
    design_state = solve_sweep(design, build_sweep(sweep_spec, design))[0][0]
    setup_state = solve_sweep(setup, build_sweep(sweep_spec, setup))[0][0]

    setup_pushrod_length = float(
        (
            setup_state.get(PointID.PUSHROD_OUTBOARD)
            - setup_state.get(PointID.PUSHROD_INBOARD)
        ).norm()
    )
    design_wheel_height = float(design_state.get(PointID.WHEEL_CENTER)[Axis.Z])
    setup_wheel_height = float(setup_state.get(PointID.WHEEL_CENTER)[Axis.Z])
    assert setup_pushrod_length == pytest.approx(
        setup_constraint.target_distance, abs=1e-5
    )
    assert setup_wheel_height != pytest.approx(design_wheel_height, abs=1e-3)


def test_toe_shim_adds_to_solved_trackrod_length_and_changes_toe() -> None:
    design_data = _geometry_data("geometry.yaml")
    setup_data = _geometry_data("geometry.yaml")
    setup_config = cast("dict[str, object]", setup_data["config"])
    setup_config["toe_shim"] = {
        "design_thickness": 1.0,
        "setup_thickness": 4.0,
    }

    design = build_suspension(design_data)
    setup = build_suspension(setup_data)
    assert isinstance(design, DoubleWishboneSuspension)
    assert isinstance(setup, DoubleWishboneSuspension)

    design_constraint = _distance_constraint(
        design,
        PointID.TRACKROD_INBOARD,
        PointID.TRACKROD_OUTBOARD,
    )
    setup_constraint = _distance_constraint(
        setup,
        PointID.TRACKROD_INBOARD,
        PointID.TRACKROD_OUTBOARD,
    )
    assert setup_constraint.target_distance == pytest.approx(
        design_constraint.target_distance + 3.0
    )

    sweep_spec = {
        "version": 1,
        "targets": [
            {
                "type": "point",
                "point": "wheel_center",
                "side": "left",
                "direction": {"axis": "z"},
                "mode": "relative",
                "values": [0.0],
            },
            {
                "type": "actuator_position",
                "actuator": "rack",
                "direction": {"axis": "y"},
                "mode": "relative",
                "values": [0.0],
            },
        ],
    }
    design_state = solve_sweep(design, build_sweep(sweep_spec, design))[0][0]
    setup_state = solve_sweep(setup, build_sweep(sweep_spec, setup))[0][0]

    setup_trackrod_length = float(
        (
            setup_state.get(PointID.TRACKROD_INBOARD)
            - setup_state.get(PointID.TRACKROD_OUTBOARD)
        ).norm()
    )
    design_toe = design.compute_state_metrics(design_state)["toe_angle"]
    setup_toe = setup.compute_state_metrics(setup_state)["toe_angle"]
    assert setup_trackrod_length == pytest.approx(
        setup_constraint.target_distance, abs=1e-5
    )
    assert setup_toe != pytest.approx(design_toe, abs=1e-4)


def test_static_pose_applies_length_shims_for_live_preview() -> None:
    pushrod_design_data = _geometry_data("corner_strut_rocker_geometry.yaml")
    pushrod_setup_data = _geometry_data("corner_strut_rocker_geometry.yaml")
    pushrod_setup_config = cast("dict[str, object]", pushrod_setup_data["config"])
    pushrod_setup_config["pushrod_shim"] = {
        "design_thickness": 1.0,
        "setup_thickness": 6.0,
    }
    pushrod_design = initial_pose(build_suspension(pushrod_design_data))
    pushrod_setup = initial_pose(build_suspension(pushrod_setup_data))
    assert _rendered_distance(
        pushrod_setup.positions, "pushrod_outboard", "pushrod_inboard"
    ) == pytest.approx(
        _rendered_distance(
            pushrod_design.positions, "pushrod_outboard", "pushrod_inboard"
        )
        + 5.0,
        abs=1e-5,
    )
    assert pushrod_setup.positions["wheel_center"][2] != pytest.approx(
        pushrod_design.positions["wheel_center"][2], abs=1e-3
    )

    toe_design_data = _geometry_data("geometry.yaml")
    toe_setup_data = _geometry_data("geometry.yaml")
    toe_setup_config = cast("dict[str, object]", toe_setup_data["config"])
    toe_setup_config["toe_shim"] = {
        "design_thickness": 1.0,
        "setup_thickness": 4.0,
    }
    toe_design = initial_pose(build_suspension(toe_design_data))
    toe_setup = initial_pose(build_suspension(toe_setup_data))
    assert _rendered_distance(
        toe_setup.positions, "trackrod_inboard", "trackrod_outboard"
    ) == pytest.approx(
        _rendered_distance(
            toe_design.positions, "trackrod_inboard", "trackrod_outboard"
        )
        + 3.0,
        abs=1e-5,
    )
    assert toe_setup.positions["axle_outboard"] != pytest.approx(
        toe_design.positions["axle_outboard"], abs=1e-4
    )
