"""Steering actuation and toe-control topology tests."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from kinematics.core.constraints import PointOnLineConstraint
from kinematics.core.elements import ElementType, RackElement, RigidLinkElement
from kinematics.core.enums import Axis, PointID, SteeringType
from kinematics.core.input import build_suspension, build_sweep
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Geometry fixture must contain a mapping")
    return data


def _build_fixed_toe_axle(test_data_dir: Path, geometry_name: str) -> AxleSuspension:
    data = _read_yaml_mapping(test_data_dir / geometry_name)
    data["axle_config"]["steering"] = {"type": "none"}
    for hardpoints in data["hardpoints"].values():
        hardpoints["toe_link_inboard"] = hardpoints.pop("trackrod_inboard")
        hardpoints["toe_link_outboard"] = hardpoints.pop("trackrod_outboard")
    suspension = build_suspension(data)
    assert isinstance(suspension, AxleSuspension)
    return suspension


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
def test_rack_steering_uses_track_rods_not_toe_links(
    test_data_dir: Path,
    geometry_name: str,
) -> None:
    data = _read_yaml_mapping(test_data_dir / geometry_name)
    axle = build_suspension(data)
    assert isinstance(axle, AxleSuspension)

    state = axle.initial_state()
    for side in (Side.LEFT, Side.RIGHT):
        assert PointRef(side, PointID.TRACKROD_INBOARD) in state.free_points
        assert PointRef(side, PointID.TRACKROD_OUTBOARD) in state.free_points
        assert PointRef(side, PointID.TOE_LINK_INBOARD) not in state.positions
        assert PointRef(side, PointID.TOE_LINK_OUTBOARD) not in state.positions

    track_rods = [
        element
        for element in axle.elements()
        if isinstance(element, RigidLinkElement)
        and element.type is ElementType.TRACK_ROD
    ]
    assert len(track_rods) == 2
    assert not any(
        isinstance(element, RigidLinkElement) and element.type is ElementType.TOE_LINK
        for element in axle.elements()
    )


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
def test_steered_axle_requires_rack_control_target(
    test_data_dir: Path,
    geometry_name: str,
) -> None:
    axle = build_suspension(_read_yaml_mapping(test_data_dir / geometry_name))

    with pytest.raises(ValueError, match="target for actuator 'steering rack'"):
        build_sweep(
            {
                "version": 1,
                "targets": [
                    {
                        "point": "wheel_center",
                        "side": side,
                        "direction": {"axis": "z"},
                        "values": [0.0, 10.0],
                    }
                    for side in ("left", "right")
                ],
            },
            axle,
        )


def test_solve_revalidates_required_rack_control(test_data_dir: Path) -> None:
    axle = build_suspension(_read_yaml_mapping(test_data_dir / "axle_geometry.yaml"))
    sweep = SweepConfig(
        [
            [
                PointTarget(
                    PointRef(side, PointID.WHEEL_CENTER),
                    PointTargetAxis(Axis.Z),
                    0.0,
                )
            ]
            for side in (Side.LEFT, Side.RIGHT)
        ]
    )

    with pytest.raises(ValueError, match="target for actuator 'steering rack'"):
        solve_sweep(axle, sweep)


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
@pytest.mark.parametrize("rack_side", ["left", "right"])
def test_shared_rack_target_emits_derivatives_for_both_corners(
    test_data_dir: Path,
    geometry_name: str,
    rack_side: str,
) -> None:
    axle = build_suspension(_read_yaml_mapping(test_data_dir / geometry_name))
    sweep = build_sweep(
        {
            "version": 1,
            "targets": [
                *[
                    {
                        "point": "wheel_center",
                        "side": side,
                        "direction": {"axis": "z"},
                        "values": [0.0],
                    }
                    for side in ("left", "right")
                ],
                {
                    "point": "trackrod_inboard",
                    "side": rack_side,
                    "direction": {"axis": "y"},
                    "values": [0.0],
                },
            ],
        },
        axle,
    )

    states, _ = solve_sweep(axle, sweep)
    metrics = compute_sweep_metrics(axle, sweep, states)

    assert metrics.derivative_error is None
    row = metrics.rows[0]
    assert isinstance(row, AxleMetricRows)
    for side in (Side.LEFT, Side.RIGHT):
        assert (
            row.corners[side]["deriv_roadwheel_angle_wrt_rack_displacement"] is not None
        )


def test_steering_type_requires_matching_heading_link_hardpoints(
    test_data_dir: Path,
) -> None:
    rack_points_for_fixed_toe = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml")
    rack_points_for_fixed_toe["axle_config"]["steering"] = {"type": "none"}
    with pytest.raises(ValueError, match="Missing required hardpoints:.*TOE_LINK"):
        build_suspension(rack_points_for_fixed_toe)

    toe_points_for_rack = _read_yaml_mapping(test_data_dir / "axle_geometry.yaml")
    for hardpoints in toe_points_for_rack["hardpoints"].values():
        hardpoints["toe_link_inboard"] = hardpoints.pop("trackrod_inboard")
        hardpoints["toe_link_outboard"] = hardpoints.pop("trackrod_outboard")
    with pytest.raises(ValueError, match="Missing required hardpoints:.*TRACKROD"):
        build_suspension(toe_points_for_rack)


def test_camber_shim_preserves_nonsteered_toe_link_length(
    test_data_dir: Path,
) -> None:
    data = _read_yaml_mapping(test_data_dir / "geometry.yaml")
    data["config"]["steering"] = {"type": "none"}
    data["config"]["camber_shim"]["setup_thickness"] = 40.0
    hardpoints = data["hardpoints"]
    hardpoints["toe_link_inboard"] = hardpoints.pop("trackrod_inboard")
    hardpoints["toe_link_outboard"] = hardpoints.pop("trackrod_outboard")

    suspension = build_suspension(data)
    state = suspension.initial_state()
    design_length = (
        suspension.hardpoints[PointID.TOE_LINK_OUTBOARD]
        - suspension.hardpoints[PointID.TOE_LINK_INBOARD]
    ).norm()
    shimmed_length = (
        state.get(PointID.TOE_LINK_OUTBOARD) - state.get(PointID.TOE_LINK_INBOARD)
    ).norm()

    assert shimmed_length == pytest.approx(design_length, abs=1e-3)
    assert PointID.TOE_LINK_INBOARD in state.fixed_points


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
def test_nonsteered_axle_fixes_toe_link_inboards(
    test_data_dir: Path,
    geometry_name: str,
) -> None:
    axle = _build_fixed_toe_axle(test_data_dir, geometry_name)

    assert axle.config is not None
    assert axle.config.steering.type is SteeringType.NONE
    assert axle.rack_attachment_points() is None
    assert not any(isinstance(element, RackElement) for element in axle.elements())
    assert not any(
        isinstance(constraint, PointOnLineConstraint)
        for constraint in axle.constraints()
    )

    state = axle.initial_state()
    for side in (Side.LEFT, Side.RIGHT):
        inboard = PointRef(side, PointID.TOE_LINK_INBOARD)
        outboard = PointRef(side, PointID.TOE_LINK_OUTBOARD)
        assert inboard in state.fixed_points
        assert outboard in state.free_points
        assert axle.corners[side].rack_attachment_point() is None

    toe_links = [
        element
        for element in axle.elements()
        if isinstance(element, RigidLinkElement)
        and element.type is ElementType.TOE_LINK
    ]
    assert len(toe_links) == 2


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
def test_nonsteered_axle_solves_bump_without_a_rack_target(
    test_data_dir: Path,
    geometry_name: str,
) -> None:
    axle = _build_fixed_toe_axle(test_data_dir, geometry_name)
    sweep = build_sweep(
        {
            "version": 1,
            "targets": [
                {
                    "point": "wheel_center",
                    "side": side,
                    "direction": {"axis": "z"},
                    "mode": "relative",
                    "values": [0.0, 10.0],
                }
                for side in ("left", "right")
            ],
        },
        axle,
    )

    design = axle.initial_state()
    states, solve_infos = solve_sweep(axle, sweep)
    assert all(info.converged for info in solve_infos)
    assert all(info.max_residual < 1e-3 for info in solve_infos)

    for state in states:
        for side in (Side.LEFT, Side.RIGHT):
            inboard = PointRef(side, PointID.TOE_LINK_INBOARD)
            outboard = PointRef(side, PointID.TOE_LINK_OUTBOARD)
            assert state.get(inboard).data == pytest.approx(design.get(inboard).data)
            assert (state.get(outboard) - state.get(inboard)).norm() == pytest.approx(
                (design.get(outboard) - design.get(inboard)).norm(),
                abs=1e-3,
            )

    metrics = compute_sweep_metrics(axle, sweep, states)
    assert metrics.derivative_error is None
    assert metrics.tangent_solve_infos is not None
    assert all(not info.rank_deficient for info in metrics.tangent_solve_infos)
    row = metrics.rows[-1]
    assert isinstance(row, AxleMetricRows)
    assert row.axle["rack_displacement"] is None
    for corner in row.corners.values():
        assert corner["deriv_roadwheel_angle_wrt_hub_z"] is not None
        assert "deriv_roadwheel_angle_wrt_rack_displacement" not in corner


@pytest.mark.parametrize(
    "geometry_name",
    ["axle_geometry.yaml", "macpherson_axle_geometry.yaml"],
)
def test_nonsteered_axle_rejects_fixed_toe_link_sweep_target(
    test_data_dir: Path,
    geometry_name: str,
) -> None:
    axle = _build_fixed_toe_axle(test_data_dir, geometry_name)

    with pytest.raises(ValueError, match="TOE_LINK_INBOARD.*fixed"):
        build_sweep(
            {
                "version": 1,
                "targets": [
                    {
                        "point": "toe_link_inboard",
                        "side": "left",
                        "direction": {"axis": "y"},
                        "values": [0.0],
                    }
                ],
            },
            axle,
        )
