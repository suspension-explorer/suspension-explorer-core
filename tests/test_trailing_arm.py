"""Trailing-arm schema, solving, spring, and presentation coverage."""

from math import degrees
from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.elements import (
    ElementType,
    TorsionElement,
    VariableLengthLinkElement,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.presentation import named_element_paths
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import Side
from kinematics.core.primitives.vector_utils.geometric import signed_angle_about_axis
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import TrailingArmSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def coilover() -> TrailingArmSuspension:
    suspension = load_geometry(TEST_DATA / "trailing_arm_coilover_geometry.yaml")
    assert isinstance(suspension, TrailingArmSuspension)
    return suspension


def _distance(state, point_a: PointID, point_b: PointID) -> float:
    return float((state.get(point_a) - state.get(point_b)).norm())


def _corner_metric_row(row: MetricRow | AxleMetricRows) -> MetricRow:
    """Narrow the shared sweep result type for a corner-only test."""
    assert not isinstance(row, AxleMetricRows)
    return row


def test_coilover_semi_trailing_arm_solves_about_an_oblique_axis(coilover):
    assert coilover.reported_type_key() == "trailing_arm"
    assert coilover.rack_attachment_point() is None
    assert coilover.damper_points() == (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)

    sweep = load_sweep(TEST_DATA / "trailing_arm_sweep.yaml", coilover)
    states, infos = solve_sweep(coilover, sweep)
    assert all(info.converged for info in infos)
    assert all(info.max_residual < 1e-3 for info in infos)

    design = coilover.initial_state()
    pivot_a = design.get(PointID.TRAILING_ARM_PIVOT_A)
    pivot_b = design.get(PointID.TRAILING_ARM_PIVOT_B)
    assert float(pivot_a[Axis.X]) != pytest.approx(float(pivot_b[Axis.X]))
    assert float(pivot_a[Axis.Z]) == pytest.approx(float(pivot_b[Axis.Z]))
    rigid_pairs = (
        (PointID.TRAILING_ARM_PIVOT_A, PointID.TRAILING_ARM_OUTBOARD),
        (PointID.TRAILING_ARM_PIVOT_B, PointID.TRAILING_ARM_OUTBOARD),
        (PointID.TRAILING_ARM_OUTBOARD, PointID.AXLE_INBOARD),
        (PointID.TRAILING_ARM_OUTBOARD, PointID.AXLE_OUTBOARD),
        (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
    )
    for state in states:
        for point_a, point_b in rigid_pairs:
            assert _distance(state, point_a, point_b) == pytest.approx(
                _distance(design, point_a, point_b), abs=1e-3
            )
        assert state.get(PointID.STRUT_TOP).data == pytest.approx(
            design.get(PointID.STRUT_TOP).data
        )

    metrics = compute_sweep_metrics(coilover, sweep, states)
    assert metrics.derivative_error is None
    first_metrics = _corner_metric_row(metrics.rows[0])
    final_metrics = _corner_metric_row(metrics.rows[-1])
    assert final_metrics["damper_length"] is not None
    assert final_metrics["deriv_damper_length_wrt_hub_z"] is not None
    first_camber = first_metrics["camber"]
    final_camber = final_metrics["camber"]
    first_toe = first_metrics["toe_angle"]
    final_toe = final_metrics["toe_angle"]
    assert first_camber is not None and final_camber is not None
    assert first_toe is not None and final_toe is not None
    assert abs(final_camber - first_camber) > 0.01
    assert abs(final_toe - first_toe) > 0.01


def test_torsion_bar_centres_on_pivot_with_direct_arm_twist_and_damper():
    torsion = load_geometry(TEST_DATA / "trailing_arm_torsion_geometry.yaml")
    assert isinstance(torsion, TrailingArmSuspension)
    bar = next(
        element for element in torsion.elements() if isinstance(element, TorsionElement)
    )
    assert bar.type is ElementType.TORSION_BAR
    assert bar.rotation_axis == (
        PointID.TORSION_BAR_AXIS_A,
        PointID.TORSION_BAR_AXIS_B,
    )
    damper = next(
        element
        for element in torsion.elements()
        if isinstance(element, VariableLengthLinkElement)
        and element.type is ElementType.DAMPER
    )
    assert (damper.point_a, damper.point_b) == (
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
    )
    labels = [path.label for path in named_element_paths(torsion.assembly())]
    assert "Semi-Trailing Arm Pivot Axis" not in labels
    assert "Semi-Trailing Arm Front Link" in labels
    assert "Semi-Trailing Arm Rear Link" in labels
    assert "Semi-Trailing Arm Torsion Bar" in labels
    assert not any("Reaction" in label for label in labels)
    assert "Damper" in labels

    sweep = load_sweep(TEST_DATA / "trailing_arm_sweep.yaml", torsion)
    states, infos = solve_sweep(torsion, sweep)
    assert all(info.converged for info in infos)
    design = torsion.initial_state()
    bar_axis = design.get(PointID.TORSION_BAR_AXIS_A)
    pivot_a = design.get(PointID.TRAILING_ARM_PIVOT_A)
    assert float(bar_axis[Axis.X]) == pytest.approx(float(pivot_a[Axis.X]))
    assert float(bar_axis[Axis.Z]) == pytest.approx(float(pivot_a[Axis.Z]))
    rigid_pairs = (
        (PointID.TRAILING_ARM_PIVOT_A, PointID.TRAILING_ARM_OUTBOARD),
        (PointID.TRAILING_ARM_PIVOT_B, PointID.TRAILING_ARM_OUTBOARD),
        (PointID.TRAILING_ARM_OUTBOARD, PointID.STRUT_BOTTOM),
    )
    for state in states:
        for point_a, point_b in rigid_pairs:
            assert _distance(state, point_a, point_b) == pytest.approx(
                _distance(design, point_a, point_b), abs=1e-3
            )
        assert state.get(PointID.STRUT_TOP).data == pytest.approx(
            design.get(PointID.STRUT_TOP).data
        )
    metrics = compute_sweep_metrics(torsion, sweep, states)
    assert metrics.derivative_error is None
    final_metrics = _corner_metric_row(metrics.rows[-1])
    torsion_twist = final_metrics["torsion_bar_twist"]
    assert torsion_twist is not None
    assert abs(torsion_twist) > 0.01
    axis_a = design.get(PointID.TORSION_BAR_AXIS_A)
    torsion_axis = (design.get(PointID.TORSION_BAR_AXIS_B) - axis_a).normalize()
    expected_twist = torsion.side.lateral_sign * degrees(
        signed_angle_about_axis(
            design.get(PointID.TRAILING_ARM_OUTBOARD),
            states[-1].get(PointID.TRAILING_ARM_OUTBOARD),
            axis_a,
            torsion_axis,
        )
    )
    assert torsion_twist == pytest.approx(expected_twist)
    assert final_metrics["deriv_torsion_bar_twist_wrt_hub_z"] is not None
    assert final_metrics["damper_length"] is not None
    assert final_metrics["deriv_damper_length_wrt_hub_z"] is not None


def test_trailing_arm_axle_mirrors_an_unsteered_pair():
    axle = load_geometry(TEST_DATA / "trailing_arm_axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    assert axle.reported_type_key() == "trailing_arm"
    assert axle.rack_attachment_points() is None
    assert all(
        isinstance(corner, TrailingArmSuspension) for corner in axle.corners.values()
    )
    left = axle.corners[Side.LEFT].initial_state()
    right = axle.corners[Side.RIGHT].initial_state()
    assert float(left.get(PointID.AXLE_OUTBOARD)[Axis.Y]) == pytest.approx(
        -float(right.get(PointID.AXLE_OUTBOARD)[Axis.Y])
    )


@pytest.mark.parametrize(
    ("point", "replacement", "message"),
    (
        (
            PointID.TRAILING_ARM_PIVOT_B,
            Point3([1000, 700, 300]),
            "horizontal, oblique",
        ),
        (
            PointID.TRAILING_ARM_OUTBOARD,
            Point3([1000, 850, 250]),
            "rearward of the pivot axis",
        ),
        (
            PointID.TRAILING_ARM_OUTBOARD,
            Point3([500, 1500, 300]),
            "must not lie on the trailing-arm pivot axis",
        ),
    ),
)
def test_trailing_arm_rejects_invalid_physical_conventions(
    coilover, point, replacement, message
):
    hardpoints = coilover.get_hardpoints_copy()
    hardpoints[point] = replacement
    with pytest.raises(ValueError, match=message):
        TrailingArmSuspension(
            name="invalid",
            side=Side.LEFT,
            hardpoints=hardpoints,
            config=coilover.config,
            spring_type=coilover.spring_type,
        )


def test_torsion_bar_requires_pivot_a_on_its_axis():
    torsion = load_geometry(TEST_DATA / "trailing_arm_torsion_geometry.yaml")
    assert isinstance(torsion, TrailingArmSuspension)
    hardpoints = torsion.get_hardpoints_copy()
    hardpoints[PointID.TORSION_BAR_AXIS_A] = Point3([990, 350, 300])
    hardpoints[PointID.TORSION_BAR_AXIS_B] = Point3([990, 650, 300])
    with pytest.raises(ValueError, match="PIVOT_A must lie on"):
        TrailingArmSuspension(
            name="detached torsion bar",
            side=Side.LEFT,
            hardpoints=hardpoints,
            config=torsion.config,
            spring_type=torsion.spring_type,
        )


def test_trailing_arm_schema_rejects_steering_and_shared_hardware(test_data_dir):
    import yaml

    data = yaml.safe_load(
        (test_data_dir / "trailing_arm_axle_geometry.yaml").read_text(encoding="utf-8")
    )
    data["axle_config"]["steering"]["type"] = "rack"
    with pytest.raises(ValueError, match="unsteered"):
        load_geometry_mapping(data)
    data["axle_config"]["steering"]["type"] = "none"
    data["axle_config"]["anti_roll"]["type"] = "u_bar"
    with pytest.raises(ValueError, match="anti-roll"):
        load_geometry_mapping(data)


def load_geometry_mapping(data):
    """Build one mapping without adding a CLI/YAML dependency to core code."""
    from kinematics.core.input import build_suspension

    return build_suspension(data)
