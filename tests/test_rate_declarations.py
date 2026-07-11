"""Migration contract for declarative corner derivative metrics."""

from pathlib import Path

import pytest

from kinematics.core.enums import Axis, PointID, TargetPositionMode
from kinematics.core.types import PointTarget, PointTargetAxis, SweepConfig
from kinematics.io import load_geometry
from kinematics.main import solve_sweep
from kinematics.metrics.catalog import get_default_corner_derivative_metrics
from kinematics.metrics.main import compute_metrics_for_state
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.sensitivity import compute_state_tangents

TEST_DATA = Path(__file__).parent / "data"
FD_STEP = 0.25


def _target(point: PointID, axis: Axis, value: float) -> PointTarget:
    return PointTarget(
        point_id=point,
        direction=PointTargetAxis(axis),
        value=value,
        mode=TargetPositionMode.ABSOLUTE,
    )


def _corner_definitions(corner):
    """Return the actual common and topology-specific declarations."""
    definitions = (
        *get_default_corner_derivative_metrics(corner),
        *corner.derivative_metric_definitions(),
    )
    return {definition.column_name: definition for definition in definitions}


def _solve_with_tangents(geometry_name: str):
    corner = load_geometry(TEST_DATA / geometry_name)
    initial = corner.initial_state()
    wheel_z = float(initial.get(PointID.WHEEL_CENTER)[Axis.Z]) + 10.0
    rack_y = float(initial.get(PointID.TRACKROD_INBOARD)[Axis.Y])
    targets = [
        _target(PointID.WHEEL_CENTER, Axis.Z, wheel_z),
        _target(PointID.TRACKROD_INBOARD, Axis.Y, rack_y),
    ]
    state = solve_sweep(corner, SweepConfig([[targets[0]], [targets[1]]]))[0][0]
    tangents, _ = compute_state_tangents(
        state,
        corner.constraints(),
        DerivedPointsManager(corner.derived_spec()),
        targets,
    )
    return corner, state, tangents, wheel_z, rack_y


def _metric_rows_at(
    corner,
    *,
    wheel_values: tuple[float, float],
    rack_values: tuple[float, float],
):
    states = solve_sweep(
        corner,
        SweepConfig(
            [
                [
                    _target(PointID.WHEEL_CENTER, Axis.Z, value)
                    for value in wheel_values
                ],
                [
                    _target(PointID.TRACKROD_INBOARD, Axis.Y, value)
                    for value in rack_values
                ],
            ]
        ),
    )[0]
    assert corner.config is not None
    return [compute_metrics_for_state(state, corner, corner.config) for state in states]


@pytest.mark.parametrize(
    ("column", "base_metric", "sign"),
    [
        ("camber_gain_deg_per_mm", "camber_deg", 1.0),
        ("bump_steer_deg_per_mm", "roadwheel_angle_deg", 1.0),
        ("caster_gain_deg_per_mm", "caster_deg", 1.0),
        ("kpi_gain_deg_per_mm", "kpi_deg", 1.0),
        ("half_track_rate_mm_per_mm", "half_track_change_mm", 1.0),
        ("wheel_recession_rate_mm_per_mm", "wheel_recession_mm", 1.0),
    ],
)
def test_bump_declarations_match_finite_difference(
    column: str,
    base_metric: str,
    sign: float,
) -> None:
    corner, state, tangents, wheel_z, rack_y = _solve_with_tangents("geometry.yaml")
    definition = _corner_definitions(corner)[column]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z - FD_STEP, wheel_z + FD_STEP),
        rack_values=(rack_y, rack_y),
    )
    finite_difference = sign * (high[base_metric] - low[base_metric]) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        finite_difference,
        rel=1e-3,
        abs=1e-5,
    )


@pytest.mark.parametrize(
    ("column", "base_metric"),
    [
        ("toe_vs_rack_deg_per_mm", "roadwheel_angle_deg"),
        ("camber_vs_rack_deg_per_mm", "camber_deg"),
    ],
)
def test_rack_declarations_match_finite_difference(
    column: str,
    base_metric: str,
) -> None:
    corner, state, tangents, wheel_z, rack_y = _solve_with_tangents("geometry.yaml")
    definition = _corner_definitions(corner)[column]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z, wheel_z),
        rack_values=(rack_y - FD_STEP, rack_y + FD_STEP),
    )
    finite_difference = (high[base_metric] - low[base_metric]) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        finite_difference,
        rel=1e-3,
        abs=1e-5,
    )


def test_damper_mr_declaration_matches_finite_difference() -> None:
    corner, state, tangents, wheel_z, rack_y = _solve_with_tangents(
        "corner_strut_geometry.yaml"
    )
    definition = _corner_definitions(corner)["damper_mr"]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z - FD_STEP, wheel_z + FD_STEP),
        rack_values=(rack_y, rack_y),
    )
    expected_damper_mr = -(
        high["damper_length_mm"] - low["damper_length_mm"]
    ) / (
        2 * FD_STEP
    )

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        expected_damper_mr,
        rel=1e-3,
        abs=1e-5,
    )
