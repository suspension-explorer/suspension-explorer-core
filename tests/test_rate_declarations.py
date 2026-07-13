"""Migration contract for declarative corner derivative metrics."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.metrics.catalog import get_default_corner_derivative_metrics
from kinematics.core.metrics.main import compute_metrics_for_state
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.enums import Axis, PointID, TargetPositionMode
from kinematics.core.sensitivity import compute_state_tangents
from kinematics.core.sweep import solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig

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
    trackrod_inboard_y = float(initial.get(PointID.TRACKROD_INBOARD)[Axis.Y])
    targets = [
        _target(PointID.WHEEL_CENTER, Axis.Z, wheel_z),
        _target(PointID.TRACKROD_INBOARD, Axis.Y, trackrod_inboard_y),
    ]
    state = solve_sweep(corner, SweepConfig([[targets[0]], [targets[1]]]))[0][0]
    tangents, _ = compute_state_tangents(
        state,
        corner.constraints(),
        DerivedPointsManager(corner.derived_spec()),
        targets,
    )
    return corner, state, tangents, wheel_z, trackrod_inboard_y


def _metric_rows_at(
    corner,
    *,
    wheel_values: tuple[float, float],
    trackrod_inboard_y_values: tuple[float, float],
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
                    for value in trackrod_inboard_y_values
                ],
            ]
        ),
    )[0]
    assert corner.config is not None
    return [compute_metrics_for_state(state, corner, corner.config) for state in states]


@pytest.mark.parametrize(
    ("column", "base_metric", "sign"),
    [
        ("deriv_camber_wrt_hub_z", "camber", 1.0),
        ("deriv_roadwheel_angle_wrt_hub_z", "roadwheel_angle", 1.0),
        ("deriv_caster_wrt_hub_z", "caster", 1.0),
        ("deriv_kpi_wrt_hub_z", "kpi", 1.0),
        ("deriv_half_track_wrt_hub_z", "half_track", 1.0),
    ],
)
def test_hub_z_declarations_match_finite_difference(
    column: str,
    base_metric: str,
    sign: float,
) -> None:
    corner, state, tangents, wheel_z, trackrod_inboard_y = _solve_with_tangents(
        "geometry.yaml"
    )
    definition = _corner_definitions(corner)[column]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z - FD_STEP, wheel_z + FD_STEP),
        trackrod_inboard_y_values=(trackrod_inboard_y, trackrod_inboard_y),
    )
    finite_difference = sign * (high[base_metric] - low[base_metric]) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        finite_difference,
        rel=1e-3,
        abs=1e-5,
    )


def test_wheel_center_x_derivative_matches_finite_difference() -> None:
    corner, state, tangents, wheel_z, trackrod_inboard_y = _solve_with_tangents(
        "geometry.yaml"
    )
    definition = _corner_definitions(corner)["deriv_wheel_center_x_wrt_hub_z"]
    states = solve_sweep(
        corner,
        SweepConfig(
            [
                [
                    _target(PointID.WHEEL_CENTER, Axis.Z, wheel_z - FD_STEP),
                    _target(PointID.WHEEL_CENTER, Axis.Z, wheel_z + FD_STEP),
                ],
                [
                    _target(
                        PointID.TRACKROD_INBOARD,
                        Axis.Y,
                        trackrod_inboard_y,
                    ),
                    _target(
                        PointID.TRACKROD_INBOARD,
                        Axis.Y,
                        trackrod_inboard_y,
                    ),
                ],
            ]
        ),
    )[0]
    finite_difference = (
        float(states[1].get(PointID.WHEEL_CENTER)[Axis.X])
        - float(states[0].get(PointID.WHEEL_CENTER)[Axis.X])
    ) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        finite_difference,
        rel=1e-3,
        abs=1e-5,
    )


@pytest.mark.parametrize(
    ("column", "base_metric"),
    [
        ("deriv_roadwheel_angle_wrt_trackrod_inboard_y", "roadwheel_angle"),
        ("deriv_camber_wrt_trackrod_inboard_y", "camber"),
    ],
)
def test_trackrod_inboard_y_declarations_match_finite_difference(
    column: str,
    base_metric: str,
) -> None:
    corner, state, tangents, wheel_z, trackrod_inboard_y = _solve_with_tangents(
        "geometry.yaml"
    )
    definition = _corner_definitions(corner)[column]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z, wheel_z),
        trackrod_inboard_y_values=(
            trackrod_inboard_y - FD_STEP,
            trackrod_inboard_y + FD_STEP,
        ),
    )
    finite_difference = (high[base_metric] - low[base_metric]) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        finite_difference,
        rel=1e-3,
        abs=1e-5,
    )


def test_damper_length_declaration_matches_finite_difference() -> None:
    corner, state, tangents, wheel_z, trackrod_inboard_y = _solve_with_tangents(
        "corner_strut_geometry.yaml"
    )
    definition = _corner_definitions(corner)["deriv_damper_length_wrt_hub_z"]
    low, high = _metric_rows_at(
        corner,
        wheel_values=(wheel_z - FD_STEP, wheel_z + FD_STEP),
        trackrod_inboard_y_values=(trackrod_inboard_y, trackrod_inboard_y),
    )
    expected_derivative = (high["damper_length"] - low["damper_length"]) / (2 * FD_STEP)

    assert definition.evaluate_from_tangents(state, tangents) == pytest.approx(
        expected_derivative,
        rel=1e-3,
        abs=1e-5,
    )
