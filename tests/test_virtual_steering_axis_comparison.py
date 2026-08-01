"""Double-wishbone physical and virtual steering-axis comparison."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.rigid_motion import InstantaneousScrewAxis, ScrewAxisStatus
from kinematics.core.sweep import compute_sweep_tangents, solve_evaluated_sweep

DATA_DIR = Path(__file__).parent / "data"
STEERING_METRIC_KEYS = (
    "caster",
    "kpi",
    "steering_axis_offset_ground",
    "scrub_radius",
    "mechanical_trail",
)


def _solve(sweep_name: str):
    """Solve the mirrored double-wishbone axle with one comparison sweep."""
    suspension = load_geometry(DATA_DIR / "axle_geometry.yaml")
    sweep = load_sweep(DATA_DIR / sweep_name, suspension)
    return suspension, sweep, solve_evaluated_sweep(suspension, sweep)


def test_app_style_steer_sweep_recomputes_the_virtual_axis_per_frame() -> None:
    """The stored rack-response axis must follow each solved configuration."""
    _suspension, _sweep, evaluated = _solve("axle_steer_sweep.yaml")

    left_results = [frame[0] for frame in evaluated.instantaneous_steering_axes]
    assert all(result.status is ScrewAxisStatus.VALID for result in left_results)
    assert all(result.axis is not None for result in left_results)

    axes = [cast(InstantaneousScrewAxis, result.axis) for result in left_results]
    points = np.asarray([axis.point.data for axis in axes])
    directions = np.asarray([axis.direction.data for axis in axes])

    # A stale first-frame result would make both spans zero. The generous lower
    # bounds avoid turning this into a golden-value test for the axis fitter.
    assert float(np.max(np.ptp(points, axis=0))) > 1.0
    assert float(np.max(np.ptp(directions, axis=0))) > 1e-4


def test_wheel_centre_height_control_explains_virtual_metric_divergence() -> None:
    """Fixed wheel-centre Z adds wishbone jacking to the rack-response twist."""
    suspension, sweep, evaluated = _solve("axle_steer_sweep.yaml")
    tangents = compute_sweep_tangents(suspension, sweep, evaluated.states)
    midpoint = len(evaluated.states) // 2
    steering_dof = suspension.steering_actuator_dof()
    assert steering_dof is not None
    rack_tangent = next(
        field
        for field in tangents.per_step[midpoint]
        if steering_dof.matches(field.target)
    )

    lower_velocity = rack_tangent.velocity(
        PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    )
    upper_velocity = rack_tangent.velocity(
        PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD)
    )
    wheel_centre_velocity = rack_tangent.velocity(
        PointRef(Side.LEFT, PointID.WHEEL_CENTER)
    )

    # Both physical pivots translate together while the driven wheel-centre
    # height remains fixed. The fitted absolute twist therefore includes bump
    # translation as well as rotation about the kingpin line.
    assert np.linalg.norm(lower_velocity) > 0.05
    assert lower_velocity == pytest.approx(upper_velocity, abs=2e-5)
    assert lower_velocity[:2] == pytest.approx(np.zeros(2), abs=2e-8)
    assert wheel_centre_velocity[2] == pytest.approx(0.0, abs=2e-10)

    row = evaluated.metrics.rows[midpoint]
    assert isinstance(row, AxleMetricRows)
    left = row.corners[Side.LEFT]
    assert abs(left["scrub_radius_virtual"] - left["scrub_radius"]) > 1.0
    assert abs(left["mechanical_trail_virtual"] - left["mechanical_trail"]) > 1.0


def test_balljoint_height_control_recovers_the_physical_steering_axis() -> None:
    """With stationary wishbones, physical and virtual metrics coincide."""
    suspension, sweep, evaluated = _solve("axle_steer_balljoint_fixed_sweep.yaml")
    tangents = compute_sweep_tangents(suspension, sweep, evaluated.states)
    steering_dof = suspension.steering_actuator_dof()
    assert steering_dof is not None

    for index, raw_row in enumerate(evaluated.metrics.rows):
        assert isinstance(raw_row, AxleMetricRows)
        rack_tangent = next(
            field
            for field in tangents.per_step[index]
            if steering_dof.matches(field.target)
        )
        for side in (Side.LEFT, Side.RIGHT):
            lower_velocity = rack_tangent.velocity(
                PointRef(side, PointID.LOWER_WISHBONE_OUTBOARD)
            )
            upper_velocity = rack_tangent.velocity(
                PointRef(side, PointID.UPPER_WISHBONE_OUTBOARD)
            )
            assert lower_velocity == pytest.approx(np.zeros(3), abs=2e-10)
            assert upper_velocity == pytest.approx(np.zeros(3), abs=2e-10)

            row = raw_row.corners[side]
            for key in STEERING_METRIC_KEYS:
                assert row[f"{key}_virtual"] == pytest.approx(row[key], abs=1e-8)
