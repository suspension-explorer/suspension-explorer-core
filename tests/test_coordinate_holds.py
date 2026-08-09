"""Shared coordinate holds in nonlinear sweeps and local responses."""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis
from kinematics.core.input import build_sweep
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent / "data"


def test_rack_hold_is_captured_once_during_bump_sweep() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = build_sweep(
        {
            "steps": 3,
            "targets": [
                {
                    "type": "point",
                    "point": "wheel_center",
                    "side": "left",
                    "direction": {"axis": "z"},
                    "start": -20,
                    "stop": 20,
                },
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "hold": True,
                },
            ],
        },
        suspension,
    )

    assert len(sweep.target_sweeps) == 1
    assert tuple(coordinate.id for coordinate in sweep.hold.coordinates) == ("rack",)
    rack = sweep.hold.coordinates[0]
    reference = rack.measure(suspension.initial_state().positions)
    states, _ = solve_sweep(suspension, sweep)

    assert [rack.measure(state.positions) for state in states] == pytest.approx(
        [reference] * len(states), abs=2e-6
    )


def test_element_length_hold_closes_steering_sweep() -> None:
    suspension = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    sweep = build_sweep(
        {
            "steps": 3,
            "targets": [
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "start": -5,
                    "stop": 5,
                },
                {
                    "type": "element_length",
                    "element": "damper",
                    "side": "left",
                    "hold": True,
                },
            ],
        },
        suspension,
    )

    damper = sweep.hold.coordinates[0]
    reference = damper.measure(suspension.initial_state().positions)
    states, _ = solve_sweep(suspension, sweep)

    assert [damper.measure(state.positions) for state in states] == pytest.approx(
        [reference] * len(states), abs=2e-6
    )


def test_point_axis_hold_closes_steering_sweep() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    sweep = build_sweep(
        {
            "steps": 3,
            "targets": [
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "start": -5,
                    "stop": 5,
                },
                {
                    "type": "point",
                    "point": "wheel_center",
                    "side": "left",
                    "direction": {"axis": "z"},
                    "hold": True,
                },
            ],
        },
        suspension,
    )

    wheel_z = sweep.hold.coordinates[0]
    reference = float(
        suspension.initial_state().positions[wheel_z.point_keys[0]][Axis.Z]
    )
    states, _ = solve_sweep(suspension, sweep)

    assert [wheel_z.measure(state.positions) for state in states] == pytest.approx(
        [reference] * len(states), abs=2e-6
    )
