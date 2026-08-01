"""Element-length sweep targets across schema, solver, and presentation."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.analysis import initial_pose, sweep_parameters
from kinematics.core.enums import Axis, PointID, Scope, TargetPositionMode
from kinematics.core.input import build_sweep, parse_sweep_spec
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.point_ref import Side
from kinematics.core.sensitivity import compute_state_tangents
from kinematics.core.solver import ResidualComputer, convert_targets_to_absolute
from kinematics.core.sweep import compute_sweep_tangents, solve_sweep
from kinematics.core.targeting import (
    ElementLengthTarget,
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetKind,
)

DATA_DIR = Path(__file__).parent / "data"
FD_STEP = 1.0e-5


def _element_sweep(
    suspension,
    values: list[float],
    *,
    rack_values: list[float] | None = None,
) -> SweepConfig:
    targets: list[dict[str, object]] = [
        {
            "kind": "element_length",
            "element": "damper",
            "side": "left",
            "mode": "relative",
            "values": values,
        }
    ]
    if rack_values is not None:
        targets.append(
            {
                "kind": "point",
                "point": "trackrod_inboard",
                "side": "left",
                "direction": {"axis": "y"},
                "mode": "relative",
                "values": rack_values,
            }
        )
    return build_sweep({"version": 1, "targets": targets}, suspension)


def test_element_target_modes_resolve_once_against_setup_length() -> None:
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    coordinate = next(
        item for item in suspension.drive_coordinates() if item.id == "damper"
    )
    state = suspension.initial_state()
    setup_length = coordinate.target(0.0).measure(state.positions)

    relative = coordinate.target(12.5, TargetPositionMode.RELATIVE)
    absolute = coordinate.target(321.0, TargetPositionMode.ABSOLUTE)
    resolved = convert_targets_to_absolute([relative, absolute], state)

    assert resolved[0].mode is TargetPositionMode.ABSOLUTE
    assert resolved[0].value == pytest.approx(setup_length + 12.5)
    assert resolved[1] is absolute
    assert resolved[1].value == pytest.approx(321.0)

    with pytest.raises(ValueError, match="non-negative"):
        coordinate.target(-1.0, TargetPositionMode.ABSOLUTE)
    with pytest.raises(ValueError, match="finite"):
        coordinate.target(float("nan"), TargetPositionMode.RELATIVE)
    with pytest.raises(ValueError, match=r"Sweep target 0.*non-negative"):
        convert_targets_to_absolute(
            [coordinate.target(-(setup_length + 1.0))],
            state,
        )


def test_legacy_and_mixed_yaml_specs_remain_typed_and_round_trip() -> None:
    raw = {
        "version": 1,
        "targets": [
            {
                "point": "trackrod_inboard",
                "side": "left",
                "direction": {"axis": "y"},
                "values": [0.0, 1.0],
            },
            {
                "kind": "element_length",
                "element": "damper",
                "side": "left",
                "values": [0.0, 0.0],
            },
        ],
    }
    spec = parse_sweep_spec(raw)
    dumped = spec.model_dump(mode="json")
    reparsed = parse_sweep_spec(dumped)

    assert dumped["targets"][0]["kind"] == "point"
    assert dumped["targets"][0]["side"] == "left"
    assert dumped["targets"][1]["kind"] == "element_length"
    assert dumped["targets"][1]["side"] == "left"
    assert reparsed == spec

    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = build_sweep(raw, suspension)
    assert isinstance(sweep.target_sweeps[0][0], PointTarget)
    assert isinstance(sweep.target_sweeps[1][0], ElementLengthTarget)

    invalid = {
        "targets": [
            {
                "kind": "element_length",
                "element": "damper",
                "side": "left",
                "point": "wheel_center",
                "direction": {"axis": "z"},
                "values": [0.0],
            }
        ]
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        parse_sweep_spec(invalid)


@pytest.mark.parametrize(
    "geometry_file",
    [
        "corner_strut_geometry.yaml",
        "corner_strut_rocker_geometry.yaml",
        "corner_rocker_damper_geometry.yaml",
        "macpherson_geometry.yaml",
        "trailing_arm_coilover_geometry.yaml",
    ],
)
def test_supported_corner_damper_topologies_solve_length_sweeps(
    geometry_file: str,
) -> None:
    suspension = load_geometry(DATA_DIR / geometry_file)
    rack_values = [0.0, 0.0, 0.0] if suspension.actuator_dofs() else None
    sweep = _element_sweep(
        suspension,
        [0.0, 1.0, -1.0],
        rack_values=rack_values,
    )
    states, infos = solve_sweep(suspension, sweep)
    absolute = [
        convert_targets_to_absolute([target], suspension.initial_state())[0].value
        for target in sweep.target_sweeps[0]
    ]

    assert all(info.converged for info in infos)
    for state, expected, target in zip(states, absolute, sweep.target_sweeps[0]):
        assert target.measure(state.positions) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    "geometry_file",
    ["corner_strut_geometry.yaml", "macpherson_geometry.yaml"],
    ids=("free_endpoint", "derived_endpoint"),
)
def test_element_target_jacobian_matches_central_difference(
    geometry_file: str,
) -> None:
    suspension = load_geometry(DATA_DIR / geometry_file)
    sweep = _element_sweep(suspension, [0.0], rack_values=[0.0])
    state = suspension.initial_state()
    targets = convert_targets_to_absolute(
        [dimension[0] for dimension in sweep.target_sweeps],
        state,
    )
    computer = ResidualComputer(
        constraints=suspension.constraints(),
        derived_manager=DerivedPointsManager(suspension.derived_spec()),
        state_buffer=state.copy(),
        n_target_variables=len(targets),
    )
    free = state.get_free_array()
    target_row = computer.n_constraints
    analytical = computer.compute_jacobian(free, targets)[target_row]
    numerical = np.zeros_like(analytical)
    for index in range(free.size):
        plus = free.copy()
        minus = free.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        numerical[index] = (
            computer.compute(plus, targets)[target_row]
            - computer.compute(minus, targets)[target_row]
        ) / (2.0 * FD_STEP)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)


def test_mixed_targets_produce_one_analytical_tangent_field_each() -> None:
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = _element_sweep(suspension, [0.0], rack_values=[0.0])
    state = solve_sweep(suspension, sweep)[0][0]
    targets = convert_targets_to_absolute(
        [dimension[0] for dimension in sweep.target_sweeps],
        suspension.initial_state(),
    )

    fields, info = compute_state_tangents(
        state,
        suspension.constraints(),
        DerivedPointsManager(suspension.derived_spec()),
        targets,
    )

    assert len(fields) == 2
    assert [field.target.kind for field in fields] == [
        TargetKind.ELEMENT_LENGTH,
        TargetKind.POINT,
    ]
    assert not info.rank_deficient
    element_field = fields[0]
    length_rate = sum(
        float(partial @ element_field.velocity(point))
        for point, partial in element_field.target.point_partials(state.positions)
    )
    assert length_rate == pytest.approx(1.0)


def test_damper_locked_rack_sweep_allows_wheel_center_motion() -> None:
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = _element_sweep(
        suspension,
        [0.0, 0.0, 0.0],
        rack_values=[-20.0, 0.0, 20.0],
    )
    states, _ = solve_sweep(suspension, sweep)
    damper = sweep.target_sweeps[0][0]
    lengths = [damper.measure(state.positions) for state in states]
    wheel_centres = [state.get(PointID.WHEEL_CENTER).data for state in states]

    assert max(lengths) - min(lengths) < 1e-5
    assert not np.allclose(wheel_centres[0], wheel_centres[-1])

    tangents = compute_sweep_tangents(suspension, sweep, states)
    assert all(len(fields) == 2 for fields in tangents.per_step)
    assert all(not info.rank_deficient for info in tangents.solve_infos)


def test_axle_catalog_and_side_resolution_are_stable() -> None:
    suspension = load_geometry(DATA_DIR / "trailing_arm_axle_geometry.yaml")
    coordinates = suspension.drive_coordinates()
    assert [(item.id, item.scope, item.side) for item in coordinates] == [
        ("damper", Scope.CORNER, Side.LEFT),
        ("damper", Scope.CORNER, Side.RIGHT),
    ]
    pose = initial_pose(suspension)
    assert [(item.id, item.side) for item in pose.drive_coordinates] == [
        ("damper", "left"),
        ("damper", "right"),
    ]
    assert pose.drive_coordinates[0].point_keys == (
        "left_strut_top",
        "left_strut_bottom",
    )

    with pytest.raises(ValueError, match="requires side left or right"):
        build_sweep(
            {
                "targets": [
                    {
                        "kind": "element_length",
                        "element": "damper",
                        "values": [0.0],
                    }
                ]
            },
            suspension,
        )
    with pytest.raises(ValueError, match="Unknown element-length target ID 'pushrod'"):
        build_sweep(
            {
                "targets": [
                    {
                        "kind": "element_length",
                        "element": "pushrod",
                        "side": "left",
                        "values": [0.0],
                    }
                ]
            },
            suspension,
        )


def test_axle_solves_independent_sided_damper_targets_and_reports_series() -> None:
    suspension = load_geometry(DATA_DIR / "trailing_arm_axle_geometry.yaml")
    sweep = build_sweep(
        {
            "targets": [
                {
                    "kind": "element_length",
                    "element": "damper",
                    "side": "left",
                    "values": [0.0, 1.0, -1.0],
                },
                {
                    "kind": "element_length",
                    "element": "damper",
                    "side": "right",
                    "values": [0.0, -1.0, 1.0],
                },
            ]
        },
        suspension,
    )
    states, _ = solve_sweep(suspension, sweep)
    parameters = sweep_parameters(sweep, states)

    assert [(item.coordinate_id, item.side) for item in parameters] == [
        ("damper", "left"),
        ("damper", "right"),
    ]
    for dimension, parameter in zip(sweep.target_sweeps, parameters):
        assert parameter.values == pytest.approx(
            [
                target.measure(state.positions)
                for target, state in zip(dimension, states)
            ]
        )


def test_duplicate_scalar_coordinate_is_rejected() -> None:
    target = PointTarget(
        PointID.WHEEL_CENTER,
        PointTargetAxis(Axis.Z),
        0.0,
    )
    with pytest.raises(ValueError, match="same scalar coordinate"):
        SweepConfig([[target], [target]])
