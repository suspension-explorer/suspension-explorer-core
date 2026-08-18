"""Element-length sweep targets across schema, solver, and presentation."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.analysis import initial_pose, sweep_parameters
from kinematics.core.coordinates import (
    CoordinateAxis,
    CoordinateType,
    ElementLengthCoordinate,
    PointCoordinate,
)
from kinematics.core.enums import Axis, PointID, Scope, TargetValueMode, Units
from kinematics.core.input import build_sweep, parse_sweep_spec
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.sensitivity import compute_state_tangents
from kinematics.core.solver import ResidualComputer, convert_targets_to_absolute
from kinematics.core.state import SuspensionState
from kinematics.core.sweep import compute_sweep_tangents, solve_sweep
from kinematics.core.targeting import SweepConfig

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
            "type": "element_length",
            "element": "damper",
            "side": "left",
            "mode": "relative",
            "values": values,
        }
    ]
    if rack_values is not None:
        targets.append(
            {
                "type": "actuator_position",
                "actuator": "rack",
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
    setup_length = coordinate.measure(state.positions)

    relative = coordinate.target(12.5, TargetValueMode.RELATIVE)
    absolute = coordinate.target(321.0, TargetValueMode.ABSOLUTE)
    resolved = convert_targets_to_absolute([relative, absolute], state)

    assert resolved[0].mode is TargetValueMode.ABSOLUTE
    assert resolved[0].value == pytest.approx(setup_length + 12.5)
    assert resolved[1] is absolute
    assert resolved[1].value == pytest.approx(321.0)

    with pytest.raises(ValueError, match="at least"):
        coordinate.target(-1.0, TargetValueMode.ABSOLUTE)
    with pytest.raises(ValueError, match="at least"):
        coordinate.target(0.0, TargetValueMode.ABSOLUTE)
    with pytest.raises(ValueError, match="at least"):
        coordinate.target(EPS_GEOMETRIC / 2.0, TargetValueMode.ABSOLUTE)
    with pytest.raises(ValueError, match="finite"):
        coordinate.target(float("nan"), TargetValueMode.RELATIVE)
    with pytest.raises(ValueError, match=r"Sweep target 0.*at least"):
        convert_targets_to_absolute(
            [coordinate.target(-(setup_length + 1.0))],
            state,
        )


def test_coincident_intermediate_element_endpoints_have_finite_zero_partials() -> None:
    target = ElementLengthCoordinate(
        id="damper",
        label="Damper",
        unit=Units.MILLIMETERS.symbol,
        point_a=PointID.STRUT_TOP,
        point_b=PointID.STRUT_BOTTOM,
        scope=Scope.CORNER,
    ).target(1.0, TargetValueMode.ABSOLUTE)
    coincident = Point3([1.0, 2.0, 3.0])

    partials = target.coordinate.point_partials(
        {
            PointID.STRUT_TOP: coincident,
            PointID.STRUT_BOTTOM: coincident,
        }
    )

    assert [point_id for point_id, _partial in partials] == [
        PointID.STRUT_TOP,
        PointID.STRUT_BOTTOM,
    ]
    for _point_id, partial in partials:
        np.testing.assert_array_equal(partial, np.zeros(3))


def test_structurally_driven_zero_target_jacobian_is_not_misclassified() -> None:
    coincident = Point3([1.0, 2.0, 3.0])
    state = SuspensionState(
        positions={
            PointID.STRUT_TOP: coincident,
            PointID.STRUT_BOTTOM: coincident,
        },
        free_points={PointID.STRUT_TOP, PointID.STRUT_BOTTOM},
    )
    target = ElementLengthCoordinate(
        id="damper",
        label="Damper",
        unit=Units.MILLIMETERS.symbol,
        point_a=PointID.STRUT_TOP,
        point_b=PointID.STRUT_BOTTOM,
        scope=Scope.CORNER,
    ).target(1.0, TargetValueMode.ABSOLUTE)
    computer = ResidualComputer(
        constraints=[],
        derived_manager=DerivedPointsManager(DerivedPointsSpec({}, {})),
        state_buffer=state,
        n_target_variables=1,
    )

    jacobian = computer.compute_jacobian(state.get_free_array(), [target])

    np.testing.assert_array_equal(jacobian, np.zeros((1, 6)))


def test_target_without_free_or_derived_dependency_fails_actionably() -> None:
    state = SuspensionState(
        positions={
            PointID.STRUT_TOP: Point3([1.0, 2.0, 3.0]),
            PointID.STRUT_BOTTOM: Point3([2.0, 2.0, 3.0]),
        },
        free_points=set(),
    )
    target = ElementLengthCoordinate(
        id="damper",
        label="Damper",
        unit=Units.MILLIMETERS.symbol,
        point_a=PointID.STRUT_TOP,
        point_b=PointID.STRUT_BOTTOM,
        scope=Scope.CORNER,
    ).target(1.0, TargetValueMode.ABSOLUTE)
    computer = ResidualComputer(
        constraints=[],
        derived_manager=DerivedPointsManager(DerivedPointsSpec({}, {})),
        state_buffer=state,
        n_target_variables=1,
    )

    with pytest.raises(ValueError, match="no free or derived point dependency"):
        computer.compute_jacobian(np.empty(0), [target])


def test_missing_target_endpoint_is_reported_with_target_index() -> None:
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    target = ElementLengthCoordinate(
        id="damper",
        label="Damper",
        unit=Units.MILLIMETERS.symbol,
        point_a=PointID.STRUT_TOP,
        point_b=PointID.DAMPER_ROCKER,
        scope=Scope.CORNER,
    ).target(0.0)

    with pytest.raises(
        ValueError,
        match=r"Sweep target 0: required point 'DAMPER_ROCKER'.*initial state",
    ):
        convert_targets_to_absolute([target], suspension.initial_state())


def test_canonical_mixed_yaml_specs_remain_typed_and_round_trip() -> None:
    raw = {
        "version": 1,
        "targets": [
            {
                "type": "actuator_position",
                "actuator": "rack",
                "direction": {"axis": "y"},
                "values": [0.0, 1.0],
            },
            {
                "type": "element_length",
                "element": "damper",
                "side": "left",
                "values": [0.0, 0.0],
            },
        ],
    }
    spec = parse_sweep_spec(raw)
    dumped = spec.model_dump(mode="json")
    reparsed = parse_sweep_spec(dumped)

    assert dumped["targets"][0]["type"] == "actuator_position"
    assert dumped["targets"][1]["type"] == "element_length"
    assert dumped["targets"][1]["side"] == "left"
    assert reparsed == spec

    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep = build_sweep(raw, suspension)
    assert sweep.target_sweeps[0][0].coordinate.type is CoordinateType.ACTUATOR_POSITION
    assert sweep.target_sweeps[1][0].coordinate.type is CoordinateType.ELEMENT_LENGTH

    invalid = {
        "targets": [
            {
                "type": "element_length",
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


def test_sweep_target_type_is_required() -> None:
    with pytest.raises(ValueError, match="Unable to extract tag.*type"):
        parse_sweep_spec(
            {
                "targets": [
                    {
                        "kind": "point",
                        "point": "wheel_center",
                        "direction": {"axis": "z"},
                        "values": [0.0],
                    }
                ]
            }
        )


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
    rack_values = (
        [0.0, 0.0, 0.0] if suspension.required_actuator_coordinates() else None
    )
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
        assert target.coordinate.measure(state.positions) == pytest.approx(
            expected, abs=1e-5
        )


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
    assert [field.target.coordinate.type for field in fields] == [
        CoordinateType.ELEMENT_LENGTH,
        CoordinateType.ACTUATOR_POSITION,
    ]
    assert not info.rank_deficient
    element_field = fields[0]
    length_rate = sum(
        float(partial @ element_field.rate(point))
        for point, partial in element_field.target.coordinate.point_partials(
            state.positions
        )
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
    lengths = [damper.coordinate.measure(state.positions) for state in states]
    wheel_centres = [state.get(PointID.WHEEL_CENTER).data for state in states]

    assert max(lengths) - min(lengths) < 1e-5
    assert not np.allclose(wheel_centres[0], wheel_centres[-1])

    tangents = compute_sweep_tangents(suspension, sweep, states)
    assert all(len(fields) == 2 for fields in tangents.per_step)
    assert all(not info.rank_deficient for info in tangents.solve_infos)


def test_axle_dampers_locked_during_shared_rack_steer() -> None:
    suspension = load_geometry(DATA_DIR / "macpherson_axle_geometry.yaml")
    sweep = build_sweep(
        {
            "targets": [
                {
                    "type": "element_length",
                    "element": "damper",
                    "side": side,
                    "values": [0.0, 0.0, 0.0],
                }
                for side in ("left", "right")
            ]
            + [
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "values": [-5.0, 0.0, 5.0],
                }
            ]
        },
        suspension,
    )

    states, infos = solve_sweep(suspension, sweep)

    assert all(info.converged for info in infos)
    for dimension in sweep.target_sweeps[:2]:
        coordinate = dimension[0]
        lengths = [coordinate.coordinate.measure(state.positions) for state in states]
        assert max(lengths) - min(lengths) < 1e-5
    for side in (Side.LEFT, Side.RIGHT):
        point = PointRef(side, PointID.WHEEL_CENTER)
        assert not np.allclose(states[0].get(point).data, states[-1].get(point).data)


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
                        "type": "element_length",
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
                        "type": "element_length",
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
                    "type": "element_length",
                    "element": "damper",
                    "side": "left",
                    "values": [0.0, 1.0, -1.0],
                },
                {
                    "type": "element_length",
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
                target.coordinate.measure(state.positions)
                for target, state in zip(dimension, states)
            ]
        )


def test_duplicate_scalar_coordinate_is_rejected() -> None:
    target = PointCoordinate(
        PointID.WHEEL_CENTER, CoordinateAxis(Axis.Z), Scope.CORNER
    ).target(0.0)
    with pytest.raises(
        ValueError,
        match=r"controls 0, 1.*same point coordinate 'wheel_center_z'.*step 0",
    ):
        SweepConfig([[target], [target]])
