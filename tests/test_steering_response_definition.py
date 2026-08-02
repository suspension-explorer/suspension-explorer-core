"""Topology steering-response definitions and current-state probe targets."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import TargetValueMode
from kinematics.core.isolation_coordinates import HingeAngleCoordinate
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.schema.sweep import SweepSpec, build_sweep_config
from kinematics.core.steering_response import (
    SteeringProbeOptionAvailability,
    SteeringProbeOptionClass,
    SteeringResponseDefinition,
    materialize_steering_response_probe,
    materialize_steering_response_targets,
)
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import (
    DoubleWishboneSuspension,
    MacPhersonSuspension,
)
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent / "data"


def _position_snapshot(state):
    return {point: position.data.copy() for point, position in state.positions.items()}


def test_double_wishbone_catalogue_is_layout_owned_and_damper_is_diagnostic() -> None:
    without_damper = load_geometry(DATA_DIR / "geometry.yaml")
    with_damper = load_geometry(DATA_DIR / "corner_rocker_damper_geometry.yaml")
    assert isinstance(without_damper, DoubleWishboneSuspension)
    assert isinstance(with_damper, DoubleWishboneSuspension)

    catalogue = without_damper.steering_probe_catalogue()
    assert catalogue is not None
    assert catalogue.default_option_id == "lower_wishbone_hinge_angle"
    assert [option.id for option in catalogue.options] == [
        "lower_wishbone_hinge_angle",
        "upper_wishbone_hinge_angle",
    ]
    assert catalogue.options[0].option_class is SteeringProbeOptionClass.CANONICAL

    damper_catalogue = with_damper.steering_probe_catalogue()
    assert damper_catalogue is not None
    damper = damper_catalogue.option("damper_length")
    assert damper.option_class is SteeringProbeOptionClass.DIAGNOSTIC
    assert damper.availability is SteeringProbeOptionAvailability.AVAILABLE_WITH_WARNING
    assert damper.warning is not None

    definition = with_damper.resolve_steering_probe()
    assert definition is not None
    assert definition.provenance == "double_wishbone:lower_wishbone_hinge_angle"
    assert definition.steering_coordinate_id == "rack"
    assert definition.held_coordinate_ids == ("lower_wishbone_hinge_angle",)


def test_hinge_angle_coordinate_has_exact_analytical_carried_point_gradient() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    catalogue = suspension.steering_probe_catalogue()
    assert catalogue is not None
    coordinate = catalogue.option("lower_wishbone_hinge_angle").held_coordinates[0]
    assert isinstance(coordinate, HingeAngleCoordinate)

    positions = suspension.initial_state().positions
    point = coordinate.carried_point
    direction = np.array([0.31, -0.47, 0.82], dtype=np.float64)
    analytical = float(coordinate.point_partials(positions)[0][1] @ direction)
    epsilon = 1e-5
    plus = dict(positions)
    minus = dict(positions)
    plus[point] = Point3(positions[point].data + epsilon * direction)
    minus[point] = Point3(positions[point].data - epsilon * direction)
    numerical = (coordinate.measure(plus) - coordinate.measure(minus)) / (2.0 * epsilon)

    assert analytical == pytest.approx(numerical, rel=1e-8, abs=1e-10)


def test_macpherson_probe_uses_bumped_current_strut_length_without_mutation() -> None:
    suspension = load_geometry(DATA_DIR / "macpherson_geometry.yaml")
    assert isinstance(suspension, MacPhersonSuspension)
    definition = suspension.resolve_steering_probe()
    assert definition is not None

    sweep = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    states, infos = solve_sweep(suspension, sweep)
    assert all(info.converged for info in infos)
    state = states[-1]
    before = _position_snapshot(state)
    targets = materialize_steering_response_targets(definition, state)

    assert tuple(target.mode for target in targets) == (
        TargetValueMode.ABSOLUTE,
        TargetValueMode.ABSOLUTE,
    )
    assert definition.steering_actuator.matches(targets[0])
    assert targets[0].value == pytest.approx(targets[0].measure(state.positions))
    assert targets[1].value == pytest.approx(targets[1].measure(state.positions))

    initial_hold = definition.held_coordinates[0].current_value_target(
        suspension.initial_state().positions
    )
    assert abs(targets[1].value - initial_hold.value) > 1.0
    for point, position in state.positions.items():
        np.testing.assert_array_equal(position.data, before[point])


def test_axle_catalogue_composes_one_semantic_option_into_two_corner_holds() -> None:
    suspension = load_geometry(DATA_DIR / "macpherson_axle_geometry.yaml")
    assert isinstance(suspension, AxleSuspension)
    catalogue = suspension.steering_probe_catalogue()
    assert catalogue is not None
    assert catalogue.default_option_id == "strut_length"
    assert [option.id for option in catalogue.options] == [
        "strut_length",
        "lower_arm_hinge_angle",
    ]

    definition = suspension.resolve_steering_probe()
    assert definition is not None

    assert definition.provenance == "macpherson:strut_length"
    assert definition.held_coordinate_ids == ("damper", "damper")
    assert definition.held_coordinate_sides == (Side.LEFT, Side.RIGHT)
    held_points = tuple(
        point
        for coordinate in definition.held_coordinates
        for point in coordinate.point_keys
    )
    assert all(isinstance(point, PointRef) for point in held_points)
    assert tuple(
        point.side for point in held_points if isinstance(point, PointRef)
    ) == (Side.LEFT, Side.LEFT, Side.RIGHT, Side.RIGHT)

    state = suspension.initial_state()
    before = _position_snapshot(state)
    probe = materialize_steering_response_probe(definition, state)
    assert probe is not None
    assert probe.targets[0] is probe.steering_target
    assert probe.held_targets == probe.targets[1:]
    assert all(target.mode is TargetValueMode.ABSOLUTE for target in probe.targets)
    assert tuple(target.coordinate_identity for target in probe.targets) == tuple(
        target.coordinate_identity
        for target in materialize_steering_response_targets(definition, state)
    )
    assert len({target.coordinate_identity for target in probe.targets}) == 3
    for point, position in state.positions.items():
        np.testing.assert_array_equal(position.data, before[point])


def test_definition_rejects_duplicate_holds_and_absent_definition_is_unavailable() -> (
    None
):
    suspension = load_geometry(DATA_DIR / "macpherson_geometry.yaml")
    definition = suspension.resolve_steering_probe()
    assert definition is not None
    with pytest.raises(ValueError, match="duplicate held coordinates"):
        SteeringResponseDefinition(
            steering_actuator=definition.steering_actuator,
            held_coordinates=(
                definition.held_coordinates[0],
                definition.held_coordinates[0],
            ),
            owner=definition.owner,
            definition_id=definition.definition_id,
        )

    no_hold = load_geometry(DATA_DIR / "trailing_arm_coilover_geometry.yaml")
    assert (
        materialize_steering_response_probe(
            no_hold.resolve_steering_probe(), no_hold.initial_state()
        )
        is None
    )


def test_explicit_probe_override_retains_selection_provenance() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    definition = suspension.resolve_steering_probe("upper_wishbone_hinge_angle")
    assert definition is not None
    assert definition.requested_option_id == "upper_wishbone_hinge_angle"
    assert definition.definition_id == "upper_wishbone_hinge_angle"
    assert definition.selection_source == "user_override"
    assert definition.option_class is SteeringProbeOptionClass.EQUIVALENT

    with pytest.raises(ValueError, match="Unknown steering-probe isolation"):
        suspension.resolve_steering_probe("invented_hold")


def test_sweep_schema_threads_probe_selection_without_adding_a_target() -> None:
    suspension = load_geometry(DATA_DIR / "geometry.yaml")
    spec = SweepSpec.model_validate(
        {
            "version": 1,
            "steps": 1,
            "targets": [
                {
                    "type": "actuator_position",
                    "actuator": "rack",
                    "direction": {"axis": "y"},
                    "values": [0],
                }
            ],
            "analysis": {
                "steering_probe": {
                    "isolation": "upper_wishbone_hinge_angle",
                }
            },
        }
    )

    config = build_sweep_config(spec, suspension)

    assert config.steering_probe_isolation == "upper_wishbone_hinge_angle"
    assert len(config.target_sweeps) == 1
