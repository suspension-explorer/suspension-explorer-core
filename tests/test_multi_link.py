"""Multi-link corner: solving, virtual-only steering, and axle composition."""

from pathlib import Path
from typing import cast

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import PointID
from kinematics.core.metrics.main import MetricRow
from kinematics.core.metrics.registry import metric_specs_for_suspension
from kinematics.core.primitives.point_ref import Side
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import MultiLinkSuspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep

TEST_DATA = Path(__file__).parent / "data"

PHYSICAL_STEERING_KEYS = (
    "caster",
    "kpi",
    "steering_axis_offset_ground",
    "scrub_radius",
    "mechanical_trail",
)


@pytest.fixture
def multi_link() -> MultiLinkSuspension:
    suspension = load_geometry(TEST_DATA / "multi_link_geometry.yaml")
    assert isinstance(suspension, MultiLinkSuspension)
    return suspension


def test_multi_link_declares_expected_point_roles(multi_link):
    assert multi_link.reported_type_key() == "multi_link"
    assert multi_link.steering_axis_points() is None
    assert multi_link.rack_attachment_point() is PointID.TRACKROD_INBOARD
    assert multi_link.wheel_axis_points() == (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    assert multi_link.damper_points() == (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
    outboard_joints = {
        PointID.UPPER_FRONT_LINK_OUTBOARD,
        PointID.UPPER_REAR_LINK_OUTBOARD,
        PointID.LOWER_FRONT_LINK_OUTBOARD,
        PointID.LOWER_REAR_LINK_OUTBOARD,
    }
    assert outboard_joints <= set(multi_link.free_points())


def test_hold_catalogue_offers_only_locked_internals(multi_link):
    catalogue = multi_link.suspension_hold_catalogue()
    assert catalogue is not None
    assert catalogue.default_option_id == "damper_length"
    assert [option.id for option in catalogue.options] == ["damper_length"]


def test_static_state_reproduces_input_geometry(multi_link):
    state = multi_link.initial_state()
    for point, authored in multi_link.hardpoints.items():
        assert (state.get(point) - authored).norm() == pytest.approx(0.0, abs=1e-9)


def test_sweep_solves_with_rigid_links_and_carrier(multi_link):
    sweep = load_sweep(TEST_DATA / "corner_steer_bump_sweep.yaml", multi_link)
    states, infos = solve_sweep(multi_link, sweep)
    assert all(info.converged for info in infos)
    assert all(info.max_residual < 1e-3 for info in infos)

    design = multi_link.initial_state()
    rigid_pairs = [
        (PointID.UPPER_FRONT_LINK_INBOARD, PointID.UPPER_FRONT_LINK_OUTBOARD),
        (PointID.UPPER_REAR_LINK_INBOARD, PointID.UPPER_REAR_LINK_OUTBOARD),
        (PointID.LOWER_FRONT_LINK_INBOARD, PointID.LOWER_FRONT_LINK_OUTBOARD),
        (PointID.LOWER_REAR_LINK_INBOARD, PointID.LOWER_REAR_LINK_OUTBOARD),
        (PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD),
        (PointID.UPPER_FRONT_LINK_OUTBOARD, PointID.LOWER_REAR_LINK_OUTBOARD),
        (PointID.UPPER_REAR_LINK_OUTBOARD, PointID.LOWER_FRONT_LINK_OUTBOARD),
        (PointID.AXLE_OUTBOARD, PointID.UPPER_FRONT_LINK_OUTBOARD),
        (PointID.AXLE_INBOARD, PointID.LOWER_REAR_LINK_OUTBOARD),
        (PointID.TRACKROD_OUTBOARD, PointID.UPPER_REAR_LINK_OUTBOARD),
    ]
    for state in states:
        for point_a, point_b in rigid_pairs:
            design_distance = float((design.get(point_a) - design.get(point_b)).norm())
            solved_distance = float((state.get(point_a) - state.get(point_b)).norm())
            assert solved_distance == pytest.approx(design_distance, abs=1e-3)


def test_metrics_report_virtual_steering_family_only(multi_link):
    sweep = load_sweep(TEST_DATA / "corner_steer_bump_sweep.yaml", multi_link)
    states, _ = solve_sweep(multi_link, sweep)
    evaluated = compute_sweep_metrics(multi_link, sweep, states)
    row = cast(MetricRow, evaluated.rows[len(evaluated.rows) // 2])

    for key in PHYSICAL_STEERING_KEYS:
        assert key not in row
        assert f"{key}_virtual" in row

    # The fixture is synthesized about a designed virtual kingpin; the
    # locked-internals response must recover it at the design state.
    assert row["kpi_virtual"] == pytest.approx(13.0, abs=0.3)
    assert row["caster_virtual"] == pytest.approx(4.0, abs=0.3)

    # Undefined plane-intersection instant centres degrade to None.
    assert row["svic_x"] is None
    assert row["fvic_y"] is None


def test_metric_metadata_matches_emitted_columns(multi_link):
    specs = metric_specs_for_suspension(multi_link)
    for key in PHYSICAL_STEERING_KEYS:
        assert key not in specs
        assert f"{key}_virtual" in specs


def test_axle_builds_mirrors_and_solves():
    suspension = load_geometry(TEST_DATA / "multi_link_axle_geometry.yaml")
    assert isinstance(suspension, AxleSuspension)
    assert suspension.reported_type_key() == "multi_link"

    left = suspension.corners[Side.LEFT]
    right = suspension.corners[Side.RIGHT]
    assert isinstance(left, MultiLinkSuspension)
    assert isinstance(right, MultiLinkSuspension)
    left_point = left.hardpoints[PointID.UPPER_FRONT_LINK_INBOARD]
    right_point = right.hardpoints[PointID.UPPER_FRONT_LINK_INBOARD]
    assert float(right_point[1]) == pytest.approx(-float(left_point[1]))

    catalogue = suspension.suspension_hold_catalogue()
    assert catalogue is not None
    assert catalogue.default_option_id == "damper_length"

    sweep = load_sweep(TEST_DATA / "axle_steer_sweep.yaml", suspension)
    states, infos = solve_sweep(suspension, sweep)
    assert all(info.converged for info in infos)


def test_link_mounted_pickup_is_derived_and_rides_the_link(multi_link):
    """The fixture mounts the coilover fork on the lower-rear link."""
    assert PointID.STRUT_BOTTOM in multi_link.derived_spec().functions
    assert PointID.STRUT_BOTTOM not in multi_link.free_points()

    sweep = load_sweep(TEST_DATA / "corner_steer_bump_sweep.yaml", multi_link)
    states, infos = solve_sweep(multi_link, sweep)
    assert all(info.converged for info in infos)
    design = multi_link.initial_state()
    inboard_offset = float(
        (
            design.get(PointID.STRUT_BOTTOM)
            - design.get(PointID.LOWER_REAR_LINK_INBOARD)
        ).norm()
    )
    for state in states:
        inboard = state.get(PointID.LOWER_REAR_LINK_INBOARD)
        outboard = state.get(PointID.LOWER_REAR_LINK_OUTBOARD)
        pickup = state.get(PointID.STRUT_BOTTOM)
        rod = (outboard - inboard).normalize()
        along = float((pickup - inboard).data @ rod.data)
        off_axis = float(((pickup - inboard) - rod.vector() * along).norm())
        assert off_axis == pytest.approx(0.0, abs=1e-6)
        assert along == pytest.approx(inboard_offset, abs=1e-6)


def test_off_centreline_link_pickup_is_rejected():
    import yaml

    from kinematics.core.input import build_suspension

    with (TEST_DATA / "multi_link_geometry.yaml").open() as handle:
        data = yaml.safe_load(handle)
    data["hardpoints"]["strut_bottom"] = {"x": -4.7, "y": 754.45, "z": 230}
    with pytest.raises(ValueError, match="centreline"):
        build_suspension(data)


def test_pushrod_rocker_rejects_link_mount():
    import yaml

    from kinematics.core.input import build_suspension

    with (TEST_DATA / "multi_link_geometry.yaml").open() as handle:
        data = yaml.safe_load(handle)
    data["actuation"] = {"type": "pushrod_rocker", "mount": "lower_rear_link"}
    with pytest.raises(ValueError, match="upright"):
        build_suspension(data)


def test_upright_mount_remains_available():
    import yaml

    from kinematics.core.input import build_suspension

    with (TEST_DATA / "multi_link_geometry.yaml").open() as handle:
        data = yaml.safe_load(handle)
    data["actuation"] = {"type": "direct", "mount": "upright"}
    data["hardpoints"]["strut_bottom"] = {"x": 5, "y": 795, "z": 220}
    suspension = build_suspension(data)
    assert isinstance(suspension, MultiLinkSuspension)
    assert PointID.STRUT_BOTTOM in suspension.free_points()
    assert PointID.STRUT_BOTTOM not in suspension.derived_spec().functions


def test_unsteered_corner_installs_toe_link():
    data_path = TEST_DATA / "multi_link_geometry.yaml"
    import yaml

    with data_path.open() as handle:
        data = yaml.safe_load(handle)
    data["config"]["steering"] = {"type": "none"}
    hardpoints = data["hardpoints"]
    hardpoints["toe_link_inboard"] = hardpoints.pop("trackrod_inboard")
    hardpoints["toe_link_outboard"] = hardpoints.pop("trackrod_outboard")

    from kinematics.core.input import build_suspension

    suspension = build_suspension(data)
    assert isinstance(suspension, MultiLinkSuspension)
    assert suspension.rack_attachment_point() is None
    assert suspension.steering_actuator_coordinate() is None
    assert suspension.suspension_hold_catalogue() is None
