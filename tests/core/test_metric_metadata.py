"""Tests for canonical metric identities and display metadata."""

from pathlib import Path

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis, PointID, Scope
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
)
from kinematics.core.metrics.metadata import metric_display
from kinematics.core.metrics.registry import (
    MetricKind,
    derivative_spec,
    flat_key,
    metric_specs_for_suspension,
    specs_by_key,
    split_flat_key,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.suspensions.axle import AxleSuspension, HeaveLinkRockerToRocker


def test_static_metric_specs_use_unit_free_identities() -> None:
    specs = specs_by_key()

    assert specs["camber"].unit is MetricUnit.DEG
    assert specs["camber"].scope is Scope.CORNER
    assert specs["track"].unit is MetricUnit.MM
    assert specs["track"].scope is Scope.AXLE
    assert specs["t_bar_heave_angle"].label == "T-Bar Heave Angle"
    assert specs["t_bar_heave_angle"].component == "arb"
    assert "camber_deg" not in specs
    assert "track_mm" not in specs


def test_flat_location_is_separate_from_metric_identity() -> None:
    assert flat_key("camber", "left") == "camber_left"
    assert split_flat_key("camber_left") == ("camber", "left")
    assert split_flat_key("track") == ("track", None)


def test_display_metadata_resolves_corner_location_without_changing_units() -> None:
    specs = specs_by_key()

    display = metric_display("camber_right", specs)

    assert display is not None
    assert display.key == "camber_right"
    assert display.label == "Right Camber"
    assert display.unit is MetricUnit.DEG
    assert display.kind is MetricKind.STATE
    assert display.scope is Scope.CORNER
    assert display.location == "right"


def test_axle_metric_rejects_corner_location() -> None:
    assert metric_display("track_left", specs_by_key()) is None


def test_current_derivative_specs_have_explicit_labels(test_data_dir: Path) -> None:
    """Pin every derivative semantic currently emitted by supported topologies."""
    geometry_names = (
        "geometry.yaml",
        "corner_rocker_geometry.yaml",
        "corner_strut_rocker_geometry.yaml",
        "macpherson_geometry.yaml",
        "axle_geometry_rocker.yaml",
        "axle_geometry_t_bar.yaml",
    )
    labels: dict[str, str] = {}
    for geometry_name in geometry_names:
        suspension = load_geometry(test_data_dir / geometry_name)
        labels.update(
            {
                key: spec.label
                for key, spec in metric_specs_for_suspension(suspension).items()
                if spec.kind is MetricKind.DERIVATIVE
            }
        )

    rocker_axle = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assert isinstance(rocker_axle, AxleSuspension)
    for definition in HeaveLinkRockerToRocker().derivative_metric_definitions(
        rocker_axle
    ):
        spec = derivative_spec(definition)
        labels[spec.key] = spec.label

    assert labels == {
        "deriv_arb_twist_wrt_hub_z_left": "ARB Twist wrt. Left Hub Z",
        "deriv_arb_twist_wrt_hub_z_right": "ARB Twist wrt. Right Hub Z",
        "deriv_camber_wrt_hub_z": "Camber wrt. Hub Z",
        "deriv_camber_wrt_rack_displacement": ("Camber wrt. Rack Displacement"),
        "deriv_caster_wrt_hub_z": "Caster wrt. Hub Z",
        "deriv_damper_length_wrt_hub_z": "Damper Length wrt. Hub Z",
        "deriv_half_track_wrt_hub_z": "Half-Track wrt. Hub Z",
        "deriv_heave_link_length_wrt_hub_z_left": ("Heave Link Length wrt. Left Hub Z"),
        "deriv_heave_link_length_wrt_hub_z_right": (
            "Heave Link Length wrt. Right Hub Z"
        ),
        "deriv_kpi_wrt_hub_z": "KPI wrt. Hub Z",
        "deriv_roadwheel_angle_wrt_hub_z": "Roadwheel Angle wrt. Hub Z",
        "deriv_roadwheel_angle_wrt_rack_displacement": (
            "Roadwheel Angle wrt. Rack Displacement"
        ),
        "deriv_rocker_angle_wrt_hub_z": "Rocker Angle wrt. Hub Z",
        "deriv_t_bar_center_x_wrt_hub_z_left": ("T-Bar Center X wrt. Left Hub Z"),
        "deriv_t_bar_center_x_wrt_hub_z_right": ("T-Bar Center X wrt. Right Hub Z"),
        "deriv_torsion_bar_twist_wrt_hub_z": ("Torsion Bar Twist wrt. Hub Z"),
        "deriv_wheel_center_x_wrt_hub_z": "Wheel Center X wrt. Hub Z",
    }


def test_unlabelled_derivative_falls_back_to_exact_export_key() -> None:
    definition = DerivativeMetricDefinition(
        response=CallableScalarResponse(
            lambda positions: positions[PointID.WHEEL_CENTER][Axis.X],
            name="future_response",
            unit=MetricUnit.MM,
        ),
        driver=PointCoordinateResponse.from_world_axis(
            PointID.WHEEL_CENTER,
            Axis.Z,
            name="future_driver",
            unit=MetricUnit.MM,
        ),
    )

    spec = derivative_spec(definition)

    assert spec.label == "deriv_future_response_wrt_future_driver"
