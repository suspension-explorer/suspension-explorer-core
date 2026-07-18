"""Label pinning for derivative metrics across supported topologies.

This test loads YAML fixtures through the CLI loader, so it lives outside
tests/core: the core-boundary CI environment installs the core package
without the CLI's YAML dependency.
"""

from pathlib import Path

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.metrics.registry import (
    MetricKind,
    derivative_spec,
    metric_specs_for_suspension,
)
from kinematics.core.suspensions.axle import AxleSuspension, HeaveLinkRockerToRocker


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


def test_state_metric_specs_match_selected_topology_values(
    test_data_dir: Path,
) -> None:
    geometry_names = (
        "geometry.yaml",
        "macpherson_geometry.yaml",
        "corner_rocker_geometry.yaml",
        "axle_geometry.yaml",
        "axle_geometry_rocker.yaml",
        "axle_geometry_t_bar.yaml",
    )
    for geometry_name in geometry_names:
        suspension = load_geometry(test_data_dir / geometry_name)
        row = suspension.compute_state_metrics(suspension.initial_state())
        if isinstance(row, AxleMetricRows):
            emitted = set(row.axle)
            for corner_row in row.corners.values():
                emitted.update(corner_row)
        else:
            emitted = set(row)

        declared = {
            key
            for key, spec in metric_specs_for_suspension(suspension).items()
            if spec.kind is MetricKind.STATE
        }
        assert declared == emitted, geometry_name
