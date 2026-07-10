"""
Tests for the high-level analysis layer and metric display metadata.

These guard the package's front-end contract: analyze_sweep must return a
complete, self-consistent result (frames, metrics, display topology), and
every metric column the package emits must resolve to display metadata.
"""

from pathlib import Path

import pytest

from kinematics.analysis import analyze_sweep, initial_pose
from kinematics.io import load_geometry, load_sweep
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.metrics.main import AxleMetricRows
from kinematics.metrics.metadata import metric_display, metric_display_for_keys
from kinematics.visualization.display import AXIS_FOOT_SUFFIX

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corner_analysis():
    suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
    sweep_config = load_sweep(DATA_DIR / "sweep.yaml", suspension)
    return analyze_sweep(suspension, sweep_config)


@pytest.fixture(scope="module")
def axle_analysis():
    suspension = load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")
    sweep_config = load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", suspension)
    return analyze_sweep(suspension, sweep_config)


class TestMetricDisplayMetadata:
    """
    Every emitted column must resolve to display metadata.
    """

    def test_every_emitted_corner_column_resolves(self):
        suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
        sweep_config = load_sweep(DATA_DIR / "sweep.yaml", suspension)
        states, _ = solve_sweep(suspension, sweep_config)
        rows = compute_sweep_metrics(suspension, sweep_config, states)

        row = rows[0]
        # A corner model's row is already the flat rendering (location-less
        # keys); an axle model flattens to location-suffixed columns.
        flat = row.flat_row() if isinstance(row, AxleMetricRows) else row
        missing = [key for key in flat if metric_display(key) is None]
        assert missing == [], f"Columns without display metadata: {missing}"

    def test_every_emitted_axle_column_resolves(self):
        suspension = load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")
        sweep_config = load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", suspension)
        states, _ = solve_sweep(suspension, sweep_config)
        rows = compute_sweep_metrics(suspension, sweep_config, states)

        row = rows[0]
        assert isinstance(row, AxleMetricRows)
        missing = [key for key in row.flat_row() if metric_display(key) is None]
        assert missing == [], f"Columns without display metadata: {missing}"

    def test_side_suffix_resolution(self):
        display = metric_display("camber_gain_deg_per_mm_left")
        assert display is not None
        assert display.label == "Left Camber Gain"
        assert display.unit == "deg/mm"

    def test_unknown_column_returns_none(self):
        assert metric_display("not_a_real_metric") is None
        assert metric_display_for_keys(["not_a_real_metric"]) == []


class TestCornerAnalysis:
    """
    analyze_sweep on the strut-equipped corner fixture.
    """

    def test_frames_and_metrics(self, corner_analysis):
        analysis = corner_analysis
        assert analysis.steps == len(analysis.frames) > 0
        assert analysis.suspension.type_key == "double_wishbone"

        first = analysis.frames[0]
        assert "WHEEL_CENTER" in first.positions
        assert "STRUT_BOTTOM" in first.positions
        assert first.solver.converged

        # Derivative metrics are present without the caller ever touching
        # tangents: the analysis layer orchestrates them internally.
        assert "damper_motion_ratio" in analysis.metric_keys
        assert "camber_gain_deg_per_mm" in analysis.metric_keys

    def test_metric_display_covers_all_keys(self, corner_analysis):
        analysis = corner_analysis
        covered = {display.key for display in analysis.metric_display}
        assert covered == set(analysis.metric_keys)

    def test_setup_reference_matches_frame_columns(self, corner_analysis):
        analysis = corner_analysis
        setup = analysis.references["setup"]
        assert list(setup.metrics.keys()) == analysis.metric_keys
        assert "WHEEL_CENTER" in setup.positions

    def test_wheel_and_links(self, corner_analysis):
        analysis = corner_analysis
        assert analysis.wheel is not None
        assert analysis.wheel.radius > analysis.wheel.rim_radius > 0
        labels = {link.label for link in analysis.links}
        assert "Spring/Damper" in labels
        assert len(analysis.wheel_anchors) == 1

    def test_sweep_parameters(self, corner_analysis):
        parameters = {
            (p.point, p.axis, p.side) for p in corner_analysis.sweep_parameters
        }
        assert ("WHEEL_CENTER", "Z", None) in parameters
        assert ("TRACKROD_INBOARD", "Y", None) in parameters


class TestAxleAnalysis:
    """
    analyze_sweep on the rocker/ARB axle fixture.
    """

    def test_rocker_fan_replaced_by_axis_and_arms(self, axle_analysis):
        labels = {link.label for link in axle_analysis.links}
        assert not any(label.endswith("Rocker") for label in labels)
        for side in ("Left", "Right"):
            assert f"{side} Rocker Axis" in labels
            assert f"{side} Rocker Pushrod Arm" in labels
            assert f"{side} Rocker Droplink Arm" in labels

    def test_axis_feet_are_perpendicular_projections(self, axle_analysis):
        positions = axle_analysis.frames[0].positions
        for pickup in ("LEFT_PUSHROD_INBOARD", "LEFT_DROPLINK_ROCKER"):
            point = positions[pickup]
            foot = positions[f"{pickup}{AXIS_FOOT_SUFFIX}"]
            axis_a = positions["LEFT_ROCKER_AXIS_FRONT"]
            axis_b = positions["LEFT_ROCKER_AXIS_REAR"]
            axis = [axis_b[i] - axis_a[i] for i in range(3)]
            arm = [point[i] - foot[i] for i in range(3)]

            # The arm is perpendicular to the axis: dot(axis, arm) = 0.
            dot = sum(axis[i] * arm[i] for i in range(3))
            assert abs(dot) < 1e-6, pickup

            # The foot lies on the axis line: (foot - a) parallel to axis.
            offset = [foot[i] - axis_a[i] for i in range(3)]
            cross_sq = (
                (offset[1] * axis[2] - offset[2] * axis[1]) ** 2
                + (offset[2] * axis[0] - offset[0] * axis[2]) ** 2
                + (offset[0] * axis[1] - offset[1] * axis[0]) ** 2
            )
            assert cross_sq < 1e-9, pickup

    def test_axle_metrics_and_display(self, axle_analysis):
        analysis = axle_analysis
        # Axle-level base keys in metric_keys; per-corner base keys and the
        # locations present are carried separately (structural, not mangled).
        for key in ("heave_mm", "roll_deg", "arb_twist_vs_roll_deg_per_deg"):
            assert key in analysis.metric_keys, key
        for key in ("camber_deg", "rocker_motion_ratio_deg_per_mm"):
            assert key in analysis.corner_metric_keys, key
        assert analysis.locations == ["left", "right"]
        covered = {display.key for display in analysis.metric_display}
        assert covered == set(analysis.metric_keys) | set(
            analysis.corner_metric_keys
        )

    def test_frames_carry_structured_corner_metrics(self, axle_analysis):
        frame = axle_analysis.frames[0]
        assert set(frame.corner_metrics) == {"left", "right"}
        assert "camber_deg" in frame.corner_metrics["left"]
        assert "arb_twist_deg" in frame.metrics
        setup = axle_analysis.references["setup"]
        assert set(setup.corner_metrics) == {"left", "right"}

    def test_center_arb_axis_points_exported(self, axle_analysis):
        # The display point set extends output_points with link-referenced
        # points such as the shared ARB axis, so polylines can resolve.
        assert "CENTER_ARB_AXIS_A" in analysis_point_keys(axle_analysis)

    def test_sweep_parameters_carry_sides(self, axle_analysis):
        parameters = {(p.point, p.axis, p.side) for p in axle_analysis.sweep_parameters}
        assert ("LEFT_WHEEL_CENTER", "Z", "left") in parameters
        assert ("RIGHT_WHEEL_CENTER", "Z", "right") in parameters


def analysis_point_keys(analysis):
    return set(analysis.point_keys)


class TestInitialPose:
    """
    Static preview pose assembly.
    """

    def test_static_pose_has_positions_and_topology(self):
        suspension = load_geometry(DATA_DIR / "corner_strut_geometry.yaml")
        pose = initial_pose(suspension)
        assert "WHEEL_CENTER" in pose.positions
        assert pose.wheel is not None
        assert any(link.label == "Spring/Damper" for link in pose.links)


class TestSweepSpecSteps:
    """
    SweepSpec.n_steps expansion semantics.
    """

    def test_n_steps_from_shared_count(self):
        from kinematics import SweepSpec

        spec = SweepSpec.model_validate(
            {
                "version": 1,
                "steps": 7,
                "targets": [
                    {
                        "point": "WHEEL_CENTER",
                        "direction": {"axis": "Z"},
                        "start": -10,
                        "stop": 10,
                    }
                ],
            }
        )
        assert spec.n_steps == 7

    def test_n_steps_explicit_values_win(self):
        from kinematics import SweepSpec

        spec = SweepSpec.model_validate(
            {
                "version": 1,
                "steps": 3,
                "targets": [
                    {
                        "point": "WHEEL_CENTER",
                        "direction": {"axis": "Z"},
                        "values": [0, 1, 2, 3, 4],
                    }
                ],
            }
        )
        assert spec.n_steps == 5
