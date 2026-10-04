"""Physical and numerical regression checks for the public race-car examples."""

import json
from functools import cache
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.diagnostics import diagnose_sweep
from kinematics.core.elements import ElementType
from kinematics.core.enums import PointID
from kinematics.core.input import build_sweep
from kinematics.core.metrics.main import AxleMetricRows
from kinematics.core.presentation import (
    NamedElementPath,
    element_paths,
    named_element_paths,
    resolve_positions,
)
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.solver import SolverInfo
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.sweep import compute_sweep_metrics, solve_sweep
from kinematics.core.targeting import SweepConfig

EXAMPLES = Path(__file__).parents[1] / "examples" / "race-cars"
CATALOG = json.loads((EXAMPLES / "catalog.json").read_text())
MODES = ("bump", "roll", "web_bump", "steer")

MIN_CENTERLINE_CLEARANCE_MM = 25.0
# A rod converges on the ball joint beside its own outboard mount. Separation
# there depends on the bracket, so that end of the rod is exempt.
ROD_MOUNT_EXEMPT_LENGTH_MM = 50.0

MECHANISM_TYPES = {
    ElementType.ROCKER,
    ElementType.PUSHROD,
    ElementType.SPRING_DAMPER,
    ElementType.DAMPER,
    ElementType.TORSION_BAR,
    ElementType.HEAVE_LINK,
    ElementType.DROPLINK,
    ElementType.ANTI_ROLL_BAR,
}
LOCATING_TYPES = {
    ElementType.WISHBONE,
    ElementType.UPRIGHT,
    ElementType.TRACK_ROD,
    ElementType.TOE_LINK,
    ElementType.RACK,
}


def _mode_profile(mode: str) -> dict[str, Any]:
    """Return the sweep profile exercised for one motion mode."""
    profile = yaml.safe_load(
        (EXAMPLES / f"axle-{'roll' if mode == 'roll' else 'bump'}.yaml").read_text()
    )
    if mode == "web_bump":
        for target in profile["targets"][:2]:
            target.update(start=-40, stop=80)
    elif mode == "steer":
        profile["targets"] = [
            {
                key: value
                for key, value in target.items()
                if key not in {"start", "stop", "mode"}
            }
            | {"hold": True}
            for target in profile["targets"][:2]
        ] + [
            {
                "type": "actuator_position",
                "actuator": "rack",
                "direction": {"axis": "y"},
                "mode": "relative",
                "start": -15,
                "stop": 15,
            }
        ]
    return profile


@cache
def _solve_mode(
    geometry: str, mode: str
) -> tuple[Suspension, SweepConfig, list[SuspensionState], list[SolverInfo]]:
    """Solve one example through one motion mode; tests share the solution."""
    suspension = load_geometry(EXAMPLES / geometry)
    sweep = build_sweep(_mode_profile(mode), suspension)
    states, stats = solve_sweep(suspension, sweep)
    return suspension, sweep, states, stats


def test_formula_one_variants_share_wheel_steering_and_vehicle_baseline() -> None:
    pushrod = yaml.safe_load((EXAMPLES / "formula_one_pushrod.yaml").read_text())
    pullrod = yaml.safe_load((EXAMPLES / "formula_one_pullrod.yaml").read_text())
    assert pullrod["version"] == pushrod["version"]
    assert pullrod["vehicle_config"] == pushrod["vehicle_config"]
    for name in ("axle_position", "steering", "wheel", "left_setup"):
        assert pullrod["axle_config"][name] == pushrod["axle_config"][name], name
    pushrod_points = pushrod["hardpoints"]["left"]
    pullrod_points = pullrod["hardpoints"]["left"]
    for name in pushrod_points:
        if any(part in name for part in ("wishbone", "trackrod", "axle_")):
            assert pullrod_points[name] == pushrod_points[name], name
    assert pullrod["axle_config"]["actuation"]["mount"] == "upper_wishbone"


def test_formula_one_variants_have_matching_wheel_kinematics() -> None:
    profile = yaml.safe_load((EXAMPLES / "axle-bump.yaml").read_text())
    metrics_by_variant = []
    for variant in ("pushrod", "pullrod"):
        suspension = load_geometry(EXAMPLES / f"formula_one_{variant}.yaml")
        sweep = build_sweep(profile, suspension)
        states, _ = solve_sweep(suspension, sweep)
        metrics_by_variant.append(compute_sweep_metrics(suspension, sweep, states))

    pushrod_metrics, pullrod_metrics = metrics_by_variant
    for pushrod_row, pullrod_row in zip(
        pushrod_metrics.rows, pullrod_metrics.rows, strict=True
    ):
        assert isinstance(pushrod_row, AxleMetricRows)
        assert isinstance(pullrod_row, AxleMetricRows)
        for side in (Side.LEFT, Side.RIGHT):
            for name in ("camber", "caster", "kpi", "toe_angle", "steer_angle"):
                assert pullrod_row.corners[side][name] == pytest.approx(
                    pushrod_row.corners[side][name], abs=1e-7
                )
    for metrics in metrics_by_variant:
        design = metrics.rows[len(metrics.rows) // 2]
        assert isinstance(design, AxleMetricRows)
        assert design.corners[Side.LEFT]["caster"] == pytest.approx(10, abs=0.1)


def _segment_distance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """Minimum distance between finite line segments, including their endpoints."""
    u, v, w = b - a, d - c, a - c
    uu, uv, vv = float(u @ u), float(u @ v), float(v @ v)
    uw, vw = float(u @ w), float(v @ w)

    def to_segment(
        point: np.ndarray, start: np.ndarray, direction: np.ndarray, length_sq: float
    ) -> float:
        parameter = np.clip((point - start) @ direction / length_sq, 0, 1)
        return float(np.linalg.norm(point - start - parameter * direction))

    distances = [
        to_segment(a, c, v, vv),
        to_segment(b, c, v, vv),
        to_segment(c, a, u, uu),
        to_segment(d, a, u, uu),
    ]
    determinant = uu * vv - uv * uv
    if determinant > 1e-9 * uu * vv:
        s = (uv * vw - vv * uw) / determinant
        t = (uu * vw - uv * uw) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            distances.append(float(np.linalg.norm(w + s * u - t * v)))
    return min(distances)


def _is_rocker_shaft(path: NamedElementPath) -> bool:
    """The shaft joins every rocker arm, as a bare axis or a coaxial torsion bar."""
    return path.type is ElementType.TORSION_BAR or path.label.endswith(" Axis")


def _clearance_segments(
    path: NamedElementPath, positions: dict[str, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a path's centerline segments, less a rod's exempt mount end."""
    points = [positions[name] for name in path.points]
    if path.type is ElementType.PUSHROD:
        # The rod element runs from its outboard pickup to the rocker.
        outboard, inboard = points
        direction = (inboard - outboard) / np.linalg.norm(inboard - outboard)
        points = [outboard + ROD_MOUNT_EXEMPT_LENGTH_MM * direction, inboard]
    return list(pairwise(points))


@pytest.mark.parametrize("example", CATALOG, ids=lambda entry: entry["key"])
@pytest.mark.parametrize("mode", MODES)
def test_mechanism_centerlines_keep_clearance(example: dict, mode: str) -> None:
    suspension, _, states, _ = _solve_mode(example["geometry"], mode)
    assembly = suspension.assembly()
    paths = named_element_paths(assembly)
    mechanism = [path for path in paths if path.type in MECHANISM_TYPES]
    locating = [path for path in paths if path.type in LOCATING_TYPES]
    # Every layout keeps its rod and inboard mechanism clear of the wishbones,
    # uprights, and steering linkage.
    candidates = [(first, second) for first in mechanism for second in locating]
    if "pullrod" in example["features"]:
        # The pullrod layouts also separate their inboard elements from one
        # another. The rocker shaft joins every arm, so it is left out.
        separated = [path for path in mechanism if not _is_rocker_shaft(path)]
        candidates.extend(combinations(separated, 2))
    # Shared joints are intentional connections.
    pairs = [
        (first, second)
        for first, second in candidates
        if not set(first.points).intersection(second.points)
    ]
    for state in [suspension.initial_state(), *states]:
        positions = {
            name: np.asarray(value)
            for name, value in resolve_positions(state.positions, assembly).items()
        }
        segments = {
            path: _clearance_segments(path, positions)
            for path in (*mechanism, *locating)
        }
        for first, second in pairs:
            distance = min(
                _segment_distance(a, b, c, d)
                for a, b in segments[first]
                for c, d in segments[second]
            )
            assert distance >= MIN_CENTERLINE_CLEARANCE_MM, (
                f"{first.label} / {second.label}: {distance:.3f} mm separation"
            )


@pytest.mark.parametrize("example", CATALOG, ids=lambda entry: entry["key"])
@pytest.mark.parametrize("mode", MODES)
def test_race_layout_motion(example: dict, mode: str) -> None:
    suspension, sweep, states, stats = _solve_mode(example["geometry"], mode)
    design = suspension.initial_state()
    profile = _mode_profile(mode)
    assert all(info.converged for info in stats)
    assert not diagnose_sweep(suspension, states, stats).issues
    metrics = compute_sweep_metrics(suspension, sweep, states)
    assert metrics.derivative_error is None

    def axle_values(key: str) -> list[float]:
        values = []
        for row in metrics.rows:
            assert isinstance(row, AxleMetricRows)
            value = row.axle[key]
            assert value is not None
            values.append(float(value))
        return values

    def point(name: PointID, side: Side = Side.LEFT) -> PointRef:
        return PointRef(side, name)

    axis = (
        design.get(point(PointID.ROCKER_AXIS_B))
        - design.get(point(PointID.ROCKER_AXIS_A))
    ).data
    assert axis[1] == 0
    if "horizontal_rocker" in example["features"]:
        assert abs(axis[0]) > 10 * abs(axis[2])
    else:
        assert "vertical_rocker" in example["features"]
        assert abs(axis[2]) > 10 * abs(axis[0])
    rod_in = point(PointID.PUSHROD_INBOARD)
    rod_out = point(PointID.PUSHROD_OUTBOARD)
    assert (design.get(rod_in)[2] < design.get(rod_out)[2]) == (
        "pullrod" in example["features"]
    )
    rod_length = (design.get(rod_in) - design.get(rod_out)).norm()
    assert [
        (state.get(rod_in) - state.get(rod_out)).norm() for state in states
    ] == pytest.approx([rod_length] * len(states), abs=1e-5)

    types = [element.type for element in element_paths(suspension.assembly())]
    assert types.count(ElementType.HEAVE_LINK) == 1
    if "coilover" in example["features"]:
        assert types.count(ElementType.SPRING_DAMPER) == 2
        damper_a, damper_b = PointID.STRUT_TOP, PointID.STRUT_BOTTOM
    else:
        assert types.count(ElementType.TORSION_BAR) == 2
        assert types.count(ElementType.DAMPER) == 2
        damper_a, damper_b = PointID.DAMPER_CHASSIS, PointID.DAMPER_ROCKER

    if mode in {"bump", "web_bump"}:
        heave = axle_values("heave_link_length")
        if example["key"] == "formula_one_pushrod":
            # Preserve the original template: its below-axis pickup extends
            # the cross-car heave link in bump.
            assert np.all(np.diff(heave) > 0)
        else:
            assert np.all(np.diff(heave) < 0)
        assert max(abs(twist) for twist in axle_values("arb_twist")) < 1e-5
        for side in (Side.LEFT, Side.RIGHT):
            lengths = [
                (
                    state.get(point(damper_a, side)) - state.get(point(damper_b, side))
                ).norm()
                for state in states
            ]
            assert np.all(np.diff(lengths) < 0), (
                "Corner spring/damper must compress in bump"
            )
            travel = [
                state.get(point(PointID.WHEEL_CENTER, side))[2]
                - design.get(point(PointID.WHEEL_CENTER, side))[2]
                for state in states
            ]
            assert travel == pytest.approx(
                np.linspace(
                    profile["targets"][0]["start"],
                    profile["targets"][0]["stop"],
                    profile["steps"],
                ),
                abs=1e-5,
            )
    elif mode == "roll":
        twists = axle_values("arb_twist")
        assert twists[0] * twists[-1] < 0
        assert abs(twists[-1] - twists[0]) > 1
        assert abs(twists[len(twists) // 2]) < 1e-5
        heave = axle_values("heave_link_length")
        # Mirrored geometry cancels first-order heave in roll; finite travel
        # still produces a small second-order change in this cross-car link.
        assert heave[0] == pytest.approx(heave[-1], abs=1e-5)
        assert heave[len(heave) // 2 - 1] == pytest.approx(
            heave[len(heave) // 2 + 1], abs=1e-5
        )
