"""Tests for derived points that are reported but cannot be driven.

The wheel contact centre is an observable derived from the solved wheel
state. It is deliberately unsupported as an actuator because the axle
construction is branch-sensitive and has a bounded validity domain, so it is
declared output-only and rejected as a sweep target.
"""

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.enums import Axis, PointID
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.schema.sweep import SweepSpec, build_sweep_config
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.sweep import solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig

OUTPUT_ONLY_MESSAGE = "is a derived output of suspension type .* and cannot be driven"


def _target(
    point: PointID,
    axis: Axis,
    side: Side | None = Side.LEFT,
) -> dict[str, object]:
    """Build one relative single-step target specification."""
    target: dict[str, object] = {
        "point": point,
        "direction": {"axis": axis},
        "values": [0.0, 10.0],
    }
    if side is not None:
        target["side"] = side
    return target


def _spec(*targets: dict[str, object]) -> SweepSpec:
    """Validate a sweep specification from raw target mappings."""
    return SweepSpec.model_validate({"version": 1, "targets": list(targets)})


def test_corner_rejects_ground_tangent_sweep_target(test_data_dir: Path) -> None:
    corner = load_geometry(test_data_dir / "geometry.yaml")
    spec = _spec(
        _target(PointID.TRACKROD_INBOARD, Axis.Y),
        _target(PointID.WHEEL_CONTACT_CENTRE, Axis.Z),
    )

    with pytest.raises(ValueError, match=OUTPUT_ONLY_MESSAGE):
        build_sweep_config(spec, corner)


def test_output_only_rejection_uses_the_point_declaration_guidance(
    test_data_dir: Path,
) -> None:
    corner = load_geometry(test_data_dir / "geometry.yaml")
    spec = _spec(_target(PointID.WHEEL_CONTACT_CENTRE, Axis.Z))

    with pytest.raises(ValueError) as error:
        build_sweep_config(spec, corner)

    message = str(error.value)
    # The guidance names the honest control: wheel-centre Z is the heave
    # input, and ride height is read back from the solved metric.
    assert "'wheel_center'" in message
    assert "heave input" in message
    assert "'ride_height_change'" in message


def test_direct_sweep_config_cannot_bypass_output_only_validation(
    test_data_dir: Path,
) -> None:
    """The solve boundary rejects targets that bypass schema construction."""
    corner = load_geometry(test_data_dir / "geometry.yaml")
    sweep = SweepConfig(
        [
            [
                PointTarget(
                    PointID.WHEEL_CONTACT_CENTRE,
                    PointTargetAxis(Axis.Z),
                    0.0,
                )
            ]
        ]
    )

    with pytest.raises(ValueError, match=OUTPUT_ONLY_MESSAGE):
        solve_sweep(corner, sweep)


def test_macpherson_corner_rejects_ground_tangent_sweep_target(
    test_data_dir: Path,
) -> None:
    corner = load_geometry(test_data_dir / "macpherson_geometry.yaml")
    spec = _spec(_target(PointID.WHEEL_CONTACT_CENTRE, Axis.Z))

    with pytest.raises(ValueError, match=OUTPUT_ONLY_MESSAGE):
        build_sweep_config(spec, corner)


def test_axle_rejects_ground_tangent_sweep_target(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    spec = _spec(
        _target(PointID.WHEEL_CENTER, Axis.Z, Side.LEFT),
        _target(PointID.WHEEL_CONTACT_CENTRE, Axis.Z, Side.RIGHT),
        _target(PointID.TRACKROD_INBOARD, Axis.Y, Side.LEFT),
    )

    with pytest.raises(ValueError, match=OUTPUT_ONLY_MESSAGE):
        build_sweep_config(spec, axle)


def test_corner_accepts_wheel_center_sweep_target(test_data_dir: Path) -> None:
    corner = load_geometry(test_data_dir / "geometry.yaml")
    spec = _spec(
        _target(PointID.TRACKROD_INBOARD, Axis.Y),
        _target(PointID.WHEEL_CENTER, Axis.Z),
    )

    config = build_sweep_config(spec, corner)

    targets = [sweep[0] for sweep in config.target_sweeps]
    assert all(isinstance(target, PointTarget) for target in targets)
    assert [
        target.point_id for target in targets if isinstance(target, PointTarget)
    ] == [
        PointID.TRACKROD_INBOARD,
        PointID.WHEEL_CENTER,
    ]


def test_repeated_target_validation_reuses_the_validated_assembly(
    test_data_dir: Path,
) -> None:
    corner = load_geometry(test_data_dir / "geometry.yaml")

    first = corner.assembly()
    corner.validate_sweep_target_points([PointID.WHEEL_CENTER])

    assert corner.assembly() is first


def test_axle_accepts_wheel_center_sweep_target(test_data_dir: Path) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    spec = _spec(
        _target(PointID.WHEEL_CENTER, Axis.Z, Side.LEFT),
        _target(PointID.WHEEL_CENTER, Axis.Z, Side.RIGHT),
        _target(PointID.TRACKROD_INBOARD, Axis.Y, Side.LEFT),
    )

    config = build_sweep_config(spec, axle)

    targets = [sweep[0] for sweep in config.target_sweeps]
    assert all(isinstance(target, PointTarget) for target in targets)
    assert [
        target.point_id for target in targets if isinstance(target, PointTarget)
    ] == [
        PointRef(Side.LEFT, PointID.WHEEL_CENTER),
        PointRef(Side.RIGHT, PointID.WHEEL_CENTER),
        PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
    ]


def test_axle_catalog_marks_both_ground_tangents_output_only(
    test_data_dir: Path,
) -> None:
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)

    points = axle.assembly().points
    tangents = frozenset(
        {
            PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE),
            PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE),
        }
    )

    assert points.output_only == tangents
    # The coupled tangents are post-solve closure outputs: computed from the
    # state each solve, so published as derived — never free, never fixed.
    assert tangents <= points.derived
    assert not tangents & points.free
    assert not tangents & points.fixed
