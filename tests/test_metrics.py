from math import cos, radians, sin

import numpy as np

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.enums import Axis, AxlePosition, PointID
from kinematics.core.metrics.anti_geometry import (
    _cg_height_above_ground,
    calculate_anti_dive_pct,
    calculate_anti_squat_pct,
)
from kinematics.core.metrics.catalog import get_default_corner_metrics
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.ground import GroundDatum
from kinematics.core.metrics.main import compute_metrics_for_state_from_suspension
from kinematics.core.metrics.steering_geometry import (
    calculate_mechanical_trail,
    calculate_scrub_radius,
    calculate_steering_axis_offset_ground,
)
from kinematics.core.metrics.swing_arms import calculate_fvsa_length
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.primitives.geometry import Direction3, Vector3
from kinematics.core.primitives.point_ref import PointRef, Side
from kinematics.core.suspensions.axle import AxleSuspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.sweep import solve_sweep


def test_metric_side_sign_uses_declared_side(
    double_wishbone_geometry_file,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None
    state = suspension.initial_state()

    suspension.side = Side.RIGHT
    context = MetricContext(state, suspension, suspension.config)

    assert state.get(PointID.AXLE_OUTBOARD)[Axis.Y] > 0.0
    assert context.side_sign == -1.0


def test_metric_catalog_uses_supported_units() -> None:
    catalog = get_default_corner_metrics()

    assert {definition.unit for definition in catalog} == {
        MetricUnit.MM,
        MetricUnit.DEG,
        MetricUnit.PERCENT,
    }


def _shift_x(point: object, delta_x: float):
    """
    Shift a 3D point along the chassis X axis by a fixed amount.

    Returns a Point3 so it can be used with Pydantic's `model_copy(update=...)`
    which does not re-run field validators.
    """
    from kinematics.core.primitives.geometry import Point3, Vector3, extract_array

    return Point3(extract_array(point)) + Vector3([delta_x, 0.0, 0.0])


def _translate_double_wishbone_x(
    suspension: DoubleWishboneSuspension, delta_x: float
) -> DoubleWishboneSuspension:
    """
    Build a rigidly translated copy of a double wishbone suspension.

    Hardpoints and any configuration points that live in chassis-space coordinates
    are shifted together so the translated suspension is geometrically
    identical to the original one.
    """
    from kinematics.core.primitives.geometry import Vector3

    translation = Vector3([delta_x, 0.0, 0.0])
    hardpoints = {
        point_id: position + translation
        for point_id, position in suspension.hardpoints.items()
    }

    config = suspension.config
    translated_config = None
    if config is not None:
        config_updates: dict[str, object] = {
            "cg_position": _shift_x(config.cg_position, delta_x)
        }

        if config.camber_shim is not None:
            translated_shim = config.camber_shim.model_copy(
                update={
                    "shim_face_point_a": _shift_x(
                        config.camber_shim.shim_face_point_a, delta_x
                    ),
                    "shim_face_point_b": _shift_x(
                        config.camber_shim.shim_face_point_b, delta_x
                    ),
                }
            )
            config_updates["camber_shim"] = translated_shim

        translated_config = config.model_copy(update=config_updates)

    return DoubleWishboneSuspension(
        name=suspension.name,
        version=suspension.version,
        units=suspension.units,
        side=suspension.side,
        hardpoints=hardpoints,
        config=translated_config,
    )


def test_front_view_metrics_are_invariant_to_rigid_x_translation(
    double_wishbone_geometry_file,
    test_data_dir,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)

    sweep_config = load_sweep(test_data_dir / "sweep.yaml")
    states, _ = solve_sweep(suspension, sweep_config)

    translated = _translate_double_wishbone_x(suspension, 100.0)
    translated_states, _ = solve_sweep(translated, sweep_config)

    original_metrics = [
        compute_metrics_for_state_from_suspension(state, suspension) for state in states
    ]
    translated_metrics = [
        compute_metrics_for_state_from_suspension(state, translated)
        for state in translated_states
    ]

    comparison_index = next(
        index
        for index, metrics in enumerate(original_metrics)
        if metrics["fvic_y"] is not None
    )

    for column_name in ("fvic_y", "fvic_z", "fvsa_length"):
        original_value = original_metrics[comparison_index][column_name]
        translated_value = translated_metrics[comparison_index][column_name]
        assert original_value is not None, f"{column_name} is None in original"
        assert translated_value is not None, f"{column_name} is None in translated"
        np.testing.assert_allclose(
            original_value,
            translated_value,
            atol=TEST_TOLERANCE,
            rtol=TEST_TOLERANCE,
            err_msg=f"{column_name} changed under rigid X translation",
        )


def test_parallel_wishbone_planes_produce_null_ic_metrics(
    double_wishbone_geometry_file,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)

    from kinematics.core.primitives.geometry import Vector3

    state = suspension.initial_state().copy()
    plane_offset = Vector3([0.0, 0.0, 300.0])

    # Make the upper wishbone plane a translated copy of the lower
    # wishbone plane so the planes are parallel and have no unique
    # instant-axis intersection.
    state[PointID.UPPER_WISHBONE_INBOARD_FRONT] = (
        state[PointID.LOWER_WISHBONE_INBOARD_FRONT] + plane_offset
    )
    state[PointID.UPPER_WISHBONE_INBOARD_REAR] = (
        state[PointID.LOWER_WISHBONE_INBOARD_REAR] + plane_offset
    )
    state[PointID.UPPER_WISHBONE_OUTBOARD] = (
        state[PointID.LOWER_WISHBONE_OUTBOARD] + plane_offset
    )

    assert suspension.compute_instant_axis(state) is None
    assert suspension.compute_side_view_instant_center(state) is None
    assert suspension.compute_front_view_instant_center(state) is None

    metrics = compute_metrics_for_state_from_suspension(state, suspension)

    assert metrics["svic_x"] is None
    assert metrics["svic_z"] is None
    assert metrics["svsa_length"] is None
    assert metrics["fvic_y"] is None
    assert metrics["fvic_z"] is None
    assert metrics["fvsa_length"] is None


def test_steering_axis_ground_intersection_uses_ground_tangent_height(
    double_wishbone_geometry_file,
) -> None:
    """
    The steering-axis ground intersection should be evaluated on the
    horizontal plane through the wheel-ground tangent, not on chassis Z = 0.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None

    from kinematics.core.primitives.geometry import Point3

    state = suspension.initial_state().copy()

    lower = state.get(PointID.LOWER_WISHBONE_OUTBOARD).copy()
    upper = state.get(PointID.UPPER_WISHBONE_OUTBOARD).copy()
    direction = upper - lower

    ground_tangent_data = state.get(PointID.WHEEL_GROUND_TANGENT).data.copy()
    ground_tangent_data[2] = 123.456
    ground_tangent = Point3(ground_tangent_data)
    state[PointID.WHEEL_GROUND_TANGENT] = ground_tangent

    expected_t = (ground_tangent[2] - lower[2]) / direction[2]
    expected_intersection = lower + expected_t * direction

    ctx = MetricContext(state=state, suspension=suspension, config=suspension.config)
    actual_intersection = ctx.steering_axis_ground_intersection

    assert actual_intersection is not None
    np.testing.assert_allclose(
        actual_intersection.data,
        expected_intersection.data,
        atol=TEST_TOLERANCE,
        err_msg=("Steering-axis intersection should use wheel-ground tangent Z height"),
    )


def test_metric_context_exposes_cg_position(double_wishbone_geometry_file) -> None:
    """
    CG position should remain available after config coercion to Point3.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None

    state = suspension.initial_state()
    ctx = MetricContext(state=state, suspension=suspension, config=suspension.config)

    np.testing.assert_allclose(
        ctx.cg_position.data,
        suspension.config.cg_position.data,
        atol=TEST_TOLERANCE,
    )
    assert ctx.cg_position is not suspension.config.cg_position


def test_iso_steering_ground_metrics_use_wheel_relative_axes(
    double_wishbone_geometry_file,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None

    from kinematics.core.primitives.geometry import Vector3

    state = suspension.initial_state().copy()
    axle_inboard = state.get(PointID.AXLE_INBOARD).copy()

    # Force a state with both steer and camber so the ground-plane
    # projection differs measurably from the raw 3D axle direction.
    state[PointID.AXLE_OUTBOARD] = axle_inboard + Vector3(
        [120.0, 150.0, 120.0],
    )
    DerivedPointsManager(suspension.derived_spec()).update_in_place(state.positions)

    metrics = compute_metrics_for_state_from_suspension(state, suspension)
    steering_axis_offset = metrics["steering_axis_offset_ground"]
    scrub_radius = metrics["scrub_radius"]
    mechanical_trail = metrics["mechanical_trail"]
    roadwheel_angle = metrics["roadwheel_angle"]
    camber = metrics["camber"]

    assert steering_axis_offset is not None
    assert scrub_radius is not None
    assert mechanical_trail is not None
    assert roadwheel_angle is not None
    assert camber is not None
    assert abs(roadwheel_angle) > 1.0
    assert abs(camber) > 1.0

    ctx = MetricContext(state=state, suspension=suspension, config=suspension.config)
    ground_pt = ctx.steering_axis_ground_intersection
    assert ground_pt is not None

    displacement = ground_pt - ctx.wheel_ground_tangent
    ground_normal = ctx.ground.normal
    wheel_lateral_ground = (
        ctx.wheel_axis.vector() - ground_normal * ctx.wheel_axis.dot(ground_normal)
    ).normalize()
    wheel_longitudinal_ground = (
        ctx.side_sign * wheel_lateral_ground.cross(ground_normal)
    ).normalize()

    expected_offset = -float(displacement.dot(wheel_lateral_ground))
    expected_trail = float(displacement.dot(wheel_longitudinal_ground))
    expected_scrub = displacement.norm()

    np.testing.assert_allclose(
        steering_axis_offset, expected_offset, atol=TEST_TOLERANCE
    )
    np.testing.assert_allclose(mechanical_trail, expected_trail, atol=TEST_TOLERANCE)
    np.testing.assert_allclose(scrub_radius, expected_scrub, atol=TEST_TOLERANCE)
    np.testing.assert_allclose(
        scrub_radius,
        np.hypot(steering_axis_offset, mechanical_trail),
        atol=TEST_TOLERANCE,
    )
    assert not np.isclose(
        mechanical_trail,
        displacement[Axis.X],
        atol=1e-3,
    ), "Mechanical trail should follow tyre X_T, not chassis X"


def test_steering_geometry_uses_actual_banked_ground_plane(
    double_wishbone_geometry_file,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None

    from kinematics.core.primitives.geometry import Vector3

    state = suspension.initial_state().copy()
    axle_inboard = state.get(PointID.AXLE_INBOARD)
    state[PointID.AXLE_OUTBOARD] = axle_inboard + Vector3((80.0, 150.0, 60.0))
    tangent = state.get(PointID.WHEEL_GROUND_TANGENT)

    bank_angle = radians(12.0)
    tangent_y = cos(bank_angle)
    tangent_z = sin(bank_angle)
    normal_y = -tangent_z
    normal_z = tangent_y
    ground = GroundDatum.through(
        Direction3((0.0, normal_y, normal_z)),
        tangent,
    )
    ctx = MetricContext(
        state=state,
        suspension=suspension,
        config=suspension.config,
        ground=ground,
    )

    intersection = ctx.steering_axis_ground_intersection
    assert intersection is not None
    np.testing.assert_allclose(
        ground.signed_distance(intersection), 0.0, atol=TEST_TOLERANCE
    )

    ground_normal = ctx.ground.normal
    projected_axis = (
        ctx.wheel_axis.vector() - ground_normal * ctx.wheel_axis.dot(ground_normal)
    ).normalize()
    displacement = intersection - ctx.wheel_ground_tangent
    expected_offset = -float(displacement.dot(projected_axis))
    forward_axis = (ctx.side_sign * projected_axis.cross(ground_normal)).normalize()
    expected_trail = float(displacement.dot(forward_axis))
    expected_scrub = displacement.norm()

    steering_axis_offset = calculate_steering_axis_offset_ground(ctx)
    scrub_radius = calculate_scrub_radius(ctx)
    mechanical_trail = calculate_mechanical_trail(ctx)
    assert steering_axis_offset is not None
    assert scrub_radius is not None
    assert mechanical_trail is not None
    np.testing.assert_allclose(
        steering_axis_offset, expected_offset, atol=TEST_TOLERANCE
    )
    np.testing.assert_allclose(scrub_radius, expected_scrub, atol=TEST_TOLERANCE)
    np.testing.assert_allclose(mechanical_trail, expected_trail, atol=TEST_TOLERANCE)


def _banked_ground_through(point, bank_angle_deg: float) -> GroundDatum:
    """Build a ground datum rolled by ``bank_angle_deg`` through ``point``."""
    bank_angle = radians(bank_angle_deg)
    normal_y = -sin(bank_angle)
    normal_z = cos(bank_angle)
    return GroundDatum.through(Direction3((0.0, normal_y, normal_z)), point)


def test_anti_geometry_uses_perpendicular_cg_height_above_the_ground_plane(
    double_wishbone_geometry_file,
    test_data_dir,
) -> None:
    """
    The anti formulas take the CG's perpendicular distance to the ground plane,
    which on a banked ground line differs from the raw chassis-Z difference,
    and resolve the reaction-line rise along the same ground normal.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None
    config = suspension.config.model_copy(
        update={"axle_position": AxlePosition.FRONT, "front_brake_bias": 0.6}
    )
    # The design state has side-view-parallel wishbones, so its SVIC is at
    # infinity; take a swept state where the anti formulas are defined.
    states, _ = solve_sweep(suspension, load_sweep(test_data_dir / "sweep.yaml"))
    state = next(
        candidate
        for candidate in states
        if suspension.compute_side_view_instant_center(candidate) is not None
    )
    tangent = state.get(PointID.WHEEL_GROUND_TANGENT)

    ground = _banked_ground_through(tangent, 12.0)
    ctx = MetricContext(
        state=state,
        suspension=suspension,
        config=config,
        ground=ground,
    )
    cg = config.cg_position

    expected_height = ground.signed_distance(cg)
    height = _cg_height_above_ground(ctx)
    assert height is not None
    np.testing.assert_allclose(height, expected_height, atol=TEST_TOLERANCE)

    chassis_z_height = float(cg[Axis.Z]) - float(tangent[Axis.Z])
    assert not np.isclose(height, chassis_z_height, atol=1e-3), (
        "Banked ground should not reduce to the chassis-Z height difference"
    )

    svic = ctx.side_view_ic
    assert svic is not None
    run = float(tangent[Axis.X]) - float(svic[Axis.X])
    # The datum passes through the tangent, so the tangent -> SVIC rise along
    # the ground normal is n . (SVIC - T), written out with the datum's
    # normal components rather than through the implementation's helpers.
    expected_rise = ground.normal.dot(svic - tangent)
    expected_tan_theta = expected_rise / run
    expected_anti_dive = (
        100.0 * 0.6 * (config.wheelbase / expected_height) * expected_tan_theta
    )
    anti_dive = calculate_anti_dive_pct(ctx)
    assert anti_dive is not None
    np.testing.assert_allclose(anti_dive, expected_anti_dive, atol=TEST_TOLERANCE)

    # The pre-fix hybrid formula took the rise as raw chassis delta-Z; on a
    # banked datum it must disagree, or a silent revert would go unnoticed.
    hybrid_tan_theta = (float(svic[Axis.Z]) - float(tangent[Axis.Z])) / run
    hybrid_anti_dive = (
        100.0 * 0.6 * (config.wheelbase / expected_height) * hybrid_tan_theta
    )
    assert not np.isclose(anti_dive, hybrid_anti_dive, atol=TEST_TOLERANCE)

    # A level ground line through the same tangent must still give the
    # chassis-Z height, leaving flat-ground anti percentages untouched.
    flat_ctx = MetricContext(state=state, suspension=suspension, config=config)
    flat_height = _cg_height_above_ground(flat_ctx)
    assert flat_height is not None
    np.testing.assert_allclose(flat_height, chassis_z_height, atol=TEST_TOLERANCE)


def test_anti_dive_on_flat_ground_equals_the_chassis_z_formula(
    double_wishbone_geometry_file,
    test_data_dir,
) -> None:
    """
    On the default flat datum the ground normal is +Z, so the ground-normal
    rise and CG height reduce exactly to the chassis-Z differences of the
    classic flat-ground formula.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None
    config = suspension.config.model_copy(
        update={"axle_position": AxlePosition.FRONT, "front_brake_bias": 0.6}
    )
    states, _ = solve_sweep(suspension, load_sweep(test_data_dir / "sweep.yaml"))
    state = next(
        candidate
        for candidate in states
        if suspension.compute_side_view_instant_center(candidate) is not None
    )
    ctx = MetricContext(state=state, suspension=suspension, config=config)

    tangent = state.get(PointID.WHEEL_GROUND_TANGENT)
    svic = ctx.side_view_ic
    assert svic is not None
    run = float(tangent[Axis.X]) - float(svic[Axis.X])
    tan_theta = (float(svic[Axis.Z]) - float(tangent[Axis.Z])) / run
    height = float(config.cg_position[Axis.Z]) - float(tangent[Axis.Z])
    expected = 100.0 * 0.6 * (config.wheelbase / height) * tan_theta

    anti_dive = calculate_anti_dive_pct(ctx)
    assert anti_dive is not None
    assert anti_dive == expected


def test_anti_squat_resolves_the_rise_along_a_banked_ground_normal(
    double_wishbone_geometry_file,
    test_data_dir,
) -> None:
    """
    Anti-squat's wheel-center -> SVIC line has neither endpoint on the ground
    plane, so its rise is the endpoints' signed-distance difference along the
    ground normal rather than the raw chassis-Z difference.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None
    config = suspension.config.model_copy(
        update={
            "axle_position": AxlePosition.REAR,
            "driven_axle": AxlePosition.REAR,
        }
    )
    states, _ = solve_sweep(suspension, load_sweep(test_data_dir / "sweep.yaml"))
    state = next(
        candidate
        for candidate in states
        if suspension.compute_side_view_instant_center(candidate) is not None
    )
    tangent = state.get(PointID.WHEEL_GROUND_TANGENT)
    ground = _banked_ground_through(tangent, 12.0)
    ctx = MetricContext(
        state=state, suspension=suspension, config=config, ground=ground
    )

    svic = ctx.side_view_ic
    assert svic is not None
    wc = state.get(PointID.WHEEL_CENTER)
    run = float(svic[Axis.X]) - float(wc[Axis.X])
    # n . (SVIC - WC), written out with the datum's normal components; the
    # plane offset cancels in the endpoint difference.
    expected_rise = ground.normal.dot(svic - wc)
    cg = config.cg_position
    height = ground.signed_distance(cg)
    expected = 100.0 * (config.wheelbase / height) * (expected_rise / run)

    anti_squat = calculate_anti_squat_pct(ctx)
    assert anti_squat is not None
    np.testing.assert_allclose(anti_squat, expected, atol=TEST_TOLERANCE)

    # The pre-fix hybrid formula took the rise as raw chassis delta-Z; on a
    # banked datum it must disagree, or a silent revert would go unnoticed.
    hybrid = (
        100.0
        * (config.wheelbase / height)
        * ((float(svic[Axis.Z]) - float(wc[Axis.Z])) / run)
    )
    assert not np.isclose(anti_squat, hybrid, atol=TEST_TOLERANCE)


def test_anti_cg_height_is_shared_by_both_corners_of_a_banked_axle(
    test_data_dir,
) -> None:
    """Both corners of an axle measure the CG against the same ground line."""
    axle = load_geometry(test_data_dir / "axle_geometry.yaml")
    assert isinstance(axle, AxleSuspension)
    assert axle.config is not None

    state = axle.initial_state().copy()
    left_tangent_ref = PointRef(Side.LEFT, PointID.WHEEL_GROUND_TANGENT)
    left_tangent = state.get(left_tangent_ref)
    state.set(left_tangent_ref, left_tangent + Vector3((0.0, 0.0, 40.0)))

    left = state.get(left_tangent_ref)
    right = state.get(PointRef(Side.RIGHT, PointID.WHEEL_GROUND_TANGENT))
    lateral = (left - right).normalize()
    ground = GroundDatum.through(
        Direction3(Direction3((1.0, 0.0, 0.0)).cross(lateral)),
        left,
    )
    assert abs(float(ground.normal[Axis.Y])) > 0.01

    heights: dict[Side, float] = {}
    chassis_z_heights: dict[Side, float] = {}
    for side in (Side.LEFT, Side.RIGHT):
        corner = axle.corners[side]
        corner_config = corner.config if corner.config is not None else axle.config
        corner_state = axle.corner_state(state, side)
        ctx = MetricContext(
            state=corner_state,
            suspension=corner,
            config=corner_config,
            ground=ground,
        )
        height = _cg_height_above_ground(ctx)
        assert height is not None
        heights[side] = height
        chassis_z_heights[side] = float(corner_config.cg_position[Axis.Z]) - float(
            corner_state.get(PointID.WHEEL_GROUND_TANGENT)[Axis.Z]
        )

    np.testing.assert_allclose(
        heights[Side.LEFT], heights[Side.RIGHT], atol=TEST_TOLERANCE
    )
    assert not np.isclose(
        chassis_z_heights[Side.LEFT], chassis_z_heights[Side.RIGHT], atol=1e-3
    ), "The chassis-Z heights must disagree for this test to be meaningful"


def test_fvsa_sign_follows_the_ground_line_rather_than_chassis_y(
    double_wishbone_geometry_file,
) -> None:
    """
    The FVSA magnitude is a plain YZ distance, but its inboard/outboard sign
    is resolved along the ground line, so a steep bank can flip it.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None
    state = suspension.initial_state()
    tangent = state.get(PointID.WHEEL_GROUND_TANGENT)

    # Place the FVIC so its chassis-Y and along-ground components disagree on
    # a 45-degree bank; the solved geometry never sits this close to the
    # tangent, so the case has to be posed directly.
    offset = Vector3((0.0, 100.0, -200.0))
    ground = _banked_ground_through(tangent, 45.0)
    ctx = MetricContext(
        state=state,
        suspension=suspension,
        config=suspension.config,
        ground=ground,
    )
    ctx.front_view_ic = tangent + offset

    expected_magnitude = float(np.hypot(offset[Axis.Y], offset[Axis.Z]))
    along_ground = ground.lateral.dot(offset)
    expected = expected_magnitude * (-ctx.side_sign * np.sign(along_ground))

    fvsa_length = calculate_fvsa_length(ctx)
    assert fvsa_length is not None
    np.testing.assert_allclose(fvsa_length, expected, atol=TEST_TOLERANCE)

    flat_ctx = MetricContext(
        state=state, suspension=suspension, config=suspension.config
    )
    flat_ctx.front_view_ic = tangent + offset
    flat_fvsa_length = calculate_fvsa_length(flat_ctx)
    assert flat_fvsa_length is not None
    np.testing.assert_allclose(
        flat_fvsa_length,
        expected_magnitude * (-ctx.side_sign * np.sign(float(offset[Axis.Y]))),
        atol=TEST_TOLERANCE,
    )
    np.testing.assert_allclose(
        flat_fvsa_length,
        -fvsa_length,
        atol=TEST_TOLERANCE,
    )


class TestSignConventionsAndKnownValues:
    """
    Direct validation tests for metric sign conventions and
    known-value cases using the test geometry.
    """

    def test_camber_sign_negative_means_top_tilted_inward(
        self, double_wishbone_geometry_file
    ) -> None:
        """
        The test geometry has the upper ball joint inboard of the lower,
        tilting the top of the wheel inward. Camber must be negative.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        state = suspension.initial_state()
        metrics = compute_metrics_for_state_from_suspension(state, suspension)

        camber = metrics["camber"]
        assert camber is not None
        assert camber < 0, f"Expected negative camber (top tilted inward), got {camber}"

    def test_camber_known_value_at_design_position(
        self, double_wishbone_geometry_file
    ) -> None:
        """
        Verify the camber value at design position against a hand-checked
        reference. The axle vector has a small Z component over a 150 mm
        lateral span, giving roughly -1.9 degrees.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        state = suspension.initial_state()
        metrics = compute_metrics_for_state_from_suspension(state, suspension)

        camber = metrics["camber"]
        assert camber is not None, "camber is None"
        np.testing.assert_allclose(
            camber,
            -1.909,
            atol=TEST_TOLERANCE,
            err_msg="Camber at design position",
        )

    def test_caster_sign_positive_means_top_tilted_rearward(
        self, double_wishbone_geometry_file
    ) -> None:
        """
        The test geometry has the upper ball joint behind the lower
        (X = -25 vs X = 0), tilting the steering axis top rearward.
        Caster must be positive.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        state = suspension.initial_state()
        metrics = compute_metrics_for_state_from_suspension(state, suspension)

        caster = metrics["caster"]
        assert caster is not None
        assert caster > 0, (
            f"Expected positive caster (top tilted rearward), got {caster}"
        )

    def test_caster_known_value_at_design_position(
        self, double_wishbone_geometry_file
    ) -> None:
        """
        Verify the caster value at design position. The steering axis
        from lower (0, 900, 200) to upper (-25, 750, 500) gives roughly
        4.76 degrees.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        state = suspension.initial_state()
        metrics = compute_metrics_for_state_from_suspension(state, suspension)

        caster = metrics["caster"]
        assert caster is not None, "caster is None"
        np.testing.assert_allclose(
            caster,
            4.764,
            atol=TEST_TOLERANCE,
            err_msg="Caster at design position",
        )

    def test_roadwheel_angle_zero_at_design_position(
        self, double_wishbone_geometry_file
    ) -> None:
        """
        At the design position with no steering input the axle is
        purely lateral, so the roadwheel angle must be zero.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        state = suspension.initial_state()
        metrics = compute_metrics_for_state_from_suspension(state, suspension)

        roadwheel_angle = metrics["roadwheel_angle"]
        assert roadwheel_angle is not None, "roadwheel_angle is None"
        np.testing.assert_allclose(
            roadwheel_angle,
            0.0,
            atol=TEST_TOLERANCE,
            err_msg="Roadwheel angle at design position",
        )

    def test_roadwheel_angle_positive_means_turned_inward(
        self, double_wishbone_geometry_file, test_data_dir
    ) -> None:
        """
        During a toe-in sweep (positive roadwheel angle), the front
        of the wheel points toward the vehicle center. Verify the first
        sweep step produces a positive angle for the left-side suspension.
        """
        suspension = load_geometry(double_wishbone_geometry_file)
        sweep_config = load_sweep(test_data_dir / "sweep.yaml")
        states, _ = solve_sweep(suspension, sweep_config)

        first_metrics = compute_metrics_for_state_from_suspension(states[0], suspension)
        last_metrics = compute_metrics_for_state_from_suspension(states[-1], suspension)

        first_rwa = first_metrics["roadwheel_angle"]
        last_rwa = last_metrics["roadwheel_angle"]
        assert first_rwa is not None
        assert last_rwa is not None

        # The sweep goes from positive to negative roadwheel angle,
        # confirming both sign directions.
        assert first_rwa > 0, "Expected positive roadwheel angle at start of sweep"
        assert last_rwa < 0, "Expected negative roadwheel angle at end of sweep"


def test_default_corner_metric_catalog_matches_trusted_set() -> None:
    column_names = [metric.column_name for metric in get_default_corner_metrics()]

    expected = [
        "camber",
        "caster",
        "kpi",
        "steering_axis_offset_ground",
        "scrub_radius",
        "mechanical_trail",
        "roadwheel_angle",
        "svic_x",
        "svic_z",
        "svsa_length",
        "fvic_y",
        "fvic_z",
        "fvsa_length",
        "wheel_travel",
        "half_track",
        "damper_length",
        "svsa_angle",
        "anti_dive",
        "anti_lift",
        "anti_squat",
    ]
    assert column_names == expected
