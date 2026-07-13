import numpy as np

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.metrics.catalog import get_default_corner_metrics
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.main import compute_metrics_for_state_from_suspension
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.point_ref import Side
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
    Shift a 3D point along the world X axis by a fixed amount.

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

    Hardpoints and any configuration points that live in world coordinates
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


def test_steering_axis_ground_intersection_uses_contact_patch_height(
    double_wishbone_geometry_file,
) -> None:
    """
    The steering-axis ground intersection should be evaluated on the
    horizontal plane through the contact patch, not on world Z = 0.
    """
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)
    assert suspension.config is not None

    from kinematics.core.primitives.geometry import Point3

    state = suspension.initial_state().copy()

    lower = state.get(PointID.LOWER_WISHBONE_OUTBOARD).copy()
    upper = state.get(PointID.UPPER_WISHBONE_OUTBOARD).copy()
    direction = upper - lower

    contact_patch_data = state.get(PointID.CONTACT_PATCH_CENTER).data.copy()
    contact_patch_data[2] = 123.456
    contact_patch = Point3(contact_patch_data)
    state[PointID.CONTACT_PATCH_CENTER] = contact_patch

    expected_t = (contact_patch[2] - lower[2]) / direction[2]
    expected_intersection = lower + expected_t * direction

    ctx = MetricContext(state=state, suspension=suspension, config=suspension.config)
    actual_intersection = ctx.steering_axis_ground_intersection

    assert actual_intersection is not None
    np.testing.assert_allclose(
        actual_intersection.data,
        expected_intersection.data,
        atol=TEST_TOLERANCE,
        err_msg="Steering-axis intersection should use contact patch Z height",
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


def test_scrub_radius_uses_ground_plane_wheel_lateral_direction(
    double_wishbone_geometry_file,
) -> None:
    """
    Scrub radius should use the wheel lateral direction in the ground
    plane, not the full 3D axle direction.
    """
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
    scrub_radius = metrics["scrub_radius"]
    roadwheel_angle = metrics["roadwheel_angle"]
    camber = metrics["camber"]

    assert scrub_radius is not None
    assert roadwheel_angle is not None
    assert camber is not None
    assert abs(roadwheel_angle) > 1.0
    assert abs(camber) > 1.0

    ctx = MetricContext(state=state, suspension=suspension, config=suspension.config)
    ground_pt = ctx.steering_axis_ground_intersection
    assert ground_pt is not None

    displacement = (ground_pt - ctx.contact_patch_center).data
    wheel_lateral_ground = ctx.wheel_axis.data.copy()
    wheel_lateral_ground[2] = 0.0
    wheel_lateral_ground /= np.linalg.norm(wheel_lateral_ground)

    expected_scrub_radius = -float(np.dot(displacement, wheel_lateral_ground))
    old_3d_axle_projection = -float(np.dot(displacement, ctx.wheel_axis.data))

    np.testing.assert_allclose(
        scrub_radius,
        expected_scrub_radius,
        atol=TEST_TOLERANCE,
        err_msg="Scrub radius should use wheel lateral direction on the ground plane",
    )
    assert not np.isclose(
        scrub_radius,
        old_3d_axle_projection,
        atol=1e-3,
    ), "Scrub radius should not use the full 3D axle direction"


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
