"""
Tests for the Upright rigid body implementation.

This module verifies:
1. Local coordinate system construction from hardpoints
2. Attachment transformations during suspension travel
3. Camber shim application (rotates attachments only, not hardpoints)
"""

import numpy as np
import pytest

from kinematics.components.upright import Upright, UprightAttachments, UprightHardpoints
from kinematics.core.enums import PointID
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.rigid_body import LocalCoordinateSystem


class TestLocalCoordinateSystem:
    """
    Tests for the LocalCoordinateSystem class.
    """

    def test_world_to_local_at_origin(self):
        """
        Test that a point at the origin transforms to (0,0,0) in local coords.
        """
        lcs = LocalCoordinateSystem(
            origin=Point3([0, 0, 0]),
            x_axis=Direction3([1, 0, 0]),
            y_axis=Direction3([0, 1, 0]),
            z_axis=Direction3([0, 0, 1]),
        )

        world_point = Point3([0, 0, 0])
        local_point = lcs.world_to_local(world_point)

        np.testing.assert_allclose(local_point.data, [0, 0, 0], atol=1e-10)

    def test_local_to_world_identity(self):
        """
        Test that local (1,0,0) transforms to world x-axis direction.
        """
        lcs = LocalCoordinateSystem(
            origin=Point3([10, 20, 30]),
            x_axis=Direction3([1, 0, 0]),
            y_axis=Direction3([0, 1, 0]),
            z_axis=Direction3([0, 0, 1]),
        )

        from kinematics.core.geometry import Vector3

        local_point = Vector3([5, 0, 0])
        world_point = lcs.local_to_world(local_point)

        # Should be origin + 5 in X direction
        np.testing.assert_allclose(world_point.data, [15, 20, 30], atol=1e-10)

    def test_round_trip_transformation(self):
        """
        Test that world->local->world returns the original point.
        """
        lcs = LocalCoordinateSystem(
            origin=Point3([100, 200, 300]),
            x_axis=Direction3([1, 0, 0]),
            y_axis=Direction3([0, 1, 0]),
            z_axis=Direction3([0, 0, 1]),
        )

        original_world = Point3([150, 250, 350])
        local = lcs.world_to_local(original_world)
        recovered_world = lcs.local_to_world(local)

        np.testing.assert_allclose(
            recovered_world.data, original_world.data, atol=1e-10
        )

    def test_from_three_points_vertical_z(self):
        """
        Test LCS construction with Z axis pointing vertically.
        """
        origin = Point3([0, 900, 200])  # Lower ball joint
        z_point = Point3([0, 900, 500])  # Upper ball joint (directly above)
        y_reference = Point3([150, 800, 275])  # Tie rod (forward and inboard)

        lcs = LocalCoordinateSystem.from_three_points(origin, z_point, y_reference)

        # Z axis should point straight up
        np.testing.assert_allclose(lcs.z_axis.data, [0, 0, 1], atol=1e-10)

        # Y axis should be in the XY plane (no Z component) and point generally forward
        assert abs(lcs.y_axis[2]) < 1e-10, "Y axis should be in XY plane"
        assert lcs.y_axis[0] > 0, "Y axis should point forward (positive X)"

        # X axis should complete the right-handed system
        x_computed = np.cross(lcs.y_axis.data, lcs.z_axis.data)
        np.testing.assert_allclose(lcs.x_axis.data, x_computed, atol=1e-10)


class TestUprightRigidBody:
    """
    Tests for the Upright rigid body class.
    """

    @pytest.fixture
    def upright_design_positions(self):
        """
        Design positions for a typical double wishbone upright.

        These correspond to a suspension in its nominal design position.
        Coordinate system: X=forward, Y=outboard, Z=up
        """
        hardpoints = UprightHardpoints(
            lower_ball_joint=Point3([0, 900, 200]),
            upper_ball_joint=Point3([-25, 750, 500]),
            trackrod_outboard=Point3([150, 800, 275]),
        )

        attachments = UprightAttachments(
            axle_inboard=Point3([-20, 800, 308.426]),
            axle_outboard=Point3([-20, 950, 313.426]),
        )

        return hardpoints, attachments

    def test_upright_initialization(self, upright_design_positions):
        """
        Test that upright can be created and baked from global positions.
        """
        hardpoints, attachments = upright_design_positions

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Verify LCS is established
        assert upright.lcs is not None

        # Verify attachment local offsets are computed
        assert "axle_inboard" in upright.attachment_local_offsets
        assert "axle_outboard" in upright.attachment_local_offsets

    def test_attachment_world_positions_match_input(self, upright_design_positions):
        """
        Test that getting world positions returns the original input positions.
        """
        hardpoints, attachments = upright_design_positions

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Get world positions and verify they match the inputs
        axle_inboard_world = upright.get_world_position("axle_inboard")
        axle_outboard_world = upright.get_world_position("axle_outboard")

        np.testing.assert_allclose(
            axle_inboard_world.data, attachments.axle_inboard.data, atol=1e-6
        )
        np.testing.assert_allclose(
            axle_outboard_world.data, attachments.axle_outboard.data, atol=1e-6
        )

    def test_camber_shim_rotates_attachments_not_hardpoints(
        self, upright_design_positions
    ):
        """Verify that camber shim rotates attachments but NOT hardpoints.

        The shim sits between the structural upright (hardpoints) and the hub
        (attachments). Ball joint positions must remain unchanged.
        """
        hardpoints, attachments = upright_design_positions

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Store original hardpoint positions
        original_upper_bj = upright.hardpoints.upper_ball_joint.copy()
        original_lower_bj = upright.hardpoints.lower_ball_joint.copy()
        original_trackrod = upright.hardpoints.trackrod_outboard.copy()

        # Store original attachment positions
        original_axle_inboard = upright.get_world_position("axle_inboard").copy()

        # Apply a camber shim rotation
        # Rotation about lower ball joint, rotating about X axis (fore-aft)
        pivot = upright.hardpoints.lower_ball_joint
        rotation_axis = Direction3([1, 0, 0])  # Rotate about X
        rotation_angle = np.deg2rad(2.0)  # 2 degrees

        upright.apply_camber_shim(pivot, rotation_axis, rotation_angle)

        # VERIFY: Hardpoints have NOT moved
        np.testing.assert_allclose(
            upright.hardpoints.upper_ball_joint.data,
            original_upper_bj.data,
            atol=1e-10,
            err_msg="Upper ball joint MUST NOT move when shim is applied",
        )
        np.testing.assert_allclose(
            upright.hardpoints.lower_ball_joint.data,
            original_lower_bj.data,
            atol=1e-10,
            err_msg="Lower ball joint MUST NOT move when shim is applied",
        )
        np.testing.assert_allclose(
            upright.hardpoints.trackrod_outboard.data,
            original_trackrod.data,
            atol=1e-10,
            err_msg="Trackrod outboard MUST NOT move when shim is applied",
        )

        # VERIFY: Attachments HAVE moved
        new_axle_inboard = upright.get_world_position("axle_inboard")
        assert not np.allclose(
            new_axle_inboard.data, original_axle_inboard.data, atol=1e-6
        ), "Axle attachment MUST move when shim is applied"

    def test_camber_shim_produces_expected_camber_change(
        self, upright_design_positions
    ):
        """
        Test that applying a camber shim produces the expected camber angle change.

        This verifies the physics of the shim rotation.
        """
        hardpoints, attachments = upright_design_positions

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Get initial axle vector (before shim)
        initial_axle_vec = upright.get_axle_vector()

        # Apply a small rotation about the fore-aft axis (X)
        # This should change the camber angle
        pivot = upright.hardpoints.lower_ball_joint
        rotation_axis = Direction3([1, 0, 0])
        rotation_angle = np.deg2rad(1.0)  # 1 degree rotation

        upright.apply_camber_shim(pivot, rotation_axis, rotation_angle)

        # Get new axle vector
        new_axle_vec = upright.get_axle_vector()

        # Compute angle change between initial and new axle vectors
        cos_angle = np.dot(initial_axle_vec, new_axle_vec)
        angle_change_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_change_deg = np.rad2deg(angle_change_rad)

        # The angle change should be approximately equal to the rotation angle
        # (may differ slightly due to the geometry of the rotation)
        assert 0.5 < angle_change_deg < 1.5, (
            f"Expected ~1 degree camber change, got {angle_change_deg:.2f} degrees"
        )

    def test_update_from_hardpoints_preserves_attachments_in_local_frame(
        self, upright_design_positions
    ):
        """
        Test that when hardpoints move (during suspension travel), attachments move
        rigidly with them.

        The local offsets should remain constant while the world positions update.
        """
        hardpoints, attachments = upright_design_positions

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Store original local offset and world position BEFORE modifying
        original_local_offset = upright.attachment_local_offsets["axle_inboard"].copy()
        original_world_z = upright.get_world_position("axle_inboard")[2]

        # Simulate suspension bump: move all hardpoints up by 50mm
        from kinematics.core.geometry import Vector3

        bump = Vector3([0, 0, 50])
        new_hardpoints = {
            "upper_ball_joint": upright.hardpoints.upper_ball_joint + bump,
            "lower_ball_joint": upright.hardpoints.lower_ball_joint + bump,
            "trackrod_outboard": upright.hardpoints.trackrod_outboard + bump,
        }

        upright.update_from_hardpoints(new_hardpoints)

        # VERIFY: Local offset is unchanged
        new_local_offset = upright.attachment_local_offsets["axle_inboard"]
        np.testing.assert_allclose(
            new_local_offset.data,
            original_local_offset.data,
            atol=1e-10,
            err_msg="Local offset must remain constant during suspension travel",
        )

        # VERIFY: World position has moved
        new_world = upright.get_world_position("axle_inboard")
        # Axle should have moved up by ~50mm (exact value depends on geometry).
        assert abs(new_world[2] - original_world_z - 50) < 10, (
            "Attachment world position should move with hardpoints"
        )


class TestUprightIntegration:
    """
    Integration tests verifying the full workflow.
    """

    def test_complete_shim_workflow(self):
        """Test the complete shim workflow from design through suspension travel."""
        # Step 1: Create upright at design
        hardpoints = UprightHardpoints(
            lower_ball_joint=Point3([0, 900, 200]),
            upper_ball_joint=Point3([-25, 750, 500]),
            trackrod_outboard=Point3([150, 800, 275]),
        )
        attachments = UprightAttachments(
            axle_inboard=Point3([-20, 800, 308.426]),
            axle_outboard=Point3([-20, 950, 313.426]),
        )

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Step 2: Apply camber shim (e.g., adding 2mm outboard shim).
        # Normally computed from shim geometry, but we'll use a direct rotation.
        pivot = upright.hardpoints.lower_ball_joint
        rotation_axis = Direction3([1, 0, 0])  # About fore-aft axis
        rotation_angle = np.deg2rad(0.5)  # Small rotation

        upright.apply_camber_shim(pivot, rotation_axis, rotation_angle)

        # Store post-shim local offsets
        shim_local_offset = upright.attachment_local_offsets["axle_inboard"].copy()

        # Step 3: Simulate suspension travel (50mm bump)
        from kinematics.core.geometry import Vector3

        bump_ubj = Vector3([0, 5, 50])
        bump_lbj = Vector3([0, 3, 45])
        bump_tro = Vector3([0, 4, 48])
        new_hardpoints = {
            "upper_ball_joint": upright.hardpoints.upper_ball_joint + bump_ubj,
            "lower_ball_joint": upright.hardpoints.lower_ball_joint + bump_lbj,
            "trackrod_outboard": upright.hardpoints.trackrod_outboard + bump_tro,
        }

        upright.update_from_hardpoints(new_hardpoints)

        # Step 4: Verify local offset is still the shimmed value
        final_local_offset = upright.attachment_local_offsets["axle_inboard"]
        np.testing.assert_allclose(
            final_local_offset.data,
            shim_local_offset.data,
            atol=1e-10,
            err_msg="Shim effect must persist through suspension travel",
        )


class TestHardpointsArchitecture:
    """
    Tests for the hardpoints and attachments architecture methods.
    """

    @pytest.fixture
    def hardpoints_registry(self) -> dict[PointID, Point3]:
        """
        Create a hardpoints registry with typical double wishbone points.

        This represents the central hardpoints registry that components reference.
        """
        return {
            PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3([250, 400, 200]),
            PointID.LOWER_WISHBONE_INBOARD_REAR: Point3([-250, 450, 200]),
            PointID.LOWER_WISHBONE_OUTBOARD: Point3([0, 900, 200]),
            PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3([225, 350, 500]),
            PointID.UPPER_WISHBONE_INBOARD_REAR: Point3([-275, 350, 500]),
            PointID.UPPER_WISHBONE_OUTBOARD: Point3([-25, 750, 500]),
            PointID.TRACKROD_INBOARD: Point3([50, 200, 250]),
            PointID.TRACKROD_OUTBOARD: Point3([150, 800, 275]),
        }

    @pytest.fixture
    def hardpoint_point_ids(self) -> dict[str, PointID]:
        """
        Standard mount ID mapping for double wishbone upright.
        """
        return {
            "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
            "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
            "trackrod_outboard": PointID.TRACKROD_OUTBOARD,
        }

    @pytest.fixture
    def attachments(self) -> dict[str, Point3]:
        """
        Attachment positions (axle) in global coordinates at design.
        """
        return {
            "axle_inboard": Point3([-20, 800, 308.426]),
            "axle_outboard": Point3([-20, 950, 313.426]),
        }

    def test_from_hardpoints_and_attachments_creates_upright(
        self, hardpoints_registry, hardpoint_point_ids, attachments
    ):
        """
        Test that from_hardpoints_and_attachments creates a valid upright.
        """
        upright = Upright.from_hardpoints_and_attachments(
            hardpoint_point_ids, hardpoints_registry, attachments
        )

        # Verify LCS is established
        assert upright.lcs is not None

        # Verify mount IDs are stored
        assert upright.hardpoint_point_ids is not None
        assert (
            upright.hardpoint_point_ids["upper_ball_joint"]
            == PointID.UPPER_WISHBONE_OUTBOARD
        )
        assert (
            upright.hardpoint_point_ids["lower_ball_joint"]
            == PointID.LOWER_WISHBONE_OUTBOARD
        )
        assert (
            upright.hardpoint_point_ids["trackrod_outboard"]
            == PointID.TRACKROD_OUTBOARD
        )

        # Verify hardpoints match registry values
        np.testing.assert_allclose(
            upright.hardpoints.upper_ball_joint.data,
            hardpoints_registry[PointID.UPPER_WISHBONE_OUTBOARD].data,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            upright.hardpoints.lower_ball_joint.data,
            hardpoints_registry[PointID.LOWER_WISHBONE_OUTBOARD].data,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            upright.hardpoints.trackrod_outboard.data,
            hardpoints_registry[PointID.TRACKROD_OUTBOARD].data,
            atol=1e-10,
        )

    def test_from_hardpoints_and_attachments_computes_local_offsets(
        self, hardpoints_registry, hardpoint_point_ids, attachments
    ):
        """
        Test that local offsets are computed for attachments.
        """
        upright = Upright.from_hardpoints_and_attachments(
            hardpoint_point_ids, hardpoints_registry, attachments
        )

        # Verify attachment local offsets are computed
        assert "axle_inboard" in upright.attachment_local_offsets
        assert "axle_outboard" in upright.attachment_local_offsets

        # Verify world positions match input attachments
        np.testing.assert_allclose(
            upright.get_world_position("axle_inboard").data,
            attachments["axle_inboard"].data,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            upright.get_world_position("axle_outboard").data,
            attachments["axle_outboard"].data,
            atol=1e-6,
        )

    def test_from_hardpoints_and_attachments_copies_point_id_mapping(
        self, hardpoints_registry, hardpoint_point_ids, attachments
    ):
        upright = Upright.from_hardpoints_and_attachments(
            hardpoint_point_ids, hardpoints_registry, attachments
        )

        hardpoint_point_ids["upper_ball_joint"] = PointID.LOWER_WISHBONE_OUTBOARD

        assert upright.hardpoint_point_ids is not None
        assert (
            upright.hardpoint_point_ids["upper_ball_joint"]
            == PointID.UPPER_WISHBONE_OUTBOARD
        )

    def test_from_hardpoints_and_attachments_missing_mount_raises(
        self, hardpoints_registry, attachments
    ):
        """
        Test that missing mount IDs raise ValueError.
        """
        incomplete_hardpoint_point_ids = {
            "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
            # Missing lower_ball_joint and trackrod_outboard
        }

        with pytest.raises(ValueError, match="Missing required mounts"):
            Upright.from_hardpoints_and_attachments(
                incomplete_hardpoint_point_ids, hardpoints_registry, attachments
            )

    def test_from_hardpoints_and_attachments_missing_attachment_raises(
        self, hardpoints_registry, hardpoint_point_ids
    ):
        """
        Test that missing attachments raise ValueError.
        """
        incomplete_attachments: dict[str, Point3] = {
            "axle_inboard": Point3([-20, 800, 308.426]),
            # Missing axle_outboard
        }

        with pytest.raises(ValueError, match="Missing required attachments"):
            Upright.from_hardpoints_and_attachments(
                hardpoint_point_ids, hardpoints_registry, incomplete_attachments
            )

    def test_update_from_hardpoints_registry_updates_hardpoints(
        self, hardpoints_registry, hardpoint_point_ids, attachments
    ):
        """
        Test that update_from_hardpoints_registry correctly updates hardpoint positions.
        """
        upright = Upright.from_hardpoints_and_attachments(
            hardpoint_point_ids, hardpoints_registry, attachments
        )

        # Store original local offset
        original_local_offset = upright.attachment_local_offsets["axle_inboard"].copy()

        # Create updated hardpoints registry (simulate 50mm bump)
        from kinematics.core.geometry import Vector3

        bump = Vector3([0, 0, 50])
        updated_hardpoints = {
            pid: pos + bump for pid, pos in hardpoints_registry.items()
        }

        upright.update_from_hardpoints_registry(updated_hardpoints)

        # Verify hardpoints have been updated
        np.testing.assert_allclose(
            upright.hardpoints.upper_ball_joint.data,
            updated_hardpoints[PointID.UPPER_WISHBONE_OUTBOARD].data,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            upright.hardpoints.lower_ball_joint.data,
            updated_hardpoints[PointID.LOWER_WISHBONE_OUTBOARD].data,
            atol=1e-10,
        )

        # Verify local offsets are preserved
        np.testing.assert_allclose(
            upright.attachment_local_offsets["axle_inboard"].data,
            original_local_offset.data,
            atol=1e-10,
        )

    def test_update_from_hardpoints_registry_without_hardpoint_point_ids_raises(self):
        """
        Test that update_from_hardpoints_registry raises if upright wasn't created with
        hardpoints.
        """
        # Create upright using old-style constructor (no hardpoint_point_ids)
        hardpoints = UprightHardpoints(
            lower_ball_joint=Point3([0, 900, 200]),
            upper_ball_joint=Point3([-25, 750, 500]),
            trackrod_outboard=Point3([150, 800, 275]),
        )
        attachments = UprightAttachments(
            axle_inboard=Point3([-20, 800, 308.426]),
            axle_outboard=Point3([-20, 950, 313.426]),
        )

        upright = Upright.from_global_positions(hardpoints, attachments)

        # Attempting to use hardpoints registry-based update should fail
        dummy_hardpoints = {PointID.LOWER_WISHBONE_OUTBOARD: Point3([0, 0, 0])}

        with pytest.raises(RuntimeError, match="not created with hardpoint references"):
            upright.update_from_hardpoints_registry(dummy_hardpoints)

    def test_hardpoint_point_ids_property_returns_none_for_legacy_upright(self):
        """
        Test that hardpoint_point_ids property returns None for legacy-constructed
        uprights.
        """
        hardpoints = UprightHardpoints(
            lower_ball_joint=Point3([0, 900, 200]),
            upper_ball_joint=Point3([-25, 750, 500]),
            trackrod_outboard=Point3([150, 800, 275]),
        )
        attachments = UprightAttachments(
            axle_inboard=Point3([-20, 800, 308.426]),
            axle_outboard=Point3([-20, 950, 313.426]),
        )

        upright = Upright.from_global_positions(hardpoints, attachments)

        assert upright.hardpoint_point_ids is None

    def test_camber_shim_works_with_hardpoints_upright(
        self, hardpoints_registry, hardpoint_point_ids, attachments
    ):
        """
        Test that camber shim can be applied to hardpoints-constructed upright.
        """
        upright = Upright.from_hardpoints_and_attachments(
            hardpoint_point_ids, hardpoints_registry, attachments
        )

        # Store original hardpoint positions
        original_lower_bj = upright.hardpoints.lower_ball_joint.copy()
        original_upper_bj = upright.hardpoints.upper_ball_joint.copy()

        # Store original attachment position
        original_axle_inboard = upright.get_world_position("axle_inboard").copy()

        # Apply camber shim
        pivot = upright.hardpoints.lower_ball_joint
        rotation_axis = Direction3([1, 0, 0])
        rotation_angle = np.deg2rad(2.0)

        upright.apply_camber_shim(pivot, rotation_axis, rotation_angle)

        # Hardpoints must NOT move
        np.testing.assert_allclose(
            upright.hardpoints.lower_ball_joint.data,
            original_lower_bj.data,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            upright.hardpoints.upper_ball_joint.data,
            original_upper_bj.data,
            atol=1e-10,
        )

        # Attachments MUST move
        new_axle_inboard = upright.get_world_position("axle_inboard")
        assert not np.allclose(
            new_axle_inboard.data, original_axle_inboard.data, atol=1e-6
        )
