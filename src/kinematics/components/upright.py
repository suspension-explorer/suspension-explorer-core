from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.geometry import Direction3, Point3, midpoint
from kinematics.core.rigid_body import LocalCoordinateSystem, RigidBody
from kinematics.core.vector_utils.geometric import rotate_point_about_axis


@dataclass
class UprightHardpoints:
    """
    Hardpoints that define the upright's kinematic reference frame.

    These are the structural attachment points that define how the upright
    moves within the suspension linkage. They do not move when an outboard camber
    shim is modified.

    Attributes:
        upper_ball_joint: Connection to upper wishbone.
        lower_ball_joint: Connection to lower wishbone (also LCS origin).
        trackrod_outboard: Connection to track/tie rod.
    """

    upper_ball_joint: Point3
    lower_ball_joint: Point3
    trackrod_outboard: Point3


@dataclass
class UprightAttachments:
    """
    Attachments that move rigidly with the upright.

    These components are affixed to the upright and move with it during
    suspension travel. They move when camber shim is applied (shim sits between
    UBJ and these components).

    The axle is defined by two points.

    Attributes:
        axle_inboard: Inboard end of the axle.
        axle_outboard: Outboard end of the axle (e.g., wheel center).
    """

    axle_inboard: Point3
    axle_outboard: Point3


class Upright(RigidBody):
    """
    Upright rigid body with hardpoints and attachments.

    The upright's local coordinate system is defined by:
    - Origin: lower_ball_joint.
    - Z axis: From lower_ball_joint to upper_ball_joint.
    - Y axis: In the plane containing tie_rod_pickup, orthogonal to Z.
    - X axis: Cross product of Y and Z (completes right-handed system).

    Workflow:
        1. Initialize with global hardpoint and attachment positions.
        2. Call init_local_frame() to establish LCS and compute local offsets.
        3. During solving, call update_from_hardpoints() with new hardpoint positions.
        4. Query attachment positions with get_world_position().
        5. Apply shims with apply_camber_shim() to rotate attachments.

    Attributes:
        hardpoints: The three hardpoints defining the upright orientation.
        attachments: Components attached to the upright.
    """

    def __init__(
        self,
        hardpoints: UprightHardpoints | None = None,
        attachments: UprightAttachments | None = None,
    ):
        """
        Initialize the upright rigid body.

        Args:
            hardpoints: The hardpoints defining the upright's kinematic frame.
            attachments: The attachments moving rigidly with the upright.
        """
        super().__init__()

        if hardpoints is None:
            hardpoints = UprightHardpoints(
                upper_ball_joint=Point3(np.zeros(3)),
                lower_ball_joint=Point3(np.zeros(3)),
                trackrod_outboard=Point3(np.zeros(3)),
            )

        if attachments is None:
            attachments = UprightAttachments(
                axle_inboard=Point3(np.zeros(3)),
                axle_outboard=Point3(np.zeros(3)),
            )

        self.hardpoints = hardpoints
        self.attachments = attachments

        # Point references used to resolve hardpoints from the central registry.
        self.hardpoint_point_ids: dict[str, PointID] | None = None

    @classmethod
    def from_global_positions(
        cls,
        hardpoints: UprightHardpoints,
        attachments: UprightAttachments,
    ) -> "Upright":
        """
        Create an upright from global (world) coordinates.

        This is the primary construction method. It takes all positions in
        world coordinates (as the user provides them in YAML), establishes
        the local coordinate system, and computes local offsets.

        Args:
            hardpoints: Hardpoint positions in world coordinates.
            attachments: Attachment positions in world coordinates.

        Returns:
            Upright with LCS established and local offsets computed.
        """
        upright = cls(hardpoints=hardpoints, attachments=attachments)
        upright.init_local_frame()
        return upright

    def construct_lcs(self) -> LocalCoordinateSystem:
        """
        Construct the upright's local coordinate system from its hardpoints.

        The LCS is defined by:
        - Origin: lower_ball_joint.
        - Z axis: From lower_ball_joint to upper_ball_joint.
        - Y axis: In plane toward tie_rod_pickup, orthogonal to Z.
        - X axis: Completes right-handed system (X = Y × Z).

        Returns:
            LocalCoordinateSystem for this upright.
        """
        return LocalCoordinateSystem.from_three_points(
            origin=self.hardpoints.lower_ball_joint,
            z_point=self.hardpoints.upper_ball_joint,
            y_reference=self.hardpoints.trackrod_outboard,
        )

    def init_local_frame(self) -> None:
        """
        Initialize the local coordinate frame and compute local offsets for all
        attachments.

        This establishes the current global positions as the design state by:
        1. Constructing the LCS from the hardpoints.
        2. Converting all attachment positions to local coordinates.
        3. Storing these as the design state offsets.

        Call this once at initialization, and again after applying shims to
        update the design state.
        """
        # Establish LCS from hardpoints.
        self.lcs = self.construct_lcs()

        # Compute local offsets for all attachments.
        self.attachment_local_offsets["axle_inboard"] = self.lcs.world_to_local(
            self.attachments.axle_inboard
        )
        self.attachment_local_offsets["axle_outboard"] = self.lcs.world_to_local(
            self.attachments.axle_outboard
        )

    def update_from_hardpoints(self, hardpoints: Mapping[str, Point3]) -> None:
        """
        Update the upright's orientation from new hardpoint positions.

        This is called during solving when the hardpoints move due to
        suspension travel. It reconstructs the LCS and automatically
        updates all attachment positions accordingly.

        Args:
            hardpoints: Dict with ball joint and trackrod keys.
        """
        # Update hardpoint positions.
        self.hardpoints.upper_ball_joint = Point3(hardpoints["upper_ball_joint"])
        self.hardpoints.lower_ball_joint = Point3(hardpoints["lower_ball_joint"])
        self.hardpoints.trackrod_outboard = Point3(hardpoints["trackrod_outboard"])

        # Reconstruct LCS from new hardpoint positions.
        self.lcs = self.construct_lcs()

        # Keep attachment world positions in sync with the updated LCS.
        self.attachments.axle_inboard = self.get_world_position("axle_inboard")
        self.attachments.axle_outboard = self.get_world_position("axle_outboard")

    def apply_camber_shim(
        self,
        pivot_point: Point3,
        rotation_axis: Direction3,
        rotation_angle_rad: float,
    ) -> None:
        """
        Apply a camber shim transformation to the upright.

        Note: this rotates only the attachments, not the hardpoints.
        The shim sits between the UBJ hardpoint and the rest of the upright body,
        so the hub/bearing assembly (attachments) rotate around the LBJ (pivot point),
        while both UBJ and LBJ hardpoints remain fixed.

        The approach:
        1. Get current world positions of attachments.
        2. Rotate them about the specified axis and pivot.
        3. Convert back to local coordinates.
        4. Update the stored local offsets (re-bake).

        Args:
            pivot_point: Pivot point for rotation (typically lower_ball_joint).
            rotation_axis: Unit direction for the rotation axis.
            rotation_angle_rad: Rotation angle in radians.
        """
        # Get current world positions of attachments.
        axle_inboard_world = self.get_world_position("axle_inboard")
        axle_outboard_world = self.get_world_position("axle_outboard")

        # Rotate attachment positions in world space.
        axle_inboard_rotated = rotate_point_about_axis(
            axle_inboard_world, pivot_point, rotation_axis, rotation_angle_rad
        )
        axle_outboard_rotated = rotate_point_about_axis(
            axle_outboard_world, pivot_point, rotation_axis, rotation_angle_rad
        )

        # Update attachment world positions.
        self.attachments.axle_inboard = axle_inboard_rotated
        self.attachments.axle_outboard = axle_outboard_rotated

        # Re-initialize to update local offsets with new positions.
        # The hardpoints haven't changed, but the attachments have rotated
        # relative to them.
        self.init_local_frame()

    def get_axle_vector(self) -> Direction3:
        """
        Get the axle direction vector (normalized).

        Returns:
            Unit direction pointing from axle_inboard to axle_outboard.
        """
        axle_inboard = self.get_world_position("axle_inboard")
        axle_outboard = self.get_world_position("axle_outboard")
        axle_vec = axle_outboard - axle_inboard
        return axle_vec.normalize()

    def get_axle_midpoint(self) -> Point3:
        """
        Get the midpoint of the axle.

        Returns:
            Midpoint between axle_inboard and axle_outboard.
        """
        axle_inboard = self.get_world_position("axle_inboard")
        axle_outboard = self.get_world_position("axle_outboard")
        return midpoint(axle_inboard, axle_outboard)

    # Hardpoints and attachments construction methods.

    @classmethod
    def from_hardpoints_and_attachments(
        cls,
        hardpoint_point_ids: dict[str, PointID],
        hardpoints: dict[PointID, Point3],
        attachments: dict[str, Point3],
    ) -> "Upright":
        """
        Create an Upright from hardpoint references and attachment positions.

        This is the primary constructor for creating uprights from a central
        hardpoints registry. The upright stores PointID references for mounts
        and resolves them from the hardpoints at construction time.

        Args:
            hardpoint_point_ids: Mapping of mount role to PointID
                       {"upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
                        "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
                        "trackrod_outboard": PointID.TRACKROD_OUTBOARD}
            hardpoints: The central hardpoints registry mapping PointID to coordinates
            attachments: Attachment positions in global coordinates at design
                         {"axle_inboard": Point3, "axle_outboard": Point3}

        Returns:
            Upright with LCS established, local offsets computed, and mount IDs stored.

        Raises:
            KeyError: If a required mount_id is not found in hardpoints
            ValueError: If required mounts or attachments are missing
        """
        # Validate required mounts.
        required_mounts = {"upper_ball_joint", "lower_ball_joint", "trackrod_outboard"}
        missing_mounts = required_mounts - set(hardpoint_point_ids.keys())
        if missing_mounts:
            raise ValueError(f"Missing required mounts: {missing_mounts}")

        # Validate required attachments.
        required_attachments = {"axle_inboard", "axle_outboard"}
        missing_attachments = required_attachments - set(attachments.keys())
        if missing_attachments:
            raise ValueError(f"Missing required attachments: {missing_attachments}")

        # Resolve mount positions from hardpoints.
        upright_hardpoints = UprightHardpoints(
            upper_ball_joint=Point3(
                hardpoints[hardpoint_point_ids["upper_ball_joint"]]
            ),
            lower_ball_joint=Point3(
                hardpoints[hardpoint_point_ids["lower_ball_joint"]]
            ),
            trackrod_outboard=Point3(
                hardpoints[hardpoint_point_ids["trackrod_outboard"]]
            ),
        )

        upright_attachments = UprightAttachments(
            axle_inboard=Point3(attachments["axle_inboard"]),
            axle_outboard=Point3(attachments["axle_outboard"]),
        )

        upright = cls(hardpoints=upright_hardpoints, attachments=upright_attachments)
        # Own the mapping so later caller mutation cannot alter the upright topology.
        upright.hardpoint_point_ids = dict(hardpoint_point_ids)
        upright.init_local_frame()
        return upright

    def update_from_hardpoints_registry(
        self, hardpoints: Mapping[PointID, Point3]
    ) -> None:
        """
        Update the upright's orientation from the hardpoints registry.

        This is called during the solver loop when hardpoints move due to
        suspension travel. It reads the new positions from the hardpoints
        registry and reconstructs the LCS accordingly.

        Args:
            hardpoints: The central hardpoints registry with updated positions

        Raises:
            RuntimeError: If upright was not created with hardpoint references
            KeyError: If a mount PointID is not found in hardpoints
        """
        if self.hardpoint_point_ids is None:
            raise RuntimeError(
                "Upright not created with hardpoint references. "
                "Use from_hardpoints_and_attachments() constructor."
            )

        point_ids = self.hardpoint_point_ids  # Local reference for type narrowing.
        new_hardpoints = {
            "upper_ball_joint": hardpoints[point_ids["upper_ball_joint"]],
            "lower_ball_joint": hardpoints[point_ids["lower_ball_joint"]],
            "trackrod_outboard": hardpoints[point_ids["trackrod_outboard"]],
        }
        self.update_from_hardpoints(new_hardpoints)
