"""
Rigid body coordinate transformations for suspension components.

This module provides a hierarchical rigid body system where:
- Hardpoints: Define the kinematic reference frame of the body.
- Attachments: Move rigidly with the body but don't define its orientation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.geometry import Direction3, Point3, Vector3
from kinematics.core.vector_utils.generic import normalize_vector


@dataclass
class LocalCoordinateSystem:
    """
    Local coordinate system defined by origin and three orthonormal axes.

    Attributes:
        origin: Origin point in world coordinates.
        x_axis: Local X axis direction in world coordinates.
        y_axis: Local Y axis direction in world coordinates.
        z_axis: Local Z axis direction in world coordinates.
    """

    origin: Point3
    x_axis: Direction3
    y_axis: Direction3
    z_axis: Direction3

    def world_to_local(self, world_point: Point3) -> Vector3:
        """
        Transform a point from world coordinates to local coordinates.

        The result is a Vector3 representing the offset from the LCS origin
        expressed in local axes.

        Args:
            world_point: Point in world coordinates.

        Returns:
            Offset vector in local coordinates.
        """
        # Translate to origin.
        relative = world_point - self.origin

        # Project onto local axes.
        local_x = relative.dot(self.x_axis)
        local_y = relative.dot(self.y_axis)
        local_z = relative.dot(self.z_axis)

        return Vector3([local_x, local_y, local_z])

    def local_to_world(self, local_offset: Vector3) -> Point3:
        """
        Transform a local offset back to a world-space point.

        Args:
            local_offset: Offset vector in local coordinates.

        Returns:
            Point in world coordinates.
        """
        # Construct world position from local coordinates.
        world_point = (
            self.origin
            + self.x_axis * local_offset[0]
            + self.y_axis * local_offset[1]
            + self.z_axis * local_offset[2]
        )

        return world_point

    @classmethod
    def from_three_points(
        cls,
        origin: Point3,
        z_point: Point3,
        y_reference: Point3,
    ) -> "LocalCoordinateSystem":
        """
        Construct a local coordinate system from three points.

        The construction follows this logic:
        1. Origin is at the specified origin point.
        2. Z axis points from origin to z_point.
        3. Y axis is in the plane defined by origin, z_point, and y_reference,
           orthogonal to Z.
        4. X axis completes the right-handed system (X = Y x Z).

        Args:
            origin: Origin point.
            z_point: Point defining the Z axis direction.
            y_reference: Point used to define the Y axis (with orthogonalisation).

        Returns:
            LocalCoordinateSystem constructed from the three points.

        Raises:
            ValueError: If points are collinear or invalid.
        """
        # Primary axis (Z): from origin to z_point.
        z_vec = z_point - origin
        z_axis = normalize_vector(z_vec)

        # Secondary axis (Y): project y_reference into plane perpendicular to Z.
        y_vec = y_reference - origin

        # Remove component along Z to ensure orthogonality.
        y_along_z = y_vec.dot(z_axis)
        y_perpendicular = y_vec - z_axis * y_along_z

        y_magnitude = y_perpendicular.norm()
        if y_magnitude < EPS_GEOMETRIC:
            raise ValueError(
                "Y reference is collinear with Z axis - cannot define unique plane."
            )

        y_axis = y_perpendicular.normalize()

        # Tertiary axis (X): complete right-handed system.
        x_axis = y_axis.cross(z_axis).normalize()

        return cls(
            origin=origin,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
        )


class RigidBody(ABC):
    """
    Abstract base class for rigid bodies in suspension kinematics.

    A rigid body is defined by hardpoints (which define the kinematic reference
    frame) and attachments (which move with the body but don't define it).

    The body maintains local offsets for attachments that are transformed
    to world coordinates based on the current hardpoint positions.

    Subclasses must implement:
    - construct_lcs(): Define how to build the LCS from hardpoints.
    - init_local_frame(): Define how to compute attachment local offsets.

    Attributes:
        attachment_local_offsets: Local coordinates of attachments (computed at init).
        lcs: Current local coordinate system (updated from hardpoints).
    """

    def __init__(self):
        """
        Initialize the rigid body with empty attachment dictionary.
        """
        self.attachment_local_offsets: dict[str, Vector3] = {}
        self.lcs: LocalCoordinateSystem | None = None

    def get_lcs(self) -> LocalCoordinateSystem:
        """
        Get the current local coordinate system.

        Returns:
            The current LCS.

        Raises:
            RuntimeError: If LCS has not been initialized.
        """
        if self.lcs is None:
            raise RuntimeError(
                f"{self.__class__.__name__} LCS not initialized. "
                f"Call init_local_frame() first."
            )
        return self.lcs

    def get_world_position(self, attachment_name: str) -> Point3:
        """
        Get the world position of an attachment.

        Args:
            attachment_name: Name of the attachment.

        Returns:
            World position of the attachment in current body configuration.

        Raises:
            KeyError: If attachment_name is not found.
        """
        if attachment_name not in self.attachment_local_offsets:
            raise KeyError(
                f"Attachment '{attachment_name}' not found in {self.__class__.__name__}"
            )

        local_offset = self.attachment_local_offsets[attachment_name]
        return self.get_lcs().local_to_world(local_offset)

    def add_attachment(self, name: str, world_position: Point3) -> None:
        """
        Add an attachment at a specific world position.

        The world position is immediately converted to local coordinates
        using the current LCS. This allows arbitrary attachment points to be
        added to any rigid body.

        Args:
            name: Name of the attachment.
            world_position: World coordinates of the attachment.

        Raises:
            RuntimeError: If LCS has not been initialized.
        """
        local_offset = self.get_lcs().world_to_local(world_position)
        self.attachment_local_offsets[name] = local_offset

    def set_attachment_local_offset(
        self, attachment_name: str, local_offset: Vector3
    ) -> None:
        """
        Directly set the local offset for an attachment.

        This is used during initialisation or when applying transformations
        like camber shims that modify the attachment positions relative to
        the hardpoints.

        Args:
            attachment_name: Name of the attachment.
            local_offset: Local coordinates of the attachment.
        """
        self.attachment_local_offsets[attachment_name] = local_offset

    @abstractmethod
    def construct_lcs(self) -> LocalCoordinateSystem:
        """
        Construct the local coordinate system from hardpoints.

        This method must be implemented by subclasses to define how their
        specific hardpoint configuration determines the body's orientation.

        Returns:
            LocalCoordinateSystem constructed from the body's hardpoints.

        Example:
            For an Upright:
            - Origin at lower_ball_joint.
            - Z axis toward upper_ball_joint.
            - Y axis in plane toward tie_rod_pickup.
        """
        raise NotImplementedError

    @abstractmethod
    def init_local_frame(self) -> None:
        """
        Initialize the local coordinate frame and compute local offsets for all
        attachments.

        This method must be implemented by subclasses to:
        1. Call construct_lcs() to build the LCS.
        2. Convert all attachment world positions to local coordinates.
        3. Store them in attachment_local_offsets.

        This is called once at initialization and again after transformations
        (like camber shims) to update the design state.
        """
        raise NotImplementedError
