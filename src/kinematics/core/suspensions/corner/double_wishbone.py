"""
Double-wishbone corner suspension implementation.

This module defines the DoubleWishboneSuspension class which combines topology
definition, geometry storage, and kinematic behavior in a single unified class.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Sequence, cast

from kinematics.core.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
    PointOnLineConstraint,
)
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    UprightElement,
    WheelElement,
)
from kinematics.core.points.derived.definitions import (
    get_axle_midpoint,
    get_contact_patch_center,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.enums import Axis, PointID, ShimType
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
    rotate_point_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.config.shims import solve_camber_shim_assembly
from kinematics.core.targeting import WorldAxisSystem


@dataclass
class DoubleWishboneSuspension(Suspension):
    """
    Double wishbone suspension with all topology and behavior in one class.

    This class:
    - Defines valid points as class attributes (replacing SuspensionTemplate)
    - Stores hardpoints and config as instance data (replacing TemplateGeometry)
    - Implements constraints, visualization, and solver interface (replacing provider)
    """

    TYPE_KEY: ClassVar[str] = "double_wishbone"
    ALIASES: ClassVar[frozenset[str]] = frozenset(
        {"double_wishbone_front", "double_wishbone_rear"}
    )
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            PointID.UPPER_WISHBONE_INBOARD_REAR,
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.TRACKROD_INBOARD,
            PointID.TRACKROD_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )

    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()

    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset(
        {ShimType.OUTBOARD_CAMBER}
    )

    # Points included in solver output (CSV/Parquet), in column order.
    # Hardpoints first, then derived points.
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_INBOARD_REAR,
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.TRACKROD_INBOARD,
        PointID.TRACKROD_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.AXLE_MIDPOINT,
        PointID.WHEEL_CENTER,
        PointID.WHEEL_INBOARD,
        PointID.WHEEL_OUTBOARD,
        PointID.CONTACT_PATCH_CENTER,
    )

    # Config names for points that should rotate with the upright body when a
    # split camber shim is solved.
    UPRIGHT_MOUNTED_POINT_IDS: ClassVar[dict[str, PointID]] = {
        "axle_inboard": PointID.AXLE_INBOARD,
        "axle_outboard": PointID.AXLE_OUTBOARD,
        "pushrod_outboard": PointID.PUSHROD_OUTBOARD,
        "trackrod_outboard": PointID.TRACKROD_OUTBOARD,
    }

    # Free points that move during solving.
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.TRACKROD_OUTBOARD,
        PointID.TRACKROD_INBOARD,
    )

    def free_points(self) -> Sequence[PointID]:
        """Points that move during solving."""
        return self.FREE_POINTS

    def initial_state(self) -> SuspensionState:
        """Build initial state from hardpoints, applying shims if configured."""
        if self._initial_state is not None:
            return self._initial_state

        positions = self.get_hardpoints_copy()

        # Get camber shim point positions.

        # Apply camber shim if configured.
        if self.config is not None and self.config.camber_shim is not None:
            self.apply_camber_shim(positions)

        # Compute derived points.
        derived_spec = self.derived_spec()
        derived_manager = DerivedPointsManager(derived_spec)
        derived_manager.update_in_place(positions)

        # get_hardpoints_copy widens keys to PointKey, so key the state on
        # PointKey too even though a corner only ever holds PointID keys.
        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set[PointKey](self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build geometric constraints for double wishbone."""
        initial_state = self.initial_state()
        constraints: list[Constraint] = []

        # Distance constraints (link lengths).
        length_pairs = [
            (PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.UPPER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
        ]

        for p1, p2 in length_pairs:
            target_distance = compute_point_point_distance(
                initial_state.positions[p1], initial_state.positions[p2]
            )
            constraints.append(DistanceConstraint(p1, p2, target_distance))

        # Angle constraint for upright rigidity.
        v1 = (
            initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]
            - initial_state.positions[PointID.UPPER_WISHBONE_OUTBOARD]
        )
        v2 = (
            initial_state.positions[PointID.AXLE_OUTBOARD]
            - initial_state.positions[PointID.AXLE_INBOARD]
        )
        target_angle = compute_vector_vector_angle(v1, v2)

        constraints.append(
            AngleConstraint(
                v1_start=PointID.UPPER_WISHBONE_OUTBOARD,
                v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
                v2_start=PointID.AXLE_INBOARD,
                v2_end=PointID.AXLE_OUTBOARD,
                target_angle=target_angle,
            )
        )

        # Point-on-line constraint for rack travel.
        constraints.append(
            PointOnLineConstraint(
                point_id=PointID.TRACKROD_INBOARD,
                line_point=initial_state.positions[PointID.TRACKROD_INBOARD],
                line_direction=WorldAxisSystem.Y,
            )
        )

        return constraints

    def derived_spec(self) -> DerivedPointsSpec:
        """Specification for derived points (wheel center, contact patch, etc.)."""
        if self.config is None:
            raise ValueError("Cannot compute derived spec without config")

        wheel_cfg = self.config.wheel
        tire_radius = wheel_cfg.tire.nominal_radius

        functions = {
            PointID.AXLE_MIDPOINT: get_axle_midpoint,
            PointID.WHEEL_CENTER: partial(
                get_wheel_center, wheel_offset=wheel_cfg.offset
            ),
            PointID.WHEEL_INBOARD: partial(
                get_wheel_inboard, wheel_width=wheel_cfg.tire.section_width
            ),
            PointID.WHEEL_OUTBOARD: partial(
                get_wheel_outboard, wheel_width=wheel_cfg.tire.section_width
            ),
            PointID.CONTACT_PATCH_CENTER: partial(
                get_contact_patch_center, tire_radius=tire_radius
            ),
        }

        dependencies = {
            PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.CONTACT_PATCH_CENTER: {
                PointID.WHEEL_CENTER,
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
            },
        }

        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """
        Compute side view instant center from wishbone planes.

        The SVIC is found by intersecting the 3D instant axis with the
        vertical plane that passes through the wheel center's Y position.
        Returns None when the instant axis is undefined or runs parallel
        to that side-view plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            raise

        wheel_center_y = float(state.positions[PointID.WHEEL_CENTER][1])

        if instant_axis is None:
            return None

        axis_point, axis_direction = instant_axis
        svic = intersect_line_with_vertical_plane(
            axis_point, axis_direction, wheel_center_y
        )

        return svic

    def compute_instant_axis(
        self, state: SuspensionState
    ) -> tuple[Point3, Direction3] | None:
        """Compute 3D instant axis from wishbone planes intersection."""
        upper_front = state.positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
        upper_rear = state.positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
        upper_outboard = state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

        lower_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        lower_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        lower_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        upper_plane = plane_from_three_points(upper_front, upper_rear, upper_outboard)
        lower_plane = plane_from_three_points(lower_front, lower_rear, lower_outboard)

        if upper_plane is None or lower_plane is None:
            raise ValueError(
                "Degenerate wishbone geometry. Cannot compute instant axis."
            )

        return intersect_two_planes(
            n1=upper_plane[0],
            d1=upper_plane[1],
            n2=lower_plane[0],
            d2=lower_plane[1],
        )

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """
        Compute front view instant center from wishbone planes.

        The FVIC is found by intersecting the 3D instant axis (the line
        where the upper and lower wishbone planes meet) with the front-view
        plane at the wheel center's X station. Using the wheel station keeps
        the front-view geometry invariant to rigid longitudinal translations
        of the entire corner. Returns None if the links are parallel or the
        instant axis runs parallel to that front-view plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            raise

        if instant_axis is None:
            return None

        axis_point, axis_direction = instant_axis
        wheel_center_x = float(state.positions[PointID.WHEEL_CENTER][Axis.X])

        return intersect_line_with_axis_aligned_plane(
            axis_point, axis_direction, Axis.X, wheel_center_x
        )

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the physical elements in this corner."""
        return (
            RigidLinkElement(
                label="Upper Wishbone Front Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.UPPER_WISHBONE_INBOARD_FRONT,
                point_b=PointID.UPPER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Upper Wishbone Rear Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.UPPER_WISHBONE_INBOARD_REAR,
                point_b=PointID.UPPER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Lower Wishbone Front Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_FRONT,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            RigidLinkElement(
                label="Lower Wishbone Rear Leg",
                type=ElementType.WISHBONE,
                point_a=PointID.LOWER_WISHBONE_INBOARD_REAR,
                point_b=PointID.LOWER_WISHBONE_OUTBOARD,
            ),
            UprightElement(
                label="Upright",
                hardpoints=(
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.TRACKROD_OUTBOARD,
                ),
                attachments=(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
                segments=(
                    (PointID.TRACKROD_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
                    (
                        PointID.UPPER_WISHBONE_OUTBOARD,
                        PointID.LOWER_WISHBONE_OUTBOARD,
                    ),
                    (PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
                ),
            ),
            RigidLinkElement(
                label="Track Rod",
                type=ElementType.TRACK_ROD,
                point_a=PointID.TRACKROD_INBOARD,
                point_b=PointID.TRACKROD_OUTBOARD,
            ),
            RigidLinkElement(
                label="Axle",
                type=ElementType.AXLE,
                point_a=PointID.AXLE_INBOARD,
                point_b=PointID.AXLE_OUTBOARD,
            ),
            WheelElement(
                label="Wheel",
                center=PointID.WHEEL_CENTER,
                inboard=PointID.WHEEL_INBOARD,
                outboard=PointID.WHEEL_OUTBOARD,
                axle_inboard=PointID.AXLE_INBOARD,
                axle_outboard=PointID.AXLE_OUTBOARD,
                contact_patch=PointID.CONTACT_PATCH_CENTER,
            ),
        )

    def apply_camber_shim(self, positions: dict[PointKey, Point3]) -> None:
        """
        Apply camber shim transformation to the suspension geometry.

        Solves the local split-body shim assembly to find how the camber block
        and upright body rotate when the shim thickness changes. Then:
        1. Writes the solved UBJ position back (it moves along the upper wishbone arc).
        2. Rotates configured upright-mounted points about the fixed LBJ using the
           solved upright-body rotation.
        3. Leaves all chassis-mounted points unchanged.
        """
        if self.config is None or self.config.camber_shim is None:
            return

        shim_config = self.config.camber_shim

        # Shim face geometry is read directly from shim_config by the solver,
        # so the positions dict only needs the kinematic hardpoints.
        # A corner keys strictly on PointID; the solver only reads hardpoints.
        assembly_solution = solve_camber_shim_assembly(
            positions=cast("dict[PointID, Point3]", positions),
            shim_config=shim_config,
        )

        # Write the solved UBJ position back. The upper wishbone arc constraint
        # means UBJ may shift slightly to accommodate the new shim thickness.
        positions[PointID.UPPER_WISHBONE_OUTBOARD] = Point3(
            assembly_solution.ubj_position
        )

        # Rotate each configured upright-mounted point about LBJ using the solved
        # upright-body rotation axis and angle.
        if assembly_solution.upright_body_rot_angle_rad > EPS_GEOMETRIC:
            lbj = positions[PointID.LOWER_WISHBONE_OUTBOARD]
            rot_axis = Direction3(assembly_solution.upright_body_rot_axis)
            for point_name in self.config.upright_mounted_points:
                point_id = self.UPRIGHT_MOUNTED_POINT_IDS.get(point_name)
                if point_id is not None and point_id in positions:
                    positions[point_id] = rotate_point_about_axis(
                        positions[point_id],
                        lbj,
                        rot_axis,
                        assembly_solution.upright_body_rot_angle_rad,
                    )
