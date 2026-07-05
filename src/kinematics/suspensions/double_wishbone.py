"""
Double wishbone suspension implementation.

This module defines the DoubleWishboneSuspension class which combines topology
definition, geometry storage, and kinematic behavior in a single unified class.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
    PointOnLineConstraint,
    ScalarTripleProductConstraint,
)
from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID, ShimType
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.types import WorldAxisSystem
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    compute_scalar_triple_product,
    compute_vector_vector_angle,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
    rotate_point_about_axis,
)
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_contact_patch_center,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.config.shims import solve_camber_shim_assembly

if TYPE_CHECKING:
    from kinematics.visualization.main import LinkVisualization


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

    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.PUSHROD_INBOARD,
            PointID.PUSHROD_OUTBOARD,
            PointID.ROCKER_AXIS_FRONT,
            PointID.ROCKER_AXIS_REAR,
            PointID.ROCKER_DROPLINK,
        }
    )

    # The pushrod/rocker group is all-or-nothing: either the pushrod link and
    # its inboard rocker (pivoting about the two chassis-fixed axis points) are
    # all present, or none of them are. ROCKER_DROPLINK is an optional extra
    # point on the same rocker body, valid only when the group is present.
    ROCKER_GROUP: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
            PointID.ROCKER_AXIS_FRONT,
            PointID.ROCKER_AXIS_REAR,
        }
    )

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

    @property
    def has_rocker(self) -> bool:
        """True when the pushrod/rocker group is present in the hardpoints."""
        return self.ROCKER_GROUP <= set(self.hardpoints)

    @property
    def has_rocker_droplink(self) -> bool:
        """True when ROCKER_DROPLINK is present (implies the rocker group)."""
        return self.has_rocker and PointID.ROCKER_DROPLINK in self.hardpoints

    def validate_hardpoints(self) -> None:
        """
        Validate required hardpoints plus the optional pushrod/rocker group.

        The rocker group (:attr:`ROCKER_GROUP`) is all-or-nothing. When present:

        - the two rocker-axis points must be distinct and lie parallel to the XZ
          plane (equal Y within ``EPS_GEOMETRIC``), since the rocker/torsion-bar
          pivot is authored parallel to XZ;
        - ``PUSHROD_INBOARD`` (and ``ROCKER_DROPLINK`` when given) must be off the
          axis (non-zero perpendicular distance), else they cannot trace a circle.

        ``ROCKER_DROPLINK`` may only appear together with the full group.
        """
        super().validate_hardpoints()

        present = set(self.hardpoints)
        group_present = self.ROCKER_GROUP & present
        if group_present and group_present != self.ROCKER_GROUP:
            missing = sorted(p.name for p in self.ROCKER_GROUP - present)
            raise ValueError(
                "Incomplete pushrod/rocker group: the points "
                f"{sorted(p.name for p in self.ROCKER_GROUP)} are all-or-nothing; "
                f"missing {missing}."
            )

        if PointID.ROCKER_DROPLINK in present and not self.has_rocker:
            raise ValueError(
                "ROCKER_DROPLINK requires the full pushrod/rocker group "
                f"({sorted(p.name for p in self.ROCKER_GROUP)})."
            )

        if not self.has_rocker:
            return

        axis_front = self.hardpoints[PointID.ROCKER_AXIS_FRONT]
        axis_rear = self.hardpoints[PointID.ROCKER_AXIS_REAR]

        if compute_point_point_distance(axis_front, axis_rear) <= EPS_GEOMETRIC:
            raise ValueError(
                "ROCKER_AXIS_FRONT and ROCKER_AXIS_REAR must be distinct points."
            )

        y_extent = abs(float(axis_front[Axis.Y]) - float(axis_rear[Axis.Y]))
        if y_extent > EPS_GEOMETRIC:
            raise ValueError(
                "Rocker axis must be parallel to the XZ plane (zero Y-extent): "
                f"|Y(ROCKER_AXIS_FRONT) - Y(ROCKER_AXIS_REAR)| = {y_extent} "
                f"exceeds {EPS_GEOMETRIC}."
            )

        axis_direction = (axis_rear - axis_front).normalize()
        off_axis_points = [PointID.PUSHROD_INBOARD]
        if PointID.ROCKER_DROPLINK in present:
            off_axis_points.append(PointID.ROCKER_DROPLINK)
        for pid in off_axis_points:
            radius = compute_point_to_line_distance(
                self.hardpoints[pid], axis_front, axis_direction
            )
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{pid.name} lies on the rocker axis (zero radius); it must "
                    "be off-axis to trace a rocker circle."
                )

    def free_points(self) -> Sequence[PointID]:
        """
        Points that move during solving.

        The base wishbone free points plus, when the pushrod/rocker group is
        present, the pushrod ends and (when given) the rocker droplink. The
        rocker-axis points are chassis-fixed and stay out of this list.
        """
        points: list[PointID] = list(self.FREE_POINTS)
        if self.has_rocker:
            points.append(PointID.PUSHROD_OUTBOARD)
            points.append(PointID.PUSHROD_INBOARD)
            if PointID.ROCKER_DROPLINK in self.hardpoints:
                points.append(PointID.ROCKER_DROPLINK)
        return points

    def output_points(self) -> tuple[PointID, ...]:
        """Static output points plus present pushrod/rocker points, in order."""
        extra: list[PointID] = []
        if self.has_rocker:
            extra.append(PointID.PUSHROD_OUTBOARD)
            extra.append(PointID.PUSHROD_INBOARD)
            if PointID.ROCKER_DROPLINK in self.hardpoints:
                extra.append(PointID.ROCKER_DROPLINK)
        return self.OUTPUT_POINTS + tuple(extra)

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

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set(self.free_points()),
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

        # Pushrod / rocker group (F1-style inboard actuation), all expressed as
        # distance constraints so no new Jacobian codegen is required.
        if self.has_rocker:
            constraints.extend(self._rocker_constraints(initial_state))

        return constraints

    def _rocker_constraints(self, initial_state: SuspensionState) -> list[Constraint]:
        """
        Distance constraints for the pushrod / rocker group.

        - ``PUSHROD_OUTBOARD`` is rigid to the upright: 4 distances to the two
          outboard ball joints and the two axle points.
        - the pushrod is a fixed-length link ``PUSHROD_OUTBOARD -> PUSHROD_INBOARD``.
        - ``PUSHROD_INBOARD`` rides the rocker circle: 2 distances to the two
          chassis-fixed rocker-axis points.
        - when ``ROCKER_DROPLINK`` is present it also rides the rocker circle
          (2 distances) and is held rigid to ``PUSHROD_INBOARD`` (1 distance),
          so the whole rocker body rotates as one about its axis.
        """
        pos = initial_state.positions
        constraints: list[Constraint] = []

        def add_distance(p1: PointID, p2: PointID) -> None:
            constraints.append(
                DistanceConstraint(
                    p1, p2, compute_point_point_distance(pos[p1], pos[p2])
                )
            )

        # Pushrod outboard rigidly attached to the upright body.
        for anchor in (
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ):
            add_distance(PointID.PUSHROD_OUTBOARD, anchor)

        # Pushrod link length.
        add_distance(PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD)

        # Pushrod inboard on the rocker circle.
        add_distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_FRONT)
        add_distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_REAR)

        # Rocker droplink: also on the rocker circle, and rigid to the pushrod
        # inboard so the rocker body is a single rotating link.
        if PointID.ROCKER_DROPLINK in self.hardpoints:
            add_distance(PointID.ROCKER_DROPLINK, PointID.ROCKER_AXIS_FRONT)
            add_distance(PointID.ROCKER_DROPLINK, PointID.ROCKER_AXIS_REAR)
            add_distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_DROPLINK)

            # Chirality pin: the droplink pickup is fixed by only distances (two
            # to the axis points, one chord to the pushrod pickup), which admit a
            # mirror solution -- reflecting the droplink through the plane of the
            # axis and the pushrod pickup satisfies every distance but inverts the
            # rigid rocker body. Hold the signed scalar triple product of
            # (axis_front, axis_rear, pushrod_inboard, rocker_droplink) at its
            # design value to select the correct handedness. Skipped for a
            # degenerate planar rocker (all four points coplanar at design), where
            # the two branches coincide and the triple product carries no sign.
            design_triple = compute_scalar_triple_product(
                pos[PointID.ROCKER_AXIS_REAR] - pos[PointID.ROCKER_AXIS_FRONT],
                pos[PointID.PUSHROD_INBOARD] - pos[PointID.ROCKER_AXIS_FRONT],
                pos[PointID.ROCKER_DROPLINK] - pos[PointID.ROCKER_AXIS_FRONT],
            )
            if abs(design_triple) >= 1e-6:
                constraints.append(
                    ScalarTripleProductConstraint(
                        PointID.ROCKER_AXIS_FRONT,
                        PointID.ROCKER_AXIS_REAR,
                        PointID.PUSHROD_INBOARD,
                        PointID.ROCKER_DROPLINK,
                        target_volume=design_triple,
                        scale=max(abs(design_triple), 1.0),
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

    def get_visualization_links(self) -> list[LinkVisualization]:
        """Visualization links for 3D rendering."""
        from kinematics.visualization.main import LinkVisualization

        links = [
            LinkVisualization(
                points=[
                    PointID.UPPER_WISHBONE_INBOARD_FRONT,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.UPPER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Upper Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.LOWER_WISHBONE_INBOARD_FRONT,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Lower Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.TRACKROD_OUTBOARD,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.TRACKROD_OUTBOARD,
                ],
                color="slategrey",
                label="Upright",
            ),
            LinkVisualization(
                points=[PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD],
                color="darkorange",
                label="Track Rod",
            ),
            LinkVisualization(
                points=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
                color="forestgreen",
                label="Axle",
            ),
            LinkVisualization(
                points=[PointID.CONTACT_PATCH_CENTER],
                color="black",
                label="Contact Patch",
                linewidth=0.0,
                marker="o",
                markersize=15.0,
            ),
        ]

        if self.has_rocker:
            links.append(
                LinkVisualization(
                    points=[PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD],
                    color="crimson",
                    label="Pushrod",
                )
            )
            # Rocker body: axis-front -> pushrod inboard -> (droplink) ->
            # axis-rear, drawing the triangular rocker where the droplink exists.
            rocker_points = [PointID.ROCKER_AXIS_FRONT, PointID.PUSHROD_INBOARD]
            if PointID.ROCKER_DROPLINK in self.hardpoints:
                rocker_points.append(PointID.ROCKER_DROPLINK)
            rocker_points.append(PointID.ROCKER_AXIS_REAR)
            links.append(
                LinkVisualization(
                    points=rocker_points,
                    color="mediumvioletred",
                    label="Rocker",
                )
            )

        return links

    def apply_camber_shim(self, positions: dict[PointID, Point3]) -> None:
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
        assembly_solution = solve_camber_shim_assembly(
            positions=positions,
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
