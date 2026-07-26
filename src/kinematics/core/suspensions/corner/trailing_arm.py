"""Unsteered semi-trailing-arm suspension with coilover or torsion-bar springing.

The two fixed arm mounts define an oblique, horizontal rotation axis. The
entire outboard wheel carrier is constrained rigidly to one arm point, so bump
and droop rotate the hub about that axis and naturally produce semi-trailing-arm
camber and toe change.

The torsion-bar variant models the Porsche 944-style arrangement: pivot A lies
on a transverse torsion-bar axis, and arm motion about that authored axis drives
bar twist. The physical reaction plate is part of the load path, not a separate
kinematic pickup. A separate damper runs from the chassis to the carrier.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from math import degrees
from typing import TYPE_CHECKING, ClassVar, Sequence

import numpy as np

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    TorsionElement,
    UprightElement,
    VariableLengthLinkElement,
    WheelElement,
)
from kinematics.core.enums import (
    Axis,
    CornerSpringType,
    PointID,
    Scope,
    SuspensionType,
)
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    DualPositions,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.registry import MetricKind, MetricSpec
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.definitions import build_wheel_derived_spec
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
    PositionValue,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.dual import DualScalar, DualVec3
from kinematics.core.primitives.dual import dot as dual_dot
from kinematics.core.primitives.geometry import Direction3, Point3, extract_array
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_vertical_plane,
    signed_angle_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.base import CornerSuspension

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow


TORSION_BAR_TWIST_SPEC = MetricSpec(
    "torsion_bar_twist",
    "Torsion Bar Twist",
    MetricUnit.DEG,
    MetricKind.STATE,
    scope=Scope.CORNER,
    component="torsion_bar",
)


def _rotate_arm_attached_point(
    positions: dict[PointKey, PositionValue],
    *,
    axis_direction: np.ndarray,
    moving_axis_foot: np.ndarray,
    moving_design_radial: np.ndarray,
    target_axis_foot: np.ndarray,
    target_design_radial: np.ndarray,
) -> PositionValue:
    """Rotate one authored carrier point with the solved trailing-arm angle."""
    moving = positions[PointID.TRAILING_ARM_OUTBOARD]
    moving_radius_sq = float(np.dot(moving_design_radial, moving_design_radial))
    tangent = np.cross(axis_direction, moving_design_radial)
    target_tangent = np.cross(axis_direction, target_design_radial)

    if isinstance(moving, DualVec3):
        current_radial = moving - moving_axis_foot
        cosine = dual_dot(current_radial, moving_design_radial) / moving_radius_sq
        sine = dual_dot(current_radial, tangent) / moving_radius_sq
        return (
            target_axis_foot
            + cosine * target_design_radial
            + sine * target_tangent
        )

    current_radial = extract_array(moving) - moving_axis_foot
    cosine = float(np.dot(current_radial, moving_design_radial)) / moving_radius_sq
    sine = float(np.dot(current_radial, tangent)) / moving_radius_sq
    return Point3(
        target_axis_foot
        + cosine * target_design_radial
        + sine * target_tangent
    )


@dataclass
class TrailingArmSuspension(CornerSuspension):
    """One unsteered wheel carrier rotating about an oblique arm pivot axis."""

    TYPE_KEY: ClassVar[SuspensionType] = SuspensionType.TRAILING_ARM
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.TRAILING_ARM_PIVOT_A,
            PointID.TRAILING_ARM_PIVOT_B,
            PointID.TRAILING_ARM_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )
    COILOVER_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {PointID.STRUT_TOP, PointID.STRUT_BOTTOM}
    )
    TORSION_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.TORSION_BAR_AXIS_A,
            PointID.TORSION_BAR_AXIS_B,
            PointID.STRUT_TOP,
            PointID.STRUT_BOTTOM,
        }
    )
    LOCATING_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.TRAILING_ARM_PIVOT_A,
        PointID.TRAILING_ARM_PIVOT_B,
        PointID.TRAILING_ARM_OUTBOARD,
    )
    WHEEL_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.AXLE_MIDPOINT,
        PointID.WHEEL_CENTER,
        PointID.WHEEL_INBOARD,
        PointID.WHEEL_OUTBOARD,
        PointID.WHEEL_PLANE_ROAD_TANGENT,
    )
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.TRAILING_ARM_OUTBOARD,
    )

    spring_type: CornerSpringType = CornerSpringType.COILOVER

    def __post_init__(self) -> None:
        """Validate the selected spring and the semi-trailing-arm geometry."""
        if self.config is None:
            raise ValueError("Semi-trailing arm suspension requires configuration")
        if self.config.steering.type.value != "none":
            raise ValueError("Semi-trailing arm suspension is unsteered")
        if self.spring_type not in (
            CornerSpringType.COILOVER,
            CornerSpringType.TORSION_BAR,
        ):
            raise ValueError(
                "Semi-trailing arm spring must be 'coilover' or 'torsion_bar'"
            )
        super().__post_init__()

    def required_points(self) -> frozenset[PointID]:
        """Return locating hardpoints plus the selected spring hardware."""
        if self.spring_type is CornerSpringType.COILOVER:
            return self.REQUIRED_POINTS | self.COILOVER_POINTS
        return self.REQUIRED_POINTS | self.TORSION_POINTS

    def validate_hardpoints(self) -> None:
        """Require an oblique arm pivot, rearward arm, and valid spring hardware."""
        super().validate_hardpoints()
        pivot_a = self.hardpoints[PointID.TRAILING_ARM_PIVOT_A]
        pivot_b = self.hardpoints[PointID.TRAILING_ARM_PIVOT_B]
        if (
            abs(float(pivot_a[Axis.X] - pivot_b[Axis.X])) <= EPS_GEOMETRIC
            or abs(float(pivot_a[Axis.Z] - pivot_b[Axis.Z])) > EPS_GEOMETRIC
            or abs(float(pivot_a[Axis.Y] - pivot_b[Axis.Y])) <= EPS_GEOMETRIC
        ):
            raise ValueError(
                "TRAILING_ARM_PIVOT_A/B must define a horizontal, oblique "
                "semi-trailing-arm axis (different X and Y, equal Z)."
            )
        arm_x = float(self.hardpoints[PointID.TRAILING_ARM_OUTBOARD][Axis.X])
        rearmost_pivot_x = min(float(pivot_a[Axis.X]), float(pivot_b[Axis.X]))
        if arm_x >= rearmost_pivot_x - EPS_GEOMETRIC:
            raise ValueError(
                "TRAILING_ARM_OUTBOARD must lie rearward of the pivot axis "
                "(X less than both pivot-mount X values)."
            )
        pivot_direction = (pivot_b - pivot_a).normalize()
        if (
            compute_point_to_line_distance(
                self.hardpoints[PointID.TRAILING_ARM_OUTBOARD],
                pivot_a,
                pivot_direction,
            )
            <= EPS_GEOMETRIC
        ):
            raise ValueError(
                "TRAILING_ARM_OUTBOARD must not lie on the trailing-arm pivot axis."
            )
        if self.spring_type is CornerSpringType.TORSION_BAR:
            axis_a = self.hardpoints[PointID.TORSION_BAR_AXIS_A]
            axis_b = self.hardpoints[PointID.TORSION_BAR_AXIS_B]
            if (
                abs(float(axis_a[Axis.X] - axis_b[Axis.X])) > EPS_GEOMETRIC
                or abs(float(axis_a[Axis.Z] - axis_b[Axis.Z])) > EPS_GEOMETRIC
                or abs(float(axis_a[Axis.Y] - axis_b[Axis.Y])) <= EPS_GEOMETRIC
            ):
                raise ValueError(
                    "TORSION_BAR_AXIS_A/B must be distinct and define an axis "
                    "parallel to vehicle Y (equal X and Z)."
                )
            axis_direction = (axis_b - axis_a).normalize()
            if (
                compute_point_to_line_distance(
                    pivot_a,
                    axis_a,
                    axis_direction,
                )
                > EPS_GEOMETRIC
            ):
                raise ValueError(
                    "TRAILING_ARM_PIVOT_A must lie on TORSION_BAR_AXIS_A/B."
                )

    def free_points(self) -> Sequence[PointID]:
        """Return the moving arm point; all carrier pickups derive from it."""
        return self.FREE_POINTS

    def output_points(self) -> tuple[PointKey, ...]:
        """Return locating, wheel, and selected spring points for export."""
        if self.spring_type is CornerSpringType.COILOVER:
            spring_points = (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)
        else:
            spring_points = (
                PointID.TORSION_BAR_AXIS_A,
                PointID.TORSION_BAR_AXIS_B,
                PointID.STRUT_TOP,
                PointID.STRUT_BOTTOM,
            )
        return (*self.LOCATING_OUTPUT_POINTS, *self.WHEEL_OUTPUT_POINTS, *spring_points)

    def steering_axis_points(self) -> tuple[PointID, PointID]:
        """Expose a stable carrier reference line for generic angle metrics.

        The architecture does not steer; this is not a physical kingpin axis.
        It merely lets the shared metric catalog evaluate its conventional
        alignment columns for the rigid carrier.
        """
        return (PointID.TRAILING_ARM_OUTBOARD, PointID.AXLE_INBOARD)

    def rack_attachment_point(self) -> PointID | None:
        """Semi-trailing arms are intentionally unsteered."""
        return None

    def damper_points(self) -> tuple[PointKey, PointKey] | None:
        """Return the chassis-to-arm damper fitted with either spring type."""
        return (PointID.STRUT_TOP, PointID.STRUT_BOTTOM)

    def initial_state(self) -> SuspensionState:
        """Build the design state and its wheel-derived presentation points."""
        if self._initial_state is not None:
            return self._initial_state
        positions = self.get_hardpoints_copy()
        DerivedPointsManager(self.derived_spec()).update_in_place(positions)
        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set[PointKey](self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build the arm, carrier, and selected spring-actuation constraints."""
        initial = self.initial_state()
        positions = initial.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        return [
            distance(PointID.TRAILING_ARM_PIVOT_A, PointID.TRAILING_ARM_OUTBOARD),
            distance(PointID.TRAILING_ARM_PIVOT_B, PointID.TRAILING_ARM_OUTBOARD),
        ]

    def derived_spec(self) -> DerivedPointsSpec:
        """Derive rigid carrier pickups, then the common wheel presentation points."""
        if self.config is None:
            raise ValueError("Cannot compute derived points without configuration")
        wheel_spec = build_wheel_derived_spec(self.config.wheel)
        functions = dict(wheel_spec.functions)
        dependencies = {
            point: set(point_dependencies)
            for point, point_dependencies in wheel_spec.dependencies.items()
        }

        pivot_a = extract_array(self.hardpoints[PointID.TRAILING_ARM_PIVOT_A])
        pivot_b = extract_array(self.hardpoints[PointID.TRAILING_ARM_PIVOT_B])
        axis_direction = pivot_b - pivot_a
        axis_direction /= float(np.linalg.norm(axis_direction))

        def decomposition(point: PointID) -> tuple[np.ndarray, np.ndarray]:
            design = extract_array(self.hardpoints[point])
            axial_distance = float(np.dot(design - pivot_a, axis_direction))
            axis_foot = pivot_a + axis_direction * axial_distance
            return axis_foot, design - axis_foot

        moving_axis_foot, moving_design_radial = decomposition(
            PointID.TRAILING_ARM_OUTBOARD
        )
        attached_points = [
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            PointID.STRUT_BOTTOM,
        ]
        for point in attached_points:
            target_axis_foot, target_design_radial = decomposition(point)
            functions[point] = partial(
                _rotate_arm_attached_point,
                axis_direction=axis_direction,
                moving_axis_foot=moving_axis_foot,
                moving_design_radial=moving_design_radial,
                target_axis_foot=target_axis_foot,
                target_design_radial=target_design_radial,
            )
            dependencies[point] = {PointID.TRAILING_ARM_OUTBOARD}

        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    def _pivot_axis(self, state: SuspensionState) -> tuple[Point3, Direction3]:
        """Return the fixed oblique arm-pivot line from the solved state."""
        pivot_a = state.get(PointID.TRAILING_ARM_PIVOT_A)
        return pivot_a, (state.get(PointID.TRAILING_ARM_PIVOT_B) - pivot_a).normalize()

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """Intersect the oblique pivot axis with the wheel's side-view plane."""
        pivot, direction = self._pivot_axis(state)
        return intersect_line_with_vertical_plane(
            pivot,
            direction,
            float(state.get(PointID.WHEEL_CENTER)[Axis.Y]),
        )

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """Intersect the oblique pivot axis with the wheel's front-view plane."""
        pivot, direction = self._pivot_axis(state)
        return intersect_line_with_axis_aligned_plane(
            pivot,
            direction,
            Axis.X,
            float(state.get(PointID.WHEEL_CENTER)[Axis.X]),
        )

    def _torsion_twist(self, state: SuspensionState) -> float:
        """Return arm motion projected about the authored torsion-bar axis."""
        initial = self.initial_state()
        axis_point = initial.get(PointID.TORSION_BAR_AXIS_A)
        axis = (
            initial.get(PointID.TORSION_BAR_AXIS_B) - axis_point
        ).normalize()
        return self.side.lateral_sign * degrees(
            signed_angle_about_axis(
                initial.get(PointID.TRAILING_ARM_OUTBOARD),
                state.get(PointID.TRAILING_ARM_OUTBOARD),
                axis_point,
                axis,
            )
        )

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare selected spring response relative to hub vertical travel."""
        driver = PointCoordinateResponse.from_world_axis(
            PointID.WHEEL_CENTER,
            Axis.Z,
            name="hub_z",
            unit=MetricUnit.MM,
            label="Hub Z",
        )
        damper_definition = DerivativeMetricDefinition(
            response=PointDistanceResponse(
                PointID.STRUT_TOP,
                PointID.STRUT_BOTTOM,
                name="damper_length",
                unit=MetricUnit.MM,
                label="Damper Length",
            ),
            driver=driver,
        )
        if self.spring_type is CornerSpringType.COILOVER:
            return (damper_definition,)
        design = extract_array(
            self.initial_state().get(PointID.TRAILING_ARM_OUTBOARD)
        )
        axis_a = extract_array(
            self.initial_state().get(PointID.TORSION_BAR_AXIS_A)
        )
        axis_b = extract_array(
            self.initial_state().get(PointID.TORSION_BAR_AXIS_B)
        )
        axis = axis_b - axis_a
        axis /= float((axis**2).sum() ** 0.5)

        def torsion_rotation(positions: DualPositions) -> DualScalar:
            result = self.side.lateral_sign * kernels.rotation_about_fixed_axis_deg(
                positions,
                PointID.TRAILING_ARM_OUTBOARD,
                design,
                axis_a,
                axis,
            )
            assert isinstance(result, DualScalar)
            return result

        return (
            damper_definition,
            DerivativeMetricDefinition(
                response=CallableScalarResponse(
                    torsion_rotation,
                    name="torsion_bar_twist",
                    unit=MetricUnit.DEG,
                    label="Torsion Bar Twist",
                ),
                driver=driver,
            ),
        )

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Expose torsion twist only when a torsion spring is selected."""
        if self.spring_type is CornerSpringType.TORSION_BAR:
            return (TORSION_BAR_TWIST_SPEC,)
        return ()

    def topology_metric_values(self, state: SuspensionState) -> MetricRow:
        """Return the torsion bar angular deflection when installed."""
        if self.spring_type is CornerSpringType.TORSION_BAR:
            return OrderedDict([("torsion_bar_twist", self._torsion_twist(state))])
        return OrderedDict()

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return renderer-neutral arm, carrier, wheel, and spring hardware."""
        carrier_hardpoints = (PointID.TRAILING_ARM_OUTBOARD,)
        carrier_segments = [
            (PointID.TRAILING_ARM_OUTBOARD, PointID.AXLE_INBOARD),
            (PointID.TRAILING_ARM_OUTBOARD, PointID.AXLE_OUTBOARD),
        ]
        elements: tuple[SuspensionElement, ...] = (
            RigidLinkElement(
                label="Semi-Trailing Arm Front Link",
                type=ElementType.WISHBONE,
                point_a=PointID.TRAILING_ARM_PIVOT_A,
                point_b=PointID.TRAILING_ARM_OUTBOARD,
            ),
            RigidLinkElement(
                label="Semi-Trailing Arm Rear Link",
                type=ElementType.WISHBONE,
                point_a=PointID.TRAILING_ARM_PIVOT_B,
                point_b=PointID.TRAILING_ARM_OUTBOARD,
            ),
            UprightElement(
                label="Semi-Trailing Arm Carrier",
                hardpoints=carrier_hardpoints,
                attachments=(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
                segments=tuple(carrier_segments),
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
                wheel_plane_road_tangent=PointID.WHEEL_PLANE_ROAD_TANGENT,
            ),
        )
        if self.spring_type is CornerSpringType.COILOVER:
            return (
                *elements,
                VariableLengthLinkElement(
                    label="Spring/Damper",
                    type=ElementType.SPRING_DAMPER,
                    point_a=PointID.STRUT_TOP,
                    point_b=PointID.STRUT_BOTTOM,
                ),
            )
        return (
            *elements,
            TorsionElement(
                label="Semi-Trailing Arm Torsion Bar",
                type=ElementType.TORSION_BAR,
                rotation_axis=(
                    PointID.TORSION_BAR_AXIS_A,
                    PointID.TORSION_BAR_AXIS_B,
                ),
                attachments=(),
            ),
            VariableLengthLinkElement(
                label="Damper",
                type=ElementType.DAMPER,
                point_a=PointID.STRUT_TOP,
                point_b=PointID.STRUT_BOTTOM,
            ),
        )
