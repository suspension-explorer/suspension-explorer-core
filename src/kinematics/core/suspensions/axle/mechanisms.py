"""Typed shared mechanisms for composed suspension axles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import acos, degrees
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from kinematics.core.constraints import (
    Constraint,
    DistanceConstraint,
    MidpointOnPlaneConstraint,
)
from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
)
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    TBarElement,
    TorsionElement,
    VariableLengthLinkElement,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.points.derived.manager import (
    DerivedPointsSpec,
)
from kinematics.core.primitives.constants import EPS_GEOMETRIC, MIN_CHIRALITY_VOLUME
from kinematics.core.primitives.geometry import Direction3, Point3, extract_array
from kinematics.core.primitives.point_ref import PointKey, PointRef, Side
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    compute_scalar_triple_product,
    signed_angle_about_axis,
)
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.suspensions.axle.suspension import (
        AxleSuspension,
    )


# Warn when a linkage transmission margin falls below this ratio.
TRANSMISSION_MARGIN_WARNING_THRESHOLD = 0.15

T_BAR_PIVOT_KEY = PointRef(Side.CENTER, PointID.ARB_T_BAR_PIVOT)
T_BAR_LEFT_KEY = PointRef(Side.LEFT, PointID.DROPLINK_T_BAR)
T_BAR_RIGHT_KEY = PointRef(Side.RIGHT, PointID.DROPLINK_T_BAR)


def calculate_t_bar_crossbar_center(
    positions: Mapping[PointKey, Any],
) -> Any:
    """Return the midpoint of the rigid T-bar crossbar."""
    left = positions[T_BAR_LEFT_KEY]
    right = positions[T_BAR_RIGHT_KEY]
    return left + (right - left) / 2.0


def calculate_arb_branch_volume(state: SuspensionState, side: Side) -> float:
    """Calculate the signed volume of one U-bar linkage branch."""
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A))
    return compute_scalar_triple_product(
        state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)) - axis_a,
        state.get(PointRef(side, PointID.DROPLINK_ROCKER)) - axis_a,
        state.get(PointRef(side, PointID.DROPLINK_U_BAR)) - axis_a,
    )


def calculate_arb_chirality_margin(state: SuspensionState, side: Side) -> float:
    """Calculate normalized signed volume for one U-bar linkage branch."""
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)).data
    axis = state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)).data - axis_a
    rocker_arm = state.get(PointRef(side, PointID.DROPLINK_ROCKER)).data - axis_a
    arb_arm = state.get(PointRef(side, PointID.DROPLINK_U_BAR)).data - axis_a
    scale = (
        float(np.linalg.norm(axis))
        * float(np.linalg.norm(rocker_arm))
        * float(np.linalg.norm(arb_arm))
    )
    if scale <= EPS_GEOMETRIC:
        return 0.0
    return calculate_arb_branch_volume(state, side) / scale


def calculate_transmission_margin(
    driven: np.ndarray,
    axis_point: np.ndarray,
    axis: np.ndarray,
    link: np.ndarray,
) -> float | None:
    """Calculate absolute link alignment with the driven circular tangent."""
    axis_norm = float(np.linalg.norm(axis))
    link_norm = float(np.linalg.norm(link))
    if axis_norm == 0.0 or link_norm == 0.0:
        return None
    axis_unit = axis / axis_norm
    radius = driven - axis_point
    radius -= axis_unit * float(np.dot(radius, axis_unit))
    tangent = np.cross(axis_unit, radius)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm == 0.0:
        return None
    return float(abs(np.dot(link / link_norm, tangent / tangent_norm)))


@dataclass(frozen=True)
class ArbNone:
    """Explicit absence of shared anti-roll hardware."""

    def validate(self, axle: AxleSuspension) -> None:
        """Accept any pair of compatible corners."""

    def add_to_state(self, state: SuspensionState) -> None:
        """Add no anti-roll points."""

    def derived_spec(self) -> DerivedPointsSpec[PointKey]:
        """Declare no anti-roll derived points."""
        return DerivedPointsSpec({}, {})

    @property
    def free_points(self) -> tuple[PointKey, ...]:
        """Return no anti-roll free points."""
        return ()

    @property
    def output_points(self) -> tuple[PointKey, ...]:
        """Return no anti-roll output points."""
        return ()

    def constraints(self, axle: AxleSuspension) -> list[Constraint]:
        """Add no anti-roll constraints."""
        return []

    def derivative_metric_definitions(
        self,
        axle: AxleSuspension,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Add no anti-roll derivative metrics."""
        return ()

    def topology_metric_values(
        self,
        axle: AxleSuspension,
        state: SuspensionState,
    ) -> MetricRow:
        """Add no anti-roll state metrics."""
        return OrderedDict()

    def topology_diagnostics(
        self,
        axle: AxleSuspension,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        """Add no anti-roll diagnostics."""
        return []

    def elements(
        self,
        axle: AxleSuspension,
    ) -> tuple[SuspensionElement, ...]:
        """Add no anti-roll elements."""
        return ()


@dataclass(frozen=True)
class ArbUBar:
    """Shared U-bar with one moving arm pickup and droplink per side."""

    center_points: dict[PointID, Point3] = field(default_factory=dict)
    droplink_points: dict[Side, Point3] = field(default_factory=dict)

    def validate(self, axle: AxleSuspension) -> None:
        """Validate droplink pickups and authored U-bar geometry."""
        for side, corner in axle.corners.items():
            # The droplink must anchor to a solver-controlled point riding on
            # a moving corner body; which body provides it is the corner's
            # concern.
            if PointID.DROPLINK_ROCKER not in corner.free_points():
                raise ValueError(
                    f"{side.name} U-bar corner does not expose DROPLINK_ROCKER "
                    "as a moving pickup"
                )

        expected_axis = {PointID.ARB_U_BAR_AXIS_A, PointID.ARB_U_BAR_AXIS_B}
        if set(self.center_points) != expected_axis:
            raise ValueError(
                "U-bar requires center ARB_U_BAR_AXIS_A and ARB_U_BAR_AXIS_B"
            )
        if set(self.droplink_points) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("U-bar requires DROPLINK_U_BAR on both sides")

        axis_a = self.center_points[PointID.ARB_U_BAR_AXIS_A]
        axis_b = self.center_points[PointID.ARB_U_BAR_AXIS_B]
        if compute_point_point_distance(axis_a, axis_b) <= EPS_GEOMETRIC:
            raise ValueError(
                "ARB_U_BAR_AXIS_A and ARB_U_BAR_AXIS_B must be distinct points"
            )
        axis_direction = (axis_b - axis_a).normalize()
        for side, droplink in self.droplink_points.items():
            radius = compute_point_to_line_distance(droplink, axis_a, axis_direction)
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{side.name} DROPLINK_U_BAR lies on the U-bar axis; "
                    "it must be off-axis"
                )
            authored_volume = compute_scalar_triple_product(
                axis_b - axis_a,
                axle.corners[side].hardpoints[PointID.DROPLINK_ROCKER] - axis_a,
                droplink - axis_a,
            )
            if abs(authored_volume) < MIN_CHIRALITY_VOLUME:
                raise ValueError(
                    f"{side.name} U-bar arm geometry does not define reliable "
                    "handedness"
                )

    def add_to_state(self, state: SuspensionState) -> None:
        """Add the shared axis and moving arm pickups to an axle state."""
        for point, position in self.center_points.items():
            state.positions[PointRef(Side.CENTER, point)] = position.copy()
        for side, position in self.droplink_points.items():
            key = PointRef(side, PointID.DROPLINK_U_BAR)
            state.positions[key] = position.copy()
            state.free_points.add(key)

    def derived_spec(self) -> DerivedPointsSpec[PointKey]:
        """Declare no U-bar derived points."""
        return DerivedPointsSpec({}, {})

    @property
    def free_points(self) -> tuple[PointKey, ...]:
        """Return both moving U-bar arm pickups."""
        return (
            PointRef(Side.LEFT, PointID.DROPLINK_U_BAR),
            PointRef(Side.RIGHT, PointID.DROPLINK_U_BAR),
        )

    @property
    def output_points(self) -> tuple[PointKey, ...]:
        """Return both U-bar arm pickups."""
        return self.free_points

    def constraints(self, axle: AxleSuspension) -> list[Constraint]:
        """Constrain each U-bar arm to its axis and rocker droplink."""
        constraints: list[Constraint] = []
        axis_a = self.center_points[PointID.ARB_U_BAR_AXIS_A]
        axis_b = self.center_points[PointID.ARB_U_BAR_AXIS_B]
        axis_a_key = PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)
        axis_b_key = PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)

        for side in (Side.LEFT, Side.RIGHT):
            droplink = self.droplink_points[side]
            arb_key = PointRef(side, PointID.DROPLINK_U_BAR)
            constraints.extend(
                (
                    DistanceConstraint(
                        arb_key,
                        axis_a_key,
                        compute_point_point_distance(droplink, axis_a),
                    ),
                    DistanceConstraint(
                        arb_key,
                        axis_b_key,
                        compute_point_point_distance(droplink, axis_b),
                    ),
                    DistanceConstraint(
                        PointRef(side, PointID.DROPLINK_ROCKER),
                        arb_key,
                        compute_point_point_distance(
                            axle.corners[side]
                            .initial_state()
                            .positions[PointID.DROPLINK_ROCKER],
                            droplink,
                        ),
                    ),
                )
            )
        return constraints

    def derivative_metric_definitions(
        self,
        axle: AxleSuspension,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare U-bar twist relative to each hub Z with the other hub held."""
        design = axle.initial_state()
        axis_a_key = PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)
        axis_b_key = PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)
        axis_point = extract_array(design.positions[axis_a_key])
        axis_direction = extract_array(design.positions[axis_b_key]) - axis_point
        axis_length = float(np.linalg.norm(axis_direction))
        if axis_length < EPS_GEOMETRIC:
            raise ValueError("ARB twist derivative requires distinct ARB axis points")
        axis_direction /= axis_length
        design_pickups = {
            side: extract_array(
                design.positions[PointRef(side, PointID.DROPLINK_U_BAR)]
            )
            for side in (Side.LEFT, Side.RIGHT)
        }

        def arb_twist(positions):
            angles = {
                side: kernels.rotation_about_fixed_axis_deg(
                    positions,
                    PointRef(side, PointID.DROPLINK_U_BAR),
                    design_pickups[side],
                    axis_point,
                    axis_direction,
                )
                for side in (Side.LEFT, Side.RIGHT)
            }
            return angles[Side.LEFT] - angles[Side.RIGHT]

        response = CallableScalarResponse(
            arb_twist,
            name="arb_twist",
            unit=MetricUnit.DEG,
        )
        return tuple(
            DerivativeMetricDefinition(
                response=response,
                driver=PointCoordinateResponse.from_world_axis(
                    PointRef(side, PointID.WHEEL_CENTER),
                    Axis.Z,
                    name=f"hub_z_{side.name.lower()}",
                    unit=MetricUnit.MM,
                ),
            )
            for side in (Side.LEFT, Side.RIGHT)
        )

    def topology_metric_values(
        self,
        axle: AxleSuspension,
        state: SuspensionState,
    ) -> MetricRow:
        """Return per-side U-bar arm rotation and total twist from design."""
        design = axle.initial_state()
        axis_a = design.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A))
        axis = (
            design.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)) - axis_a
        ).normalize()
        angles = {
            side: degrees(
                signed_angle_about_axis(
                    design.get(PointRef(side, PointID.DROPLINK_U_BAR)),
                    state.get(PointRef(side, PointID.DROPLINK_U_BAR)),
                    axis_a,
                    axis,
                )
            )
            for side in (Side.LEFT, Side.RIGHT)
        }
        return OrderedDict(
            (
                ("arb_arm_angle_left", angles[Side.LEFT]),
                ("arb_arm_angle_right", angles[Side.RIGHT]),
                ("arb_twist", angles[Side.LEFT] - angles[Side.RIGHT]),
            )
        )

    def topology_diagnostics(
        self,
        axle: AxleSuspension,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        """Report U-bar branch inversions and approaching linkage toggles."""
        issues: list[DiagnosticIssue] = []
        design = axle.initial_state()
        for side in (Side.LEFT, Side.RIGHT):
            design_sign = np.sign(calculate_arb_branch_volume(design, side))
            for step, state in enumerate(states):
                triple = calculate_arb_branch_volume(state, side)
                margin = calculate_arb_chirality_margin(state, side)
                if abs(margin) <= EPS_GEOMETRIC:
                    issues.append(
                        DiagnosticIssue(
                            step,
                            DiagnosticCategory.CHIRALITY,
                            DiagnosticSeverity.ERROR,
                            f"{side.name.lower()} U-bar arm reached its chirality "
                            f"boundary at step {step}.",
                            margin,
                        )
                    )
                elif np.sign(triple) != design_sign:
                    issues.append(
                        DiagnosticIssue(
                            step,
                            DiagnosticCategory.CHIRALITY,
                            DiagnosticSeverity.ERROR,
                            f"{side.name.lower()} U-bar arm inverted at step {step}.",
                            triple,
                        )
                    )
                issues.extend(self._transmission_issues(state, side, step))
        return issues

    def _transmission_issues(
        self,
        state: SuspensionState,
        side: Side,
        step: int,
    ) -> list[DiagnosticIssue]:
        """Return warnings for the three link-to-lever transmission margins."""

        def point(point_id: PointID) -> np.ndarray:
            return state.get(PointRef(side, point_id)).data

        droplink = point(PointID.DROPLINK_U_BAR) - point(PointID.DROPLINK_ROCKER)
        arb_axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)).data
        arb_axis = (
            state.get(PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)).data - arb_axis_a
        )
        checks = [
            (
                "droplink @ DROPLINK_U_BAR",
                calculate_transmission_margin(
                    point(PointID.DROPLINK_U_BAR),
                    arb_axis_a,
                    arb_axis,
                    droplink,
                ),
            ),
        ]

        # The rocker-side transmission checks only apply when this corner
        # actually carries a pushrod-rocker group; a differently actuated
        # corner can drive the droplink without those points existing.
        rocker_group = (
            PointID.ROCKER_AXIS_A,
            PointID.ROCKER_AXIS_B,
            PointID.PUSHROD_INBOARD,
            PointID.PUSHROD_OUTBOARD,
        )
        if all(
            PointRef(side, point_id) in state.positions for point_id in rocker_group
        ):
            rocker_axis_a = point(PointID.ROCKER_AXIS_A)
            rocker_axis = point(PointID.ROCKER_AXIS_B) - rocker_axis_a
            pushrod = point(PointID.PUSHROD_OUTBOARD) - point(PointID.PUSHROD_INBOARD)
            checks.extend(
                (
                    (
                        "pushrod @ PUSHROD_INBOARD",
                        calculate_transmission_margin(
                            point(PointID.PUSHROD_INBOARD),
                            rocker_axis_a,
                            rocker_axis,
                            pushrod,
                        ),
                    ),
                    (
                        "droplink @ DROPLINK_ROCKER",
                        calculate_transmission_margin(
                            point(PointID.DROPLINK_ROCKER),
                            rocker_axis_a,
                            rocker_axis,
                            droplink,
                        ),
                    ),
                )
            )
        issues: list[DiagnosticIssue] = []
        for joint, margin in checks:
            if margin is None or margin >= TRANSMISSION_MARGIN_WARNING_THRESHOLD:
                continue
            angle_from_toggle = 90.0 - degrees(acos(min(1.0, margin)))
            issues.append(
                DiagnosticIssue(
                    step,
                    DiagnosticCategory.TRANSMISSION,
                    DiagnosticSeverity.WARNING,
                    f"{side.name.lower()} {joint} is {angle_from_toggle:.1f} deg "
                    f"from toggle at step {step} (margin {margin:.3g}).",
                    margin,
                )
            )
        return issues

    def elements(
        self,
        axle: AxleSuspension,
    ) -> tuple[SuspensionElement, ...]:
        """Return one continuous U-bar and its two droplinks."""
        design = axle.initial_state()
        left_droplink = design.positions[PointRef(Side.LEFT, PointID.DROPLINK_U_BAR)]
        axis_a = design.positions[PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_A)]
        axis_b = design.positions[PointRef(Side.CENTER, PointID.ARB_U_BAR_AXIS_B)]
        if compute_point_point_distance(
            left_droplink, axis_a
        ) <= compute_point_point_distance(left_droplink, axis_b):
            left_end, right_end = (
                PointID.ARB_U_BAR_AXIS_A,
                PointID.ARB_U_BAR_AXIS_B,
            )
        else:
            left_end, right_end = (
                PointID.ARB_U_BAR_AXIS_B,
                PointID.ARB_U_BAR_AXIS_A,
            )

        elements: list[SuspensionElement] = [
            TorsionElement(
                label="Anti-Roll Bar",
                type=ElementType.ANTI_ROLL_BAR,
                rotation_axis=(
                    PointRef(Side.CENTER, left_end),
                    PointRef(Side.CENTER, right_end),
                ),
                attachments=(
                    PointRef(Side.LEFT, PointID.DROPLINK_U_BAR),
                    PointRef(Side.RIGHT, PointID.DROPLINK_U_BAR),
                ),
            )
        ]
        for side in (Side.LEFT, Side.RIGHT):
            elements.append(
                RigidLinkElement(
                    label=f"{side.name.title()} Droplink",
                    type=ElementType.DROPLINK,
                    point_a=PointRef(side, PointID.DROPLINK_ROCKER),
                    point_b=PointRef(side, PointID.DROPLINK_U_BAR),
                )
            )
        return tuple(elements)


@dataclass(frozen=True)
class ArbTBar:
    """Rigid T-bar connected to the rockers by two droplinks.

    The two arm pickups and chassis-fixed pivot form a rigid triangle. The
    crossbar midpoint remains on the vehicle XZ plane, so the rigid T rotates
    about its stem while both droplinks preserve their design lengths.
    """

    center_points: dict[PointID, Point3] = field(default_factory=dict)
    droplink_points: dict[Side, Point3] = field(default_factory=dict)

    def validate(self, axle: AxleSuspension) -> None:
        """Validate droplink pickups and the authored T-bar geometry."""
        for side, corner in axle.corners.items():
            if PointID.DROPLINK_ROCKER not in corner.free_points():
                raise ValueError(
                    f"{side.name} T-bar corner does not expose DROPLINK_ROCKER "
                    "as a moving pickup"
                )

        expected_center = {PointID.ARB_T_BAR_PIVOT}
        if set(self.center_points) != expected_center:
            raise ValueError("T-bar requires center ARB_T_BAR_PIVOT")
        if set(self.droplink_points) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("T-bar requires DROPLINK_T_BAR on both sides")

        pivot = self.center_points[PointID.ARB_T_BAR_PIVOT]
        if abs(float(pivot[Axis.Y])) > EPS_GEOMETRIC:
            raise ValueError("ARB_T_BAR_PIVOT must lie on the vehicle centerline Y = 0")
        left = self.droplink_points[Side.LEFT]
        right = self.droplink_points[Side.RIGHT]
        crossbar_center = left + (right - left) / 2.0
        if abs(float(crossbar_center[Axis.Y])) > EPS_GEOMETRIC:
            raise ValueError(
                "The T-bar crossbar midpoint must lie on the vehicle centerline Y = 0"
            )
        crossbar = right - left
        stem = crossbar_center - pivot
        if crossbar.norm() <= EPS_GEOMETRIC:
            raise ValueError("T-bar crossbar points must be distinct")
        if stem.norm() <= EPS_GEOMETRIC:
            raise ValueError("T-bar pivot and crossbar midpoint must be distinct")
        if crossbar.cross(stem).norm() <= EPS_GEOMETRIC:
            raise ValueError("T-bar points must define a non-degenerate triangle")

    def add_to_state(self, state: SuspensionState) -> None:
        """Add the fixed pivot and moving crossbar endpoints."""
        state.positions[T_BAR_PIVOT_KEY] = self.center_points[
            PointID.ARB_T_BAR_PIVOT
        ].copy()
        for side, position in self.droplink_points.items():
            key = PointRef(side, PointID.DROPLINK_T_BAR)
            state.positions[key] = position.copy()
            state.free_points.add(key)

    def derived_spec(self) -> DerivedPointsSpec[PointKey]:
        """Declare no T-bar derived solver points."""
        return DerivedPointsSpec({}, {})

    @property
    def free_points(self) -> tuple[PointKey, ...]:
        """Return both moving T-bar arm pickups."""
        return T_BAR_LEFT_KEY, T_BAR_RIGHT_KEY

    @property
    def output_points(self) -> tuple[PointKey, ...]:
        """Return the two crossbar endpoints."""
        return self.free_points

    def constraints(self, axle: AxleSuspension) -> list[Constraint]:
        """Preserve the rigid T triangle and both droplink lengths."""
        design = axle.initial_state()
        constraints: list[Constraint] = [
            DistanceConstraint(
                T_BAR_LEFT_KEY,
                T_BAR_RIGHT_KEY,
                compute_point_point_distance(
                    design.get(T_BAR_LEFT_KEY),
                    design.get(T_BAR_RIGHT_KEY),
                ),
            ),
            DistanceConstraint(
                T_BAR_LEFT_KEY,
                T_BAR_PIVOT_KEY,
                compute_point_point_distance(
                    design.get(T_BAR_LEFT_KEY),
                    design.get(T_BAR_PIVOT_KEY),
                ),
            ),
            DistanceConstraint(
                T_BAR_RIGHT_KEY,
                T_BAR_PIVOT_KEY,
                compute_point_point_distance(
                    design.get(T_BAR_RIGHT_KEY),
                    design.get(T_BAR_PIVOT_KEY),
                ),
            ),
            MidpointOnPlaneConstraint(
                T_BAR_LEFT_KEY,
                T_BAR_RIGHT_KEY,
                Point3([0.0, 0.0, 0.0]),
                Direction3([0.0, 1.0, 0.0]),
            ),
        ]
        for side in (Side.LEFT, Side.RIGHT):
            arb_key = PointRef(side, PointID.DROPLINK_T_BAR)
            rocker_key = PointRef(side, PointID.DROPLINK_ROCKER)
            constraints.append(
                DistanceConstraint(
                    rocker_key,
                    arb_key,
                    compute_point_point_distance(
                        design.get(rocker_key), design.get(arb_key)
                    ),
                )
            )
        return constraints

    def derivative_metric_definitions(
        self,
        axle: AxleSuspension,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare T-bar center X motion relative to each hub Z."""
        response = CallableScalarResponse(
            lambda positions: calculate_t_bar_crossbar_center(positions)[Axis.X],
            name="t_bar_center_x",
            unit=MetricUnit.MM,
        )
        return tuple(
            DerivativeMetricDefinition(
                response=response,
                driver=PointCoordinateResponse.from_world_axis(
                    PointRef(side, PointID.WHEEL_CENTER),
                    Axis.Z,
                    name=f"hub_z_{side.name.lower()}",
                    unit=MetricUnit.MM,
                ),
            )
            for side in (Side.LEFT, Side.RIGHT)
        )

    def topology_metric_values(
        self,
        axle: AxleSuspension,
        state: SuspensionState,
    ) -> MetricRow:
        """Return T-bar stem heave angle and shaft twist."""
        design = axle.initial_state()
        pivot = design.get(T_BAR_PIVOT_KEY)
        heave_angle = degrees(
            signed_angle_about_axis(
                calculate_t_bar_crossbar_center(design.positions),
                calculate_t_bar_crossbar_center(state.positions),
                pivot,
                Direction3([0.0, 1.0, 0.0]),
            )
        )
        design_twist = self._shaft_twist(design)
        current_twist = self._shaft_twist(state)
        return OrderedDict(
            (
                ("t_bar_heave_angle", heave_angle),
                ("t_bar_twist", degrees(current_twist - design_twist)),
            )
        )

    @staticmethod
    def _shaft_twist(state: SuspensionState) -> float:
        """Return crossbar rotation about the moving stem axis."""
        pivot = state.get(T_BAR_PIVOT_KEY).data
        center = calculate_t_bar_crossbar_center(state.positions).data
        stem = center - pivot
        stem /= np.linalg.norm(stem)
        crossbar = state.get(T_BAR_LEFT_KEY).data - state.get(T_BAR_RIGHT_KEY).data
        crossbar -= stem * float(np.dot(crossbar, stem))
        lateral_reference = np.array([0.0, 1.0, 0.0])
        sine = float(np.dot(stem, np.cross(lateral_reference, crossbar)))
        cosine = float(np.dot(lateral_reference, crossbar))
        return float(np.arctan2(sine, cosine))

    def topology_diagnostics(
        self,
        axle: AxleSuspension,
        states: list[SuspensionState],
    ) -> list[DiagnosticIssue]:
        """Return no T-bar-specific diagnostics yet."""
        return []

    def elements(
        self,
        axle: AxleSuspension,
    ) -> tuple[SuspensionElement, ...]:
        """Return the branched T-bar and one droplink per side."""
        left = T_BAR_LEFT_KEY
        right = T_BAR_RIGHT_KEY
        return (
            TBarElement(
                label="T-Bar Anti-Roll Bar",
                pivot=T_BAR_PIVOT_KEY,
                left_attachment=left,
                right_attachment=right,
            ),
            RigidLinkElement(
                label="Left Droplink",
                type=ElementType.DROPLINK,
                point_a=PointRef(Side.LEFT, PointID.DROPLINK_ROCKER),
                point_b=left,
            ),
            RigidLinkElement(
                label="Right Droplink",
                type=ElementType.DROPLINK,
                point_a=PointRef(Side.RIGHT, PointID.DROPLINK_ROCKER),
                point_b=right,
            ),
        )


type AxleArb = ArbNone | ArbUBar | ArbTBar


@dataclass(frozen=True)
class HeaveLinkNone:
    """Explicit absence of a rocker-to-rocker heave link."""

    def validate(self, axle: AxleSuspension) -> None:
        """Accept any pair of compatible corners."""

    def derivative_metric_definitions(
        self,
        axle: AxleSuspension,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Add no heave-link derivatives."""
        return ()

    def topology_metric_values(
        self,
        state: SuspensionState,
    ) -> MetricRow:
        """Add no heave-link state metrics."""
        return OrderedDict()

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Add no heave-link element."""
        return ()


@dataclass(frozen=True)
class HeaveLinkRockerToRocker:
    """Variable-length link between left and right rocker pickups."""

    def validate(self, axle: AxleSuspension) -> None:
        """Require a moving heave-link pickup on both corners."""
        for side, corner in axle.corners.items():
            if PointID.HEAVE_LINK_ROCKER not in corner.free_points():
                raise ValueError(
                    f"{side.name} corner does not expose HEAVE_LINK_ROCKER "
                    "as a moving pickup"
                )

    def derivative_metric_definitions(
        self,
        axle: AxleSuspension,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare heave-link length relative to each hub Z."""
        response = PointDistanceResponse(
            PointRef(Side.LEFT, PointID.HEAVE_LINK_ROCKER),
            PointRef(Side.RIGHT, PointID.HEAVE_LINK_ROCKER),
            name="heave_link_length",
            unit=MetricUnit.MM,
        )
        return tuple(
            DerivativeMetricDefinition(
                response=response,
                driver=PointCoordinateResponse.from_world_axis(
                    PointRef(side, PointID.WHEEL_CENTER),
                    Axis.Z,
                    name=f"hub_z_{side.name.lower()}",
                    unit=MetricUnit.MM,
                ),
            )
            for side in (Side.LEFT, Side.RIGHT)
        )

    def topology_metric_values(self, state: SuspensionState) -> MetricRow:
        """Return installed heave-link length."""
        left = state.get(PointRef(Side.LEFT, PointID.HEAVE_LINK_ROCKER))
        right = state.get(PointRef(Side.RIGHT, PointID.HEAVE_LINK_ROCKER))
        return OrderedDict([("heave_link_length", float((left - right).norm()))])

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return one unconstrained variable-length rocker link."""
        return (
            VariableLengthLinkElement(
                label="Heave Link",
                type=ElementType.HEAVE_LINK,
                point_a=PointRef(Side.LEFT, PointID.HEAVE_LINK_ROCKER),
                point_b=PointRef(Side.RIGHT, PointID.HEAVE_LINK_ROCKER),
            ),
        )


type AxleHeaveLink = HeaveLinkNone | HeaveLinkRockerToRocker
