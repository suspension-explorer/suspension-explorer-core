"""Double-wishbone pushrod-rocker axle with a shared anti-roll bar."""

from collections import OrderedDict
from dataclasses import dataclass, field
from math import acos, degrees
from typing import TYPE_CHECKING, ClassVar, Sequence

import numpy as np

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.diagnostics import (
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticSeverity,
)
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    TorsionElement,
)
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.constants import EPS_GEOMETRIC, MIN_CHIRALITY_VOLUME
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.geometry import Point3, extract_array
from kinematics.core.primitives.point_ref import (
    PointKey,
    PointRef,
    Side,
    side_qualified,
)
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    compute_scalar_triple_product,
    signed_angle_about_axis,
)
from kinematics.core.suspensions.axle.double_wishbone import (
    DoubleWishboneAxleSuspension,
)
from kinematics.core.suspensions.corner.double_wishbone_pushrod_rocker import (
    DoubleWishbonePushrodRockerArbSuspension,
)

TRANSMISSION_MARGIN_WARNING_THRESHOLD = 0.15
"""Warn when a linkage transmission margin falls below this ratio."""

if TYPE_CHECKING:
    from kinematics.core.state import SuspensionState


def calculate_arb_branch_volume(state: "SuspensionState", side: Side) -> float:
    """Calculate the signed volume of one ARB linkage branch."""
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A))
    return compute_scalar_triple_product(
        state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)) - axis_a,
        state.get(PointRef(side, PointID.DROPLINK_ROCKER)) - axis_a,
        state.get(PointRef(side, PointID.DROPLINK_ARB)) - axis_a,
    )


def calculate_arb_chirality_margin(
    state: "SuspensionState",
    side: Side,
) -> float:
    """Calculate normalized signed volume for one ARB linkage branch."""
    axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A)).data
    axis = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)).data - axis_a
    rocker_arm = state.get(PointRef(side, PointID.DROPLINK_ROCKER)).data - axis_a
    arb_arm = state.get(PointRef(side, PointID.DROPLINK_ARB)).data - axis_a
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


@dataclass
class DoubleWishbonePushrodRockerAxleSuspension(DoubleWishboneAxleSuspension):
    """Two ARB-ready rocker corners coupled by one anti-roll bar."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker_axle"

    corners: dict[Side, DoubleWishbonePushrodRockerArbSuspension] = field(
        default_factory=dict
    )
    center_points: dict[PointID, Point3] = field(default_factory=dict)
    droplink_arb_points: dict[Side, Point3] = field(default_factory=dict)

    @property
    def has_arb(self) -> bool:
        """This topology always includes a complete anti-roll bar."""
        return True

    @property
    def has_rocker(self) -> bool:
        """Both corners in this topology have rockers."""
        return True

    @property
    def has_torsion_bar(self) -> bool:
        """Return whether both corners use torsion bars."""
        return all(corner.has_torsion_bar for corner in self.corners.values())

    @property
    def has_strut(self) -> bool:
        """Return whether both corners use inboard coilovers."""
        return all(corner.has_strut for corner in self.corners.values())

    def validate_hardpoints(self) -> None:
        """Validate the two rocker corners and axle-owned ARB geometry."""
        super().validate_hardpoints()
        for side, corner in self.corners.items():
            if not isinstance(corner, DoubleWishbonePushrodRockerArbSuspension):
                raise ValueError(
                    f"{side.name} corner must be an ARB-ready pushrod-rocker model."
                )

        expected_axis = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B}
        if set(self.center_points) != expected_axis:
            raise ValueError("ARB axle requires center ARB_AXIS_A and ARB_AXIS_B.")
        if set(self.droplink_arb_points) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("ARB axle requires DROPLINK_ARB on both sides.")

        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]
        if compute_point_point_distance(axis_a, axis_b) <= EPS_GEOMETRIC:
            raise ValueError("ARB_AXIS_A and ARB_AXIS_B must be distinct points.")
        axis_direction = (axis_b - axis_a).normalize()
        for side, droplink in self.droplink_arb_points.items():
            radius = compute_point_to_line_distance(droplink, axis_a, axis_direction)
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{side.name} DROPLINK_ARB lies on the ARB axis (zero radius); "
                    "it must be off-axis to trace an ARB arm arc."
                )
            authored_volume = compute_scalar_triple_product(
                axis_b - axis_a,
                self.corners[side].hardpoints[PointID.DROPLINK_ROCKER] - axis_a,
                droplink - axis_a,
            )
            if abs(authored_volume) < MIN_CHIRALITY_VOLUME:
                raise ValueError(
                    f"{side.name} ARB arm geometry does not define reliable handedness"
                )

    def initial_state(self) -> "SuspensionState":
        """Add the shared ARB axis and moving arm pickups to the axle state."""
        if self._initial_state is not None:
            return self._initial_state
        state = super().initial_state()
        for point, position in self.center_points.items():
            state.positions[PointRef(Side.CENTER, point)] = position.copy()
        for side, position in self.droplink_arb_points.items():
            key = PointRef(side, PointID.DROPLINK_ARB)
            state.positions[key] = position.copy()
            state.free_points.add(key)
        state.free_points_order = sorted(state.free_points)
        return state

    def free_points(self) -> Sequence[PointKey]:
        """Return corner variables plus both moving ARB arm pickups."""
        return (
            *super().free_points(),
            PointRef(Side.LEFT, PointID.DROPLINK_ARB),
            PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
        )

    def output_points(self) -> tuple[PointKey, ...]:
        """Return per-side rocker outputs plus both ARB arm pickups."""
        result: list[PointKey] = []
        for side in (Side.LEFT, Side.RIGHT):
            result.extend(
                side_qualified(side, point)
                for point in self.corners[side].output_points()
            )
            result.append(PointRef(side, PointID.DROPLINK_ARB))
        return tuple(result)

    def constraints(self) -> list[Constraint]:
        """Return corner, steering-coupling, and ARB linkage constraints."""
        return [*super().constraints(), *self._arb_constraints()]

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare ARB twist relative to each hub Z with the other hub held."""
        design = self.initial_state()
        axis_a_key = PointRef(Side.CENTER, PointID.ARB_AXIS_A)
        axis_b_key = PointRef(Side.CENTER, PointID.ARB_AXIS_B)
        axis_point = extract_array(design.positions[axis_a_key])
        axis_direction = extract_array(design.positions[axis_b_key]) - axis_point
        axis_length = float(np.linalg.norm(axis_direction))
        if axis_length < EPS_GEOMETRIC:
            raise ValueError("ARB twist derivative requires distinct ARB axis points.")
        axis_direction /= axis_length
        design_pickups = {
            side: extract_array(design.positions[PointRef(side, PointID.DROPLINK_ARB)])
            for side in (Side.LEFT, Side.RIGHT)
        }

        def arb_twist(positions):
            angles = {
                side: kernels.rotation_about_fixed_axis_deg(
                    positions,
                    PointRef(side, PointID.DROPLINK_ARB),
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

    def topology_diagnostics(self, states):
        """Report ARB branch inversions and approaching linkage toggles."""
        issues: list[DiagnosticIssue] = []
        design = self.initial_state()
        for side in (Side.LEFT, Side.RIGHT):
            design_triple = calculate_arb_branch_volume(design, side)
            design_sign = np.sign(design_triple)
            for step, state in enumerate(states):
                triple = calculate_arb_branch_volume(state, side)
                margin = calculate_arb_chirality_margin(state, side)
                if abs(margin) <= EPS_GEOMETRIC:
                    issues.append(
                        DiagnosticIssue(
                            step,
                            DiagnosticCategory.CHIRALITY,
                            DiagnosticSeverity.ERROR,
                            f"{side.name.lower()} ARB arm reached its chirality "
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
                            f"{side.name.lower()} ARB arm inverted at step {step}.",
                            triple,
                        )
                    )
                issues.extend(self._transmission_issues(state, side, step))
        return issues

    def topology_metric_values(self, state):
        """Return per-side ARB arm rotation and total bar twist from design."""
        design = self.initial_state()
        axis_a = design.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A))
        axis = (
            design.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)) - axis_a
        ).normalize()
        angles = {
            side: degrees(
                signed_angle_about_axis(
                    design.get(PointRef(side, PointID.DROPLINK_ARB)),
                    state.get(PointRef(side, PointID.DROPLINK_ARB)),
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

    def _transmission_issues(self, state, side: Side, step: int):
        """Return warnings for the three link-to-lever transmission margins."""

        def point(point_id: PointID):
            return state.get(PointRef(side, point_id)).data

        rocker_axis_a = point(PointID.ROCKER_AXIS_FRONT)
        rocker_axis = point(PointID.ROCKER_AXIS_REAR) - rocker_axis_a
        pushrod = point(PointID.PUSHROD_OUTBOARD) - point(PointID.PUSHROD_INBOARD)
        droplink = point(PointID.DROPLINK_ARB) - point(PointID.DROPLINK_ROCKER)
        arb_axis_a = state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_A)).data
        arb_axis = (
            state.get(PointRef(Side.CENTER, PointID.ARB_AXIS_B)).data - arb_axis_a
        )
        checks = (
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
            (
                "droplink @ DROPLINK_ARB",
                calculate_transmission_margin(
                    point(PointID.DROPLINK_ARB),
                    arb_axis_a,
                    arb_axis,
                    droplink,
                ),
            ),
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

    def _arb_constraints(self) -> list[Constraint]:
        """Constrain each ARB arm to its axis and rocker droplink."""
        constraints: list[Constraint] = []
        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]
        axis_a_key = PointRef(Side.CENTER, PointID.ARB_AXIS_A)
        axis_b_key = PointRef(Side.CENTER, PointID.ARB_AXIS_B)

        for side in (Side.LEFT, Side.RIGHT):
            droplink = self.droplink_arb_points[side]
            arb_key = PointRef(side, PointID.DROPLINK_ARB)
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
                            self.corners[side]
                            .initial_state()
                            .positions[PointID.DROPLINK_ROCKER],
                            droplink,
                        ),
                    ),
                )
            )
        return constraints

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return base elements plus one continuous ARB and two droplinks."""
        elements = list(super().elements())
        design = self.initial_state()
        left_droplink = design.positions[PointRef(Side.LEFT, PointID.DROPLINK_ARB)]
        axis_a = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_A)]
        axis_b = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_B)]
        if compute_point_point_distance(
            left_droplink, axis_a
        ) <= compute_point_point_distance(left_droplink, axis_b):
            left_end, right_end = PointID.ARB_AXIS_A, PointID.ARB_AXIS_B
        else:
            left_end, right_end = PointID.ARB_AXIS_B, PointID.ARB_AXIS_A

        elements.append(
            TorsionElement(
                label="Anti-Roll Bar",
                type=ElementType.ANTI_ROLL_BAR,
                rotation_axis=(
                    PointRef(Side.CENTER, left_end),
                    PointRef(Side.CENTER, right_end),
                ),
                attachments=(
                    PointRef(Side.LEFT, PointID.DROPLINK_ARB),
                    PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
                ),
                path=(
                    PointRef(Side.LEFT, PointID.DROPLINK_ARB),
                    PointRef(Side.CENTER, left_end),
                    PointRef(Side.CENTER, right_end),
                    PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
                ),
            )
        )
        for side in (Side.LEFT, Side.RIGHT):
            elements.append(
                RigidLinkElement(
                    label=f"{side.name.title()} Droplink",
                    type=ElementType.DROPLINK,
                    point_a=PointRef(side, PointID.DROPLINK_ROCKER),
                    point_b=PointRef(side, PointID.DROPLINK_ARB),
                )
            )
        return tuple(elements)
