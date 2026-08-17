"""
Multi-link corner suspension implementation.

The wheel carrier is located by four independent two-joint rods (upper
front/rear and lower front/rear links) plus the configured wheel-heading
link, giving the classic five-link arrangement when steered by a track rod.
Every rod has its own outboard ball joint, so no joint-to-joint line exists
to act as a kingpin: the corner declares no physical steering axis and its
steering geometry is reported exclusively through the motion-derived
(screw-axis) virtual metric family.

Installed actuation and spring behaviour is composed through the same typed
mechanism fields as the double wishbone. Because every locating rod is a
two-force member, a rigid off-axis pickup can only ride the upright; a
direct spring pickup may alternatively ride a lower link's centreline as a
derived point (a damper fork clamped around the link).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    SuspensionElement,
    UprightElement,
    WheelElement,
)
from kinematics.core.enums import (
    MountBody,
    PointID,
    ShimType,
    SteeringType,
    SuspensionType,
)
from kinematics.core.holds import CoordinateHold
from kinematics.core.points.derived.definitions import build_wheel_derived_spec
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    anchored_rigid_point_constraints,
    validate_rigid_anchor_points,
)
from kinematics.core.suspensions.corner.base import CornerSuspension
from kinematics.core.suspensions.corner.mechanisms import (
    Actuation,
    ActuationDirect,
    CornerDamper,
    CornerDamperNone,
    CornerSpring,
    CornerSpringNone,
    composed_derivative_metric_definitions,
    composed_mechanism_elements,
    composed_mechanism_free_points,
    composed_output_points,
    composed_topology_metric_values,
    composed_upright_attachments,
    validate_composed_mechanisms,
)
from kinematics.core.suspensions.corner.toe_link import ToeLink
from kinematics.core.suspensions.corner.track_rod import TrackRod

if TYPE_CHECKING:
    from kinematics.core.metrics.derivatives import DerivativeMetricDefinition
    from kinematics.core.metrics.main import MetricRow
    from kinematics.core.metrics.registry import MetricSpec
    from kinematics.core.steering_response import SuspensionHoldCatalogue


@dataclass
class MultiLinkSuspension(CornerSuspension):
    """Multi-link locating geometry with composed corner mechanisms."""

    TYPE_KEY: ClassVar[SuspensionType] = SuspensionType.MULTI_LINK
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.UPPER_FRONT_LINK_INBOARD,
            PointID.UPPER_FRONT_LINK_OUTBOARD,
            PointID.UPPER_REAR_LINK_INBOARD,
            PointID.UPPER_REAR_LINK_OUTBOARD,
            PointID.LOWER_FRONT_LINK_INBOARD,
            PointID.LOWER_FRONT_LINK_OUTBOARD,
            PointID.LOWER_REAR_LINK_INBOARD,
            PointID.LOWER_REAR_LINK_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )

    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset()

    # The four locating rods as (label, inboard, outboard) triples. Each rod
    # is an independent two-force member with its own outboard ball joint.
    LINKS: ClassVar[tuple[tuple[str, PointID, PointID], ...]] = (
        (
            "Upper Front Link",
            PointID.UPPER_FRONT_LINK_INBOARD,
            PointID.UPPER_FRONT_LINK_OUTBOARD,
        ),
        (
            "Upper Rear Link",
            PointID.UPPER_REAR_LINK_INBOARD,
            PointID.UPPER_REAR_LINK_OUTBOARD,
        ),
        (
            "Lower Front Link",
            PointID.LOWER_FRONT_LINK_INBOARD,
            PointID.LOWER_FRONT_LINK_OUTBOARD,
        ),
        (
            "Lower Rear Link",
            PointID.LOWER_REAR_LINK_INBOARD,
            PointID.LOWER_REAR_LINK_OUTBOARD,
        ),
    )

    # The rigid wheel carrier. The first three points are the chirality base
    # triangle used to anchor every other carried point, so they must be
    # non-collinear in the authored geometry.
    UPRIGHT_BODY: ClassVar[tuple[PointID, ...]] = (
        PointID.LOWER_FRONT_LINK_OUTBOARD,
        PointID.LOWER_REAR_LINK_OUTBOARD,
        PointID.UPPER_FRONT_LINK_OUTBOARD,
        PointID.UPPER_REAR_LINK_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    UPRIGHT_ATTACHMENTS: ClassVar[tuple[PointID, ...]] = (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )
    # The upright carries a pickup rigidly anywhere. A lower locating rod,
    # being a two-force member, carries one only on its own centreline as a
    # derived point (a damper fork clamped around the link).
    MOUNT_BODIES: ClassVar[dict[MountBody, tuple[PointID, ...]]] = {
        MountBody.UPRIGHT: UPRIGHT_BODY,
        MountBody.LOWER_FRONT_LINK: (
            PointID.LOWER_FRONT_LINK_INBOARD,
            PointID.LOWER_FRONT_LINK_OUTBOARD,
        ),
        MountBody.LOWER_REAR_LINK: (
            PointID.LOWER_REAR_LINK_INBOARD,
            PointID.LOWER_REAR_LINK_OUTBOARD,
        ),
    }

    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset()

    # Points included in solver output (CSV/Parquet), in column order.
    # Hardpoints first, then derived points.
    LOCATING_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_FRONT_LINK_INBOARD,
        PointID.UPPER_FRONT_LINK_OUTBOARD,
        PointID.UPPER_REAR_LINK_INBOARD,
        PointID.UPPER_REAR_LINK_OUTBOARD,
        PointID.LOWER_FRONT_LINK_INBOARD,
        PointID.LOWER_FRONT_LINK_OUTBOARD,
        PointID.LOWER_REAR_LINK_INBOARD,
        PointID.LOWER_REAR_LINK_OUTBOARD,
    )
    WHEEL_OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.AXLE_MIDPOINT,
        PointID.WHEEL_CENTER,
        PointID.WHEEL_INBOARD,
        PointID.WHEEL_OUTBOARD,
        PointID.WHEEL_CONTACT_CENTRE,
    )
    OUTPUT_POINTS: ClassVar[tuple[PointID, ...]] = (
        *LOCATING_OUTPUT_POINTS,
        *WHEEL_OUTPUT_POINTS,
    )
    # The contact centre is reported but never driven. On an axle it comes from a
    # coupled solve across both corners, with a bounded validity domain and
    # non-unique roots; the standalone flat-ground construction is the same
    # quantity, so it is refused as a target at both scopes.
    OUTPUT_ONLY_POINTS: ClassVar[tuple[PointID, ...]] = (PointID.WHEEL_CONTACT_CENTRE,)

    # Free points that move during solving.
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_FRONT_LINK_OUTBOARD,
        PointID.UPPER_REAR_LINK_OUTBOARD,
        PointID.LOWER_FRONT_LINK_OUTBOARD,
        PointID.LOWER_REAR_LINK_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )

    wheel_heading_link: TrackRod | ToeLink = field(init=False)
    actuation: Actuation = field(
        default_factory=lambda: ActuationDirect(
            spring_pickup_body=MultiLinkSuspension.UPRIGHT_BODY
        ),
        kw_only=True,
    )
    spring: CornerSpring = field(default_factory=CornerSpringNone, kw_only=True)
    damper: CornerDamper = field(default_factory=CornerDamperNone, kw_only=True)

    def __post_init__(self) -> None:
        """Install a track rod or fixed toe link for wheel-heading control."""
        if self.config is None:
            raise ValueError("Multi-link suspension requires configuration")
        # The anchored attachment pins the authored assembly branch; there is
        # no separate upright angle constraint to preserve it here.
        if self.config.steering.type is SteeringType.RACK:
            self.wheel_heading_link = TrackRod(self.UPRIGHT_BODY)
        else:
            self.wheel_heading_link = ToeLink(self.UPRIGHT_BODY)
        super().__post_init__()

    def required_points(self) -> frozenset[PointID]:
        """Return base and selected mechanism point requirements."""
        return (
            self.REQUIRED_POINTS
            | self.wheel_heading_link.REQUIRED_POINTS
            | self.actuation.required_points
            | self.spring.required_points
            | self.damper.required_points
        )

    def validate_hardpoints(self) -> None:
        """Validate base geometry and selected mechanism compatibility."""
        super().validate_hardpoints()
        validate_rigid_anchor_points(
            self.hardpoints,
            self.UPRIGHT_BODY,
            "Multi-link upright",
        )
        self.wheel_heading_link.validate(self.hardpoints)
        validate_composed_mechanisms(
            self.hardpoints,
            self.actuation,
            self.spring,
            self.damper,
        )

    def free_points(self) -> Sequence[PointID]:
        """Return base and selected mechanism moving points.

        An on-link spring pickup is a derived point on the rod centreline,
        so it is excluded from the spring's declared moving points.
        """
        return (
            *self.FREE_POINTS,
            *self.wheel_heading_link.free_points,
            *composed_mechanism_free_points(
                self.actuation,
                self.spring,
                self.damper,
            ),
        )

    def output_points(self) -> tuple[PointKey, ...]:
        """Return base and selected mechanism output points."""
        return composed_output_points(
            (
                *self.LOCATING_OUTPUT_POINTS,
                *self.wheel_heading_link.OUTPUT_POINTS,
                *self.WHEEL_OUTPUT_POINTS,
            ),
            self.actuation,
            self.spring,
            self.damper,
        )

    def damper_points(self) -> tuple[PointKey, PointKey] | None:
        """Return selected linear spring/damper endpoints."""
        return self.damper.damper_points or self.spring.damper_points

    def suspension_hold_catalogue(self) -> "SuspensionHoldCatalogue | None":
        """Declare the locked-internals hold for multi-link steering response.

        Without wishbone hinges there is no fixed-axis arm-angle coordinate to
        hold, so the installed damper length is the only published boundary
        condition. The moving pickup necessarily rides on the upright, so the
        locked-internals response mixes a little suspension motion into the
        steering screw; that coupling is part of the meaning of the result.
        """
        from kinematics.core.steering_response import (
            SuspensionHoldCatalogue,
            SuspensionHoldOption,
        )

        if self.steering_actuator_coordinate() is None:
            return None
        damper = self._installed_damper_coordinate()
        if damper is None:
            return None
        return SuspensionHoldCatalogue(
            default_option_id="damper_length",
            options=(
                SuspensionHoldOption(
                    id="damper_length",
                    label="Damper length",
                    description=(
                        "Locks the internal suspension at the current installed "
                        "true damper length during the steering response. A "
                        "multi-link corner has no physical kingpin, so the "
                        "resulting screw axis is the definition of its steering "
                        "axis under this locked-internals condition."
                    ),
                    hold=CoordinateHold((damper,)),
                ),
            ),
        )

    def steering_axis_points(self) -> tuple[PointID, PointID] | None:
        """No two ball joints define a kingpin; the steering axis is virtual."""
        return None

    def rack_attachment_point(self) -> PointID | None:
        """Return the track-rod rack pickup for a steered corner."""
        if isinstance(self.wheel_heading_link, TrackRod):
            return self.wheel_heading_link.inboard_point
        return None

    def initial_state(self) -> SuspensionState:
        """Build the initial state from hardpoints plus derived points."""
        if self._initial_state is not None:
            return self._initial_state

        positions = self.get_hardpoints_copy()
        derived_manager = DerivedPointsManager(self.derived_spec())
        derived_manager.update_in_place(positions)

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set[PointKey](self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build link-length, rigid-carrier, and mechanism constraints."""
        initial_state = self.initial_state()
        positions = initial_state.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        # Each locating rod preserves its ball-joint-to-ball-joint length.
        constraints: list[Constraint] = [
            distance(inboard, outboard) for _, inboard, outboard in self.LINKS
        ]

        # Carrier rigidity: the base triangle holds its three mutual
        # distances, and every other carried point is anchored to it with
        # authored handedness.
        base_triangle = self.UPRIGHT_BODY[:3]
        constraints.extend(
            (
                distance(base_triangle[0], base_triangle[1]),
                distance(base_triangle[1], base_triangle[2]),
                distance(base_triangle[2], base_triangle[0]),
            )
        )
        for carried_point in self.UPRIGHT_BODY[3:]:
            constraints.extend(
                anchored_rigid_point_constraints(
                    initial_state,
                    carried_point,
                    base_triangle,
                )
            )
        # The axle pair also holds its own length so the wheel axis cannot
        # drift inside the redundant carrier constraint set.
        constraints.append(distance(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD))

        constraints.extend(self.wheel_heading_link.constraints(initial_state))
        constraints.extend(self.actuation.constraints(initial_state))
        constraints.extend(self.spring.constraints(initial_state, self.actuation))
        constraints.extend(self.damper.constraints(initial_state, self.actuation))
        return constraints

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Compose derivative declarations from actuation and spring mechanisms."""
        initial = self.initial_state()
        return composed_derivative_metric_definitions(
            initial,
            self.side,
            self.actuation,
            self.spring,
            self.damper,
        )

    def topology_metric_values(self, state: SuspensionState) -> MetricRow:
        """Compose state metrics from actuation and spring mechanisms."""
        return composed_topology_metric_values(
            state,
            self.initial_state(),
            self.side,
            self.actuation,
            self.spring,
        )

    def topology_metric_specs(self) -> tuple[MetricSpec, ...]:
        """Compose state metric metadata from installed corner mechanisms."""
        return (
            *self.actuation.topology_metric_specs(),
            *self.spring.topology_metric_specs(),
        )

    def derived_spec(self) -> DerivedPointsSpec:
        """Wheel derived points, plus the on-link spring pickup if selected.

        An on-link pickup rides the rod centreline at its authored axial
        offset from the inboard joint, so it follows the link through every
        solved state without contributing solver unknowns.
        """
        if self.config is None:
            raise ValueError("Cannot compute derived spec without config")
        actuation_spec = self.actuation.derived_spec(self.hardpoints)
        wheel_spec = build_wheel_derived_spec(self.config.wheel)
        return DerivedPointsSpec(
            {**actuation_spec.functions, **wheel_spec.functions},
            {**actuation_spec.dependencies, **wheel_spec.dependencies},
        )

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """
        A multi-link carrier has no plane-intersection instant axis.

        The bump motion of the carrier is a general screw, so the classical
        side-view instant-centre construction does not apply and the derived
        swing-arm metrics are reported as undefined.
        """
        return None

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """
        A multi-link carrier has no plane-intersection instant axis.

        See :meth:`compute_side_view_instant_center`; the front-view
        construction is undefined for the same reason.
        """
        return None

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return the physical elements in this corner."""
        heading_link_outboard = self.wheel_heading_link.outboard_point
        link_elements = tuple(
            RigidLinkElement(
                label=label,
                type=ElementType.WISHBONE,
                point_a=inboard,
                point_b=outboard,
            )
            for label, inboard, outboard in self.LINKS
        )
        upright_hardpoints = composed_upright_attachments(
            (
                PointID.UPPER_FRONT_LINK_OUTBOARD,
                PointID.UPPER_REAR_LINK_OUTBOARD,
                PointID.LOWER_REAR_LINK_OUTBOARD,
                PointID.LOWER_FRONT_LINK_OUTBOARD,
                heading_link_outboard,
            ),
            self.actuation,
            self.UPRIGHT_BODY,
            self.hardpoints,
        )
        base_elements: tuple[SuspensionElement, ...] = (
            *link_elements,
            UprightElement(
                label="Upright",
                hardpoints=upright_hardpoints,
                attachments=(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
                segments=(
                    (
                        PointID.UPPER_FRONT_LINK_OUTBOARD,
                        PointID.UPPER_REAR_LINK_OUTBOARD,
                    ),
                    (
                        PointID.UPPER_REAR_LINK_OUTBOARD,
                        PointID.LOWER_REAR_LINK_OUTBOARD,
                    ),
                    (
                        PointID.LOWER_REAR_LINK_OUTBOARD,
                        PointID.LOWER_FRONT_LINK_OUTBOARD,
                    ),
                    (
                        PointID.LOWER_FRONT_LINK_OUTBOARD,
                        PointID.UPPER_FRONT_LINK_OUTBOARD,
                    ),
                    (heading_link_outboard, PointID.UPPER_FRONT_LINK_OUTBOARD),
                    (heading_link_outboard, PointID.LOWER_FRONT_LINK_OUTBOARD),
                ),
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
                wheel_contact_centre=PointID.WHEEL_CONTACT_CENTRE,
            ),
        )
        return (
            *base_elements,
            *self.wheel_heading_link.elements(),
            *composed_mechanism_elements(
                self.actuation,
                self.spring,
                self.damper,
            ),
        )

    def upright_attachment_points(self) -> tuple[PointID, ...]:
        """Return points carried by the upright body."""
        return composed_upright_attachments(
            (*self.UPRIGHT_ATTACHMENTS, self.wheel_heading_link.outboard_point),
            self.actuation,
            self.UPRIGHT_BODY,
            self.hardpoints,
        )
