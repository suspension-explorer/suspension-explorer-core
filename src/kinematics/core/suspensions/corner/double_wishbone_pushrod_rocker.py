"""Double-wishbone corner with explicit pushrod and rocker actuation."""

from collections import OrderedDict
from dataclasses import dataclass, field
from math import degrees
from typing import ClassVar, Literal, Sequence

import numpy as np

from kinematics.core.constraints import Constraint, DistanceConstraint
from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    RockerElement,
    RockerPickup,
    RockerPickupType,
    SuspensionElement,
    TorsionElement,
    VariableLengthLinkElement,
)
from kinematics.core.metrics import kernels
from kinematics.core.metrics.derivatives import (
    CallableScalarResponse,
    DerivativeMetricDefinition,
    PointCoordinateResponse,
    PointDistanceResponse,
)
from kinematics.core.metrics.units import MetricUnit
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    signed_angle_about_axis,
)
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.corner.attachments import (
    chiral_rigid_point_constraints,
)
from kinematics.core.suspensions.corner.double_wishbone import DoubleWishboneSuspension

RockerSpringType = Literal["torsion_bar", "coilover"]

ROCKER_POINTS = frozenset(
    {
        PointID.PUSHROD_OUTBOARD,
        PointID.PUSHROD_INBOARD,
        PointID.ROCKER_AXIS_FRONT,
        PointID.ROCKER_AXIS_REAR,
    }
)
COILOVER_POINTS = frozenset({PointID.STRUT_TOP, PointID.STRUT_BOTTOM})


@dataclass
class DoubleWishbonePushrodRockerSuspension(DoubleWishboneSuspension):
    """A double-wishbone corner actuated by a pushrod and rocker."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = (
        DoubleWishboneSuspension.REQUIRED_POINTS | ROCKER_POINTS
    )
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = COILOVER_POINTS
    ROCKER_PICKUPS: ClassVar[tuple[PointID, ...]] = (PointID.PUSHROD_INBOARD,)

    spring_type: RockerSpringType = field(kw_only=True)

    @property
    def has_rocker(self) -> bool:
        """This topology always includes a rocker."""
        return True

    @property
    def has_droplink(self) -> bool:
        """The basic corner has no anti-roll-bar pickup."""
        return False

    @property
    def has_strut(self) -> bool:
        """Whether this rocker drives an inboard coilover."""
        return self.spring_type == "coilover"

    @property
    def has_torsion_bar(self) -> bool:
        """Whether this rocker drives a coaxial torsion bar."""
        return self.spring_type == "torsion_bar"

    def validate_hardpoints(self) -> None:
        """Validate the selected spring medium and non-degenerate rocker axis."""
        super().validate_hardpoints()
        present = set(self.hardpoints)
        coilover_present = COILOVER_POINTS & present
        if self.has_strut and coilover_present != COILOVER_POINTS:
            missing = sorted(point.name for point in COILOVER_POINTS - present)
            raise ValueError(f"Coilover rocker is missing hardpoints: {missing}")
        if self.has_torsion_bar and coilover_present:
            names = sorted(point.name for point in coilover_present)
            raise ValueError(f"Torsion-bar rocker does not accept: {names}")

        axis_front = self.hardpoints[PointID.ROCKER_AXIS_FRONT]
        axis_rear = self.hardpoints[PointID.ROCKER_AXIS_REAR]
        if compute_point_point_distance(axis_front, axis_rear) <= EPS_GEOMETRIC:
            raise ValueError("Rocker axis points must be distinct")
        axis_direction = (axis_rear - axis_front).normalize()
        for point in self.ROCKER_PICKUPS:
            radius = compute_point_to_line_distance(
                self.hardpoints[point], axis_front, axis_direction
            )
            if radius <= EPS_GEOMETRIC:
                raise ValueError(f"{point.name} must not lie on the rocker axis")

    def free_points(self) -> Sequence[PointID]:
        """Return base variables plus pushrod, rocker, and optional damper pickup."""
        points = [
            *super().free_points(),
            PointID.PUSHROD_OUTBOARD,
            *self.ROCKER_PICKUPS,
        ]
        if self.has_strut:
            points.append(PointID.STRUT_BOTTOM)
        return points

    def output_points(self) -> tuple[PointKey, ...]:
        """Return base outputs plus the explicit inboard actuation points."""
        points = [
            *super().output_points(),
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
        ]
        if self.has_droplink:
            points.append(PointID.DROPLINK_ROCKER)
        if self.has_strut:
            points.extend((PointID.STRUT_TOP, PointID.STRUT_BOTTOM))
        return tuple(points)

    def constraints(self) -> list[Constraint]:
        """Return corner constraints plus pushrod, rocker, and spring constraints."""
        constraints = [*super().constraints(), *self._rocker_constraints()]
        if self.has_strut:
            constraints.extend(
                chiral_rigid_point_constraints(
                    self.initial_state(),
                    PointID.STRUT_BOTTOM,
                    (
                        PointID.ROCKER_AXIS_FRONT,
                        PointID.ROCKER_AXIS_REAR,
                        PointID.PUSHROD_INBOARD,
                    ),
                )
            )
        return constraints

    def _rocker_constraints(self) -> list[Constraint]:
        """Build fixed pushrod length and rigid rocker constraints."""
        initial = self.initial_state()
        positions = initial.positions

        def distance(point_a: PointID, point_b: PointID) -> DistanceConstraint:
            return DistanceConstraint(
                point_a,
                point_b,
                compute_point_point_distance(positions[point_a], positions[point_b]),
            )

        constraints: list[Constraint] = [
            distance(PointID.PUSHROD_OUTBOARD, anchor)
            for anchor in (
                PointID.UPPER_WISHBONE_OUTBOARD,
                PointID.LOWER_WISHBONE_OUTBOARD,
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
            )
        ]
        constraints.extend(
            (
                distance(PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD),
                distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_FRONT),
                distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_REAR),
            )
        )
        if self.has_droplink:
            constraints.extend(
                chiral_rigid_point_constraints(
                    initial,
                    PointID.DROPLINK_ROCKER,
                    (
                        PointID.ROCKER_AXIS_FRONT,
                        PointID.ROCKER_AXIS_REAR,
                        PointID.PUSHROD_INBOARD,
                    ),
                )
            )
        return constraints

    def derivative_metric_definitions(
        self,
    ) -> tuple[DerivativeMetricDefinition, ...]:
        """Declare rocker, selected spring, and hub-relative derivatives."""
        design = self.initial_state()
        axis_front = extract_array(design.positions[PointID.ROCKER_AXIS_FRONT])
        axis_rear = extract_array(design.positions[PointID.ROCKER_AXIS_REAR])
        axis_direction = axis_rear - axis_front
        axis_length = float(np.linalg.norm(axis_direction))
        if axis_length < EPS_GEOMETRIC:
            raise ValueError(
                "Rocker angle derivative requires distinct rocker axis points."
            )
        axis_direction /= axis_length
        design_pickup = extract_array(design.positions[PointID.PUSHROD_INBOARD])
        hub_z = PointCoordinateResponse.from_world_axis(
            PointID.WHEEL_CENTER,
            Axis.Z,
            name="hub_z",
            unit=MetricUnit.MM,
        )

        def rocker_rotation(positions):
            return self.side.lateral_sign * kernels.rotation_about_fixed_axis_deg(
                positions,
                PointID.PUSHROD_INBOARD,
                design_pickup,
                axis_front,
                axis_direction,
            )

        definitions = [
            DerivativeMetricDefinition(
                response=CallableScalarResponse(
                    rocker_rotation,
                    name="rocker_angle",
                    unit=MetricUnit.DEG,
                ),
                driver=hub_z,
            )
        ]
        if self.has_torsion_bar:
            definitions.append(
                DerivativeMetricDefinition(
                    response=CallableScalarResponse(
                        rocker_rotation,
                        name="torsion_bar_twist",
                        unit=MetricUnit.DEG,
                    ),
                    driver=hub_z,
                )
            )
        if self.has_strut:
            definitions.append(
                DerivativeMetricDefinition(
                    response=PointDistanceResponse(
                        PointID.STRUT_TOP,
                        PointID.STRUT_BOTTOM,
                        name="damper_length",
                        unit=MetricUnit.MM,
                    ),
                    driver=hub_z,
                )
            )
        return tuple(definitions)

    def topology_metric_values(self, state: SuspensionState):
        """Return rocker rotation and selected spring twist from design."""
        design = self.initial_state()
        axis_front = design.get(PointID.ROCKER_AXIS_FRONT)
        axis = (design.get(PointID.ROCKER_AXIS_REAR) - axis_front).normalize()
        angle = (
            degrees(
                signed_angle_about_axis(
                    design.get(PointID.PUSHROD_INBOARD),
                    state.get(PointID.PUSHROD_INBOARD),
                    axis_front,
                    axis,
                )
            )
            * self.side.lateral_sign
        )
        row = OrderedDict([("rocker_angle", angle)])
        if self.has_torsion_bar:
            row["torsion_bar_twist"] = angle
        return row

    def elements(self) -> tuple[SuspensionElement, ...]:
        """Return base corner elements plus pushrod, spring, and rocker."""
        pickups = (RockerPickup(PointID.PUSHROD_INBOARD, RockerPickupType.PUSHROD),)
        if self.has_droplink:
            pickups += (
                RockerPickup(PointID.DROPLINK_ROCKER, RockerPickupType.DROPLINK),
            )

        rotation_axis: tuple[PointKey, PointKey] = (
            PointID.ROCKER_AXIS_FRONT,
            PointID.ROCKER_AXIS_REAR,
        )
        elements: tuple[SuspensionElement, ...] = (
            *super().elements(),
            RigidLinkElement(
                label="Pushrod",
                type=ElementType.PUSHROD,
                point_a=PointID.PUSHROD_OUTBOARD,
                point_b=PointID.PUSHROD_INBOARD,
            ),
            RockerElement(
                label="Rocker",
                rotation_axis=rotation_axis,
                pickups=pickups,
            ),
        )
        if self.has_strut:
            elements += (
                VariableLengthLinkElement(
                    label="Spring/Damper",
                    type=ElementType.SPRING_DAMPER,
                    point_a=PointID.STRUT_TOP,
                    point_b=PointID.STRUT_BOTTOM,
                ),
            )
        if self.has_torsion_bar:
            elements += (
                TorsionElement(
                    label="Torsion Bar",
                    type=ElementType.TORSION_BAR,
                    rotation_axis=rotation_axis,
                    attachments=(),
                    path=rotation_axis,
                ),
            )
        return elements


@dataclass
class DoubleWishbonePushrodRockerArbSuspension(DoubleWishbonePushrodRockerSuspension):
    """Pushrod-rocker corner exposing the rocker-side ARB pickup."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker_arb"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = (
        DoubleWishbonePushrodRockerSuspension.REQUIRED_POINTS
        | {PointID.DROPLINK_ROCKER}
    )
    ROCKER_PICKUPS: ClassVar[tuple[PointID, ...]] = (
        PointID.PUSHROD_INBOARD,
        PointID.DROPLINK_ROCKER,
    )

    @property
    def has_droplink(self) -> bool:
        """This corner exposes the rocker-side anti-roll-bar pickup."""
        return True
