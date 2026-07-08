"""
Double wishbone axle suspension (two coupled corners).

This module defines :class:`DoubleWishboneAxleSuspension`, which composes two
:class:`~kinematics.suspensions.double_wishbone.DoubleWishboneSuspension` corner
instances (left and right) into a single constraint system solved together. The
corners are coupled through a rigid steering rack: the two inboard trackrod
points are held a fixed distance apart, so a single steering input drives both
wheels.

The axle keys its state, constraints, and derived points on
:class:`~kinematics.core.point_ref.PointRef` (``(side, point)``) rather than the
bare :class:`~kinematics.core.enums.PointID`. Everything below re-keys the
unchanged corner machinery into the two side namespaces; the corner classes
themselves are reused verbatim.

Coordinate system: ISO 8855 (X forward, Y left, Z up). The LEFT corner is the
+Y side; mirroring left <-> right is a reflection through the XZ plane
(``y -> -y`` for points, Y component negated for directions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Mapping, Sequence

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey, PointRef, Side
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
)
from kinematics.points.derived.manager import (
    DerivedPointsSpec,
    PositionFn,
    PositionValue,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension

if TYPE_CHECKING:
    from kinematics.metrics.main import MetricRow
    from kinematics.sensitivity import TangentField
    from kinematics.visualization.main import LinkVisualization, WheelAnchors


class _SideView(Mapping):
    """
    A value-type-agnostic view of one side of an axle positions dict.

    Corner-level code (constraints, derived-point functions) is written against
    plain :class:`~kinematics.core.enums.PointID` keys. The axle stores every
    position under a :class:`~kinematics.core.point_ref.PointRef`. This view
    forwards ``view[pid]`` to ``positions[PointRef(side, pid)]`` so the corner
    functions can run unchanged against the side-tagged dict.

    The view only forwards lookups, so it works identically for ``Point3`` and
    ``DualVec3`` values (the dual-number autodiff path in
    :meth:`~kinematics.solver.ResidualComputer.compute_jacobian` runs derived
    functions on dual-valued dicts).
    """

    __slots__ = ("_positions", "_side")

    def __init__(self, positions: Mapping[PointKey, Any], side: Side) -> None:
        self._positions = positions
        self._side = side

    def __getitem__(self, point: PointID) -> Any:
        return self._positions[PointRef(self._side, point)]

    def __setitem__(self, point: PointID, value: Any) -> None:
        # Derived-point functions are pure (they return values); __setitem__ is
        # provided for completeness so the view can stand in for a mutable dict.
        self._positions[PointRef(self._side, point)] = value  # type: ignore[index]

    def __contains__(self, point: object) -> bool:
        return PointRef(self._side, point) in self._positions  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[PointID]:
        for key in self._positions:
            if isinstance(key, PointRef) and key.side == self._side:
                yield key.point

    def __len__(self) -> int:
        return sum(
            1
            for key in self._positions
            if isinstance(key, PointRef) and key.side == self._side
        )


@dataclass
class DoubleWishboneAxleSuspension(Suspension):
    """
    A full double-wishbone axle: two corners solved in one coupled system.

    The two corners are independent double-wishbone models coupled by a rigid
    steering rack (a fixed distance between the two inboard trackrod points).
    Each corner also keeps its own ``PointOnLineConstraint`` holding its inboard
    trackrod on the world Y line, so the rack translates purely laterally.

    Degrees of freedom: two wheel-travel DOFs (one per corner) plus one shared
    rack DOF. A sweep MUST pin the rack DOF with a steering target -- e.g. a
    target on the LEFT ``TRACKROD_INBOARD`` along Y -- exactly as the
    single-corner model requires a steering target to be well posed. A typical
    axle sweep targets left wheel-centre Z, right wheel-centre Z, and left
    trackrod-inboard Y.
    """

    TYPE_KEY: ClassVar[str] = "double_wishbone_axle"

    # No axle-level required points: hardpoints live on the corners.
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()

    # Left corner output points (corner order) followed by the right corner's.
    OUTPUT_POINTS: ClassVar[tuple[PointRef, ...]] = tuple(
        PointRef(Side.LEFT, p) for p in DoubleWishboneSuspension.OUTPUT_POINTS
    ) + tuple(PointRef(Side.RIGHT, p) for p in DoubleWishboneSuspension.OUTPUT_POINTS)

    corners: dict[Side, DoubleWishboneSuspension] = field(default_factory=dict)

    # Shared, chassis-fixed ARB axis points (ARB_AXIS_A, ARB_AXIS_B). Not
    # mirrored -- authored once about the vehicle centreline.
    center_points: dict[PointID, Point3] = field(default_factory=dict)

    # Per-side DROPLINK_ARB design positions (on each ARB arm).
    droplink_arb_points: dict[Side, Point3] = field(default_factory=dict)

    @property
    def has_arb(self) -> bool:
        """
        True when a complete inboard ARB is present.

        Requires both shared axis points, an ARB droplink on each side, and a
        rocker droplink on each side (the droplink joins the two).
        """
        axis_ok = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B} <= set(self.center_points)
        droplinks_ok = {Side.LEFT, Side.RIGHT} <= set(self.droplink_arb_points)
        rockers_ok = all(corner.has_droplink for corner in self.corners.values())
        return axis_ok and droplinks_ok and rockers_ok

    def validate_hardpoints(self) -> None:
        """Delegate corner validation, then validate the ARB group as a whole."""
        for corner in self.corners.values():
            corner.validate_hardpoints()

        # ARB is all-or-nothing across: both shared axis points, an ARB droplink
        # on each side, and a rocker droplink on each side.
        axis_points = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B} & set(self.center_points)
        arb_sides = set(self.droplink_arb_points)
        rocker_sides = {
            side for side, corner in self.corners.items() if corner.has_droplink
        }
        any_arb = bool(axis_points or arb_sides)
        if not any_arb:
            return

        problems: list[str] = []
        if axis_points != {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B}:
            problems.append("both center 'arb_axis_a' and 'arb_axis_b' must be given")
        if arb_sides != {Side.LEFT, Side.RIGHT}:
            problems.append("'droplink_arb' must be given on both sides")
        if rocker_sides != {Side.LEFT, Side.RIGHT}:
            problems.append("'droplink_rocker' must be present on both sides")
        if problems:
            raise ValueError(
                "Incomplete ARB group (all-or-nothing): " + "; ".join(problems) + "."
            )

        # Axis points distinct.
        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]
        if compute_point_point_distance(axis_a, axis_b) <= EPS_GEOMETRIC:
            raise ValueError("ARB_AXIS_A and ARB_AXIS_B must be distinct points.")

        # Each ARB droplink off-axis (non-zero radius) so it traces an arc.
        axis_direction = (axis_b - axis_a).normalize()
        for side, droplink in self.droplink_arb_points.items():
            radius = compute_point_to_line_distance(droplink, axis_a, axis_direction)
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{side.name} DROPLINK_ARB lies on the ARB axis (zero radius); "
                    "it must be off-axis to trace an ARB arm arc."
                )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def initial_state(self) -> SuspensionState:
        """Union of both corners' initial states, side-tagged by ``PointRef``."""
        if self._initial_state is not None:
            return self._initial_state

        positions: dict[PointKey, Point3] = {}
        free_points: set[PointKey] = set()
        for side, corner in self.corners.items():
            # Use the corner's own initial_state so any camber-shim adjustment is
            # applied per corner.
            corner_state = corner.initial_state()
            for pid, pos in corner_state.positions.items():
                positions[PointRef(side, pid)] = pos.copy()
            for pid in corner_state.free_points:
                free_points.add(PointRef(side, pid))

        # Shared ARB axis points (chassis-fixed) and per-side ARB droplinks (free).
        if self.has_arb:
            for pid, pos in self.center_points.items():
                positions[PointRef(Side.CENTER, pid)] = pos.copy()
            for side, pos in self.droplink_arb_points.items():
                key = PointRef(side, PointID.DROPLINK_ARB)
                positions[key] = pos.copy()
                free_points.add(key)

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=free_points,
        )
        return self._initial_state

    def free_points(self) -> Sequence[PointKey]:
        """Each corner's free points, side-tagged, plus per-side ARB droplinks."""
        result: list[PointKey] = []
        for side, corner in self.corners.items():
            result.extend(PointRef(side, pid) for pid in corner.free_points())
        if self.has_arb:
            for side in self.droplink_arb_points:
                result.append(PointRef(side, PointID.DROPLINK_ARB))
        return result

    def output_points(self) -> tuple[PointKey, ...]:
        """Per-side corner output points plus per-side ARB droplink."""
        result: list[PointKey] = []
        for side in (Side.LEFT, Side.RIGHT):
            corner = self.corners[side]
            result.extend(PointRef(side, pid) for pid in corner.output_points())
            if self.has_arb:
                result.append(PointRef(side, PointID.DROPLINK_ARB))
        return tuple(result)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def constraints(self) -> list[Constraint]:
        """Both corners' constraints (re-keyed) plus the rigid rack coupling."""
        constraints: list[Constraint] = []
        for side, corner in self.corners.items():
            for constraint in corner.constraints():
                constraints.append(
                    constraint.remap(lambda pid, s=side: PointRef(s, pid))
                )

        # Rigid steering rack: hold the two inboard trackrod points a fixed
        # (design) distance apart so a single steering input drives both wheels.
        left_tri = (
            self.corners[Side.LEFT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        right_tri = (
            self.corners[Side.RIGHT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        rack_separation = compute_point_point_distance(left_tri, right_tri)
        constraints.append(
            DistanceConstraint(
                PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                rack_separation,
            )
        )

        if self.has_arb:
            constraints.extend(self._arb_constraints())

        return constraints

    def _arb_constraints(self) -> list[Constraint]:
        """
        Distance constraints for the inboard anti-roll bar, per side.

        Each ARB arm end (``DROPLINK_ARB``) rides an arc about the shared,
        chassis-fixed ARB axis (2 distances to the two CENTER axis points), and
        the droplink is a fixed-length link from that side's ``DROPLINK_ROCKER``
        to its ``DROPLINK_ARB``. The two arms are otherwise independent -- the
        bar's torsional compliance is what the ARB *is* -- so wheel targets stay
        independent and the twist is reported as a metric.
        """
        constraints: list[Constraint] = []
        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]

        for side in (Side.LEFT, Side.RIGHT):
            droplink = self.droplink_arb_points[side]
            arb_key = PointRef(side, PointID.DROPLINK_ARB)
            axis_a_key = PointRef(Side.CENTER, PointID.ARB_AXIS_A)
            axis_b_key = PointRef(Side.CENTER, PointID.ARB_AXIS_B)

            # ARB arm end on the ARB circle.
            constraints.append(
                DistanceConstraint(
                    arb_key,
                    axis_a_key,
                    compute_point_point_distance(droplink, axis_a),
                )
            )
            constraints.append(
                DistanceConstraint(
                    arb_key,
                    axis_b_key,
                    compute_point_point_distance(droplink, axis_b),
                )
            )

            # Droplink length: rocker droplink <-> ARB droplink.
            droplink_rocker = (
                self.corners[side].initial_state().positions[PointID.DROPLINK_ROCKER]
            )
            constraints.append(
                DistanceConstraint(
                    PointRef(side, PointID.DROPLINK_ROCKER),
                    arb_key,
                    compute_point_point_distance(droplink_rocker, droplink),
                )
            )

        return constraints

    # ------------------------------------------------------------------
    # Derived points
    # ------------------------------------------------------------------

    def derived_spec(self) -> DerivedPointsSpec:
        """Each corner's derived spec, wrapped and re-keyed per side."""
        functions: dict[PointKey, PositionFn] = {}
        dependencies: dict[PointKey, set[PointKey]] = {}
        for side, corner in self.corners.items():
            spec = corner.derived_spec()
            for pid, fn in spec.functions.items():
                functions[PointRef(side, pid)] = self._wrap_derived_fn(fn, side)
            for pid, deps in spec.dependencies.items():
                dependencies[PointRef(side, pid)] = {
                    PointRef(side, dep) for dep in deps
                }
        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    @staticmethod
    def _wrap_derived_fn(fn: PositionFn, side: Side) -> PositionFn:
        """
        Adapt a corner derived function to the side-tagged positions dict.

        The returned function receives the full ``PointRef``-keyed positions
        dict and exposes it to the corner function as a plain ``PointID`` view
        for ``side``. This is value-type agnostic, so it works for both
        ``Point3`` and ``DualVec3`` dicts.
        """

        def wrapped(positions: dict[PointKey, PositionValue]) -> PositionValue:
            return fn(_SideView(positions, side))  # type: ignore[arg-type]

        return wrapped

    # ------------------------------------------------------------------
    # Per-side reuse
    # ------------------------------------------------------------------

    def corner_state(self, state: SuspensionState, side: Side) -> SuspensionState:
        """
        Strip a solved axle state down to one side's plain corner state.

        The returned state is keyed on plain ``PointID`` (side tag removed), so
        it can be fed to the unchanged corner suspension for metrics, instant
        centers, and geometry.

        Args:
            state: The solved axle state (``PointRef``-keyed).
            side: The side to extract.

        Returns:
            A ``PointID``-keyed :class:`SuspensionState` for that corner.
        """
        positions: dict[PointKey, Point3] = {}
        free_points: set[PointKey] = set()
        for key, pos in state.positions.items():
            if isinstance(key, PointRef) and key.side == side:
                positions[key.point] = pos
        for key in state.free_points:
            if isinstance(key, PointRef) and key.side == side:
                free_points.add(key.point)
        return SuspensionState(positions=positions, free_points=free_points)

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """Not defined at axle level; instant centers are per side."""
        raise NotImplementedError("per-side; use corner_state()/corners[side]")

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """Not defined at axle level; instant centers are per side."""
        raise NotImplementedError("per-side; use corner_state()/corners[side]")

    # ------------------------------------------------------------------
    # Metrics dispatch
    # ------------------------------------------------------------------

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "MetricRow":
        """Compute per-side and axle-level metrics for a solved axle state."""
        from kinematics.metrics.main import compute_metrics_for_axle_state

        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_axle_state(state, self, self.config, tangents)

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Axle sweep targets must name a side; returns a ``PointRef``."""
        if side is None:
            raise ValueError(
                f"Sweep target for '{point.name}' requires a 'side' "
                "(left/right) for an axle geometry."
            )
        return PointRef(side, point)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_visualization_links(self) -> list["LinkVisualization"]:
        """Both corners' links (re-keyed) plus a rack link between the sides."""
        from kinematics.visualization.main import LinkVisualization

        links: list[LinkVisualization] = []
        for side, corner in self.corners.items():
            for link in corner.get_visualization_links():
                links.append(
                    LinkVisualization(
                        points=[PointRef(side, p) for p in link.points],
                        color=link.color,
                        label=f"{side.name.title()} {link.label}",
                        linewidth=link.linewidth,
                        linestyle=link.linestyle,
                        marker=link.marker,
                        markersize=link.markersize,
                    )
                )

        links.append(
            LinkVisualization(
                points=[
                    PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                    PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                ],
                color="purple",
                label="Steering Rack",
            )
        )

        if self.has_arb:
            # The whole ARB as one series: left arm end, in along the left lever
            # arm to the bar end, along the bar, and out the right lever arm to
            # the right arm end. Pair each side with its nearer bar end (at
            # design) so the levers draw from the correct ends of the bar.
            design = self.initial_state()
            left_droplink = design.positions[PointRef(Side.LEFT, PointID.DROPLINK_ARB)]
            axis_a = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_A)]
            axis_b = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_B)]
            near_left = compute_point_point_distance(left_droplink, axis_a)
            near_right = compute_point_point_distance(left_droplink, axis_b)
            if near_left <= near_right:
                left_end, right_end = PointID.ARB_AXIS_A, PointID.ARB_AXIS_B
            else:
                left_end, right_end = PointID.ARB_AXIS_B, PointID.ARB_AXIS_A
            links.append(
                LinkVisualization(
                    points=[
                        PointRef(Side.LEFT, PointID.DROPLINK_ARB),
                        PointRef(Side.CENTER, left_end),
                        PointRef(Side.CENTER, right_end),
                        PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
                    ],
                    color="teal",
                    label="Anti-Roll Bar",
                )
            )
            # Droplinks joining each rocker to its ARB arm end.
            for side in (Side.LEFT, Side.RIGHT):
                links.append(
                    LinkVisualization(
                        points=[
                            PointRef(side, PointID.DROPLINK_ROCKER),
                            PointRef(side, PointID.DROPLINK_ARB),
                        ],
                        color="goldenrod",
                        label=f"{side.name.title()} Droplink",
                    )
                )

        return links

    def wheel_visualization_anchors(self) -> "list[WheelAnchors]":
        """One wheel anchor set per side so both wheels are drawn."""
        from kinematics.visualization.main import WheelAnchors

        anchors: list[WheelAnchors] = []
        for side in self.corners:
            anchors.append(
                WheelAnchors(
                    center=PointRef(side, PointID.WHEEL_CENTER),
                    inboard=PointRef(side, PointID.WHEEL_INBOARD),
                    outboard=PointRef(side, PointID.WHEEL_OUTBOARD),
                    axle_inboard=PointRef(side, PointID.AXLE_INBOARD),
                    axle_outboard=PointRef(side, PointID.AXLE_OUTBOARD),
                )
            )
        return anchors
