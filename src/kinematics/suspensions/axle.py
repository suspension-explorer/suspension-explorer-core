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
from kinematics.core.geometry import Direction3, Point3
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
from kinematics.suspensions.config.settings import SuspensionConfig
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

    # Per-side ARB_DROPLINK design positions (on each ARB arm).
    arb_droplinks: dict[Side, Point3] = field(default_factory=dict)

    @property
    def has_arb(self) -> bool:
        """
        True when a complete inboard ARB is present.

        Requires both shared axis points, an ARB droplink on each side, and a
        rocker droplink on each side (the droplink joins the two).
        """
        axis_ok = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B} <= set(self.center_points)
        droplinks_ok = {Side.LEFT, Side.RIGHT} <= set(self.arb_droplinks)
        rockers_ok = all(corner.has_rocker_droplink for corner in self.corners.values())
        return axis_ok and droplinks_ok and rockers_ok

    def validate_hardpoints(self) -> None:
        """Delegate corner validation, then validate the ARB group as a whole."""
        for corner in self.corners.values():
            corner.validate_hardpoints()

        # ARB is all-or-nothing across: both shared axis points, an ARB droplink
        # on each side, and a rocker droplink on each side.
        axis_points = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B} & set(self.center_points)
        arb_sides = set(self.arb_droplinks)
        rocker_sides = {
            side for side, corner in self.corners.items() if corner.has_rocker_droplink
        }
        any_arb = bool(axis_points or arb_sides)
        if not any_arb:
            return

        problems: list[str] = []
        if axis_points != {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B}:
            problems.append("both center 'arb_axis_a' and 'arb_axis_b' must be given")
        if arb_sides != {Side.LEFT, Side.RIGHT}:
            problems.append("'arb_droplink' must be given on both sides")
        if rocker_sides != {Side.LEFT, Side.RIGHT}:
            problems.append("'rocker_droplink' must be present on both sides")
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
        for side, droplink in self.arb_droplinks.items():
            radius = compute_point_to_line_distance(droplink, axis_a, axis_direction)
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{side.name} ARB_DROPLINK lies on the ARB axis (zero radius); "
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
            for side, pos in self.arb_droplinks.items():
                key = PointRef(side, PointID.ARB_DROPLINK)
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
            for side in self.arb_droplinks:
                result.append(PointRef(side, PointID.ARB_DROPLINK))
        return result

    def output_points(self) -> tuple[PointKey, ...]:
        """Per-side corner output points plus per-side ARB droplink."""
        result: list[PointKey] = []
        for side in (Side.LEFT, Side.RIGHT):
            corner = self.corners[side]
            result.extend(PointRef(side, pid) for pid in corner.output_points())
            if self.has_arb:
                result.append(PointRef(side, PointID.ARB_DROPLINK))
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

        Each ARB arm end (``ARB_DROPLINK``) rides an arc about the shared,
        chassis-fixed ARB axis (2 distances to the two CENTER axis points), and
        the droplink is a fixed-length link from that side's ``ROCKER_DROPLINK``
        to its ``ARB_DROPLINK``. The two arms are otherwise independent -- the
        bar's torsional compliance is what the ARB *is* -- so wheel targets stay
        independent and the twist is reported as a metric.
        """
        constraints: list[Constraint] = []
        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]

        for side in (Side.LEFT, Side.RIGHT):
            droplink = self.arb_droplinks[side]
            arb_key = PointRef(side, PointID.ARB_DROPLINK)
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
            rocker_droplink = (
                self.corners[side].initial_state().positions[PointID.ROCKER_DROPLINK]
            )
            constraints.append(
                DistanceConstraint(
                    PointRef(side, PointID.ROCKER_DROPLINK),
                    arb_key,
                    compute_point_point_distance(rocker_droplink, droplink),
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
            left_droplink = design.positions[PointRef(Side.LEFT, PointID.ARB_DROPLINK)]
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
                        PointRef(Side.LEFT, PointID.ARB_DROPLINK),
                        PointRef(Side.CENTER, left_end),
                        PointRef(Side.CENTER, right_end),
                        PointRef(Side.RIGHT, PointID.ARB_DROPLINK),
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
                            PointRef(side, PointID.ROCKER_DROPLINK),
                            PointRef(side, PointID.ARB_DROPLINK),
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

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml_data(
        cls, yaml_data: dict[str, Any]
    ) -> "DoubleWishboneAxleSuspension":
        """
        Build an axle from the axle YAML schema (see PLAN.md A3).

        Two hardpoint modes are supported:

        - **Mirror mode**: ``hardpoints`` has a ``points`` block describing one
          side (``side: left`` by default), plus an optional ``mirror`` flag
          (default true). The other side is generated by ``y -> -y``.
        - **Explicit mode**: ``hardpoints`` has both ``left`` and ``right``
          flat corner blocks.

        The shared ``config`` uses the single-corner config schema. The LEFT
        corner takes it verbatim; the RIGHT corner takes a mirrored copy (camber
        shim face points ``y``-negated and the shim face normal's Y component
        negated), since the config is authored in the vehicle/left frame.
        """
        from kinematics.core.enums import Units
        from kinematics.io.geometry_loader import parse_hardpoints
        from kinematics.io.validation import coerce_enum

        raw_hardpoints = yaml_data.get("hardpoints", {})
        if not isinstance(raw_hardpoints, dict):
            raise ValueError("Axle 'hardpoints' must be a mapping")

        config_data = yaml_data.get("config", yaml_data.get("configuration", {}))
        base_config = _validate_config(config_data)

        units_str = yaml_data.get("units", "MILLIMETERS")
        try:
            units = coerce_enum(Units, units_str)
        except ValueError:
            raise ValueError(f"Unknown units: {units_str}")

        # Resolve per-side hardpoint dicts (corner points) and the per-side ARB
        # droplink, which is an axle-only point the corner class must not accept.
        if "left" in raw_hardpoints or "right" in raw_hardpoints:
            side_hardpoints, arb_droplinks = _parse_explicit_hardpoints(
                raw_hardpoints, parse_hardpoints
            )
        else:
            side_hardpoints, arb_droplinks = _parse_mirror_hardpoints(
                raw_hardpoints, parse_hardpoints
            )

        # Shared, non-mirrored ARB axis points.
        center_points = _parse_center_points(raw_hardpoints.get("center"))

        name = yaml_data.get("name", "unnamed")
        version = yaml_data.get("version", "0.0.0")

        # LEFT gets the config verbatim; RIGHT gets the mirrored config.
        side_configs = {
            Side.LEFT: base_config,
            Side.RIGHT: _mirror_config(base_config),
        }

        corners: dict[Side, DoubleWishboneSuspension] = {}
        for side in (Side.LEFT, Side.RIGHT):
            corners[side] = DoubleWishboneSuspension(
                name=f"{name}_{side.name.lower()}",
                version=version,
                units=units,
                hardpoints=side_hardpoints[side],
                config=side_configs[side],
            )

        return cls(
            name=name,
            version=version,
            units=units,
            hardpoints={},
            config=base_config,
            corners=corners,
            center_points=center_points,
            arb_droplinks=arb_droplinks,
        )


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------


def _validate_config(config_data: Any) -> SuspensionConfig:
    """Validate a corner config block, raising a clear error on failure."""
    from pydantic import ValidationError as PydanticValidationError

    try:
        return SuspensionConfig.model_validate(config_data)
    except PydanticValidationError as e:
        raise ValueError(f"Configuration validation error: {e}") from e


def _mirror_point(point: Point3) -> Point3:
    """Reflect a point through the XZ plane (``y -> -y``)."""
    x, y, z = float(point.data[0]), float(point.data[1]), float(point.data[2])
    return Point3([x, -y, z])


def _mirror_hardpoints(
    hardpoints: dict[PointKey, Point3],
) -> dict[PointKey, Point3]:
    """Reflect every hardpoint through the XZ plane."""
    return {pid: _mirror_point(pos) for pid, pos in hardpoints.items()}


def _mirror_config(config: SuspensionConfig) -> SuspensionConfig:
    """
    Mirror a corner config for the opposite side.

    Only the camber shim (if present) carries side-dependent geometry: its face
    datum points are ``y``-negated and the face normal's Y component is negated.
    All other config fields are symmetric and copied verbatim.
    """
    if config.camber_shim is None:
        return config

    shim = config.camber_shim
    normal = shim.shim_face_normal
    mirrored_normal = Direction3(
        [float(normal.data[0]), -float(normal.data[1]), float(normal.data[2])]
    )
    mirrored_shim = shim.model_copy(
        update={
            "shim_face_point_a": _mirror_point(shim.shim_face_point_a),
            "shim_face_point_b": _mirror_point(shim.shim_face_point_b),
            "shim_face_normal": mirrored_normal,
        }
    )
    return config.model_copy(update={"camber_shim": mirrored_shim})


def _require_no_errors(errors: list[str], context: str) -> None:
    """Raise a ValueError if hardpoint parsing produced errors."""
    if errors:
        raise ValueError(f"{context}:\n  - " + "\n  - ".join(errors))


def _validate_side_signs(hardpoints: dict[PointKey, Point3], side: Side) -> None:
    """
    Check a corner's outboard Y sign matches its declared side.

    LEFT (+Y) requires ``AXLE_OUTBOARD`` Y > 0; RIGHT (-Y) requires Y < 0.
    """
    axle_out_y = float(hardpoints[PointID.AXLE_OUTBOARD].data[1])
    if side == Side.LEFT and axle_out_y <= 0:
        raise ValueError(
            "Side 'left' requires AXLE_OUTBOARD Y > 0 "
            f"(got {axle_out_y}); check the hardpoint handedness."
        )
    if side == Side.RIGHT and axle_out_y >= 0:
        raise ValueError(
            "Side 'right' requires AXLE_OUTBOARD Y < 0 "
            f"(got {axle_out_y}); check the hardpoint handedness."
        )


def _pop_arb_droplink(points_raw: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """
    Split an ``arb_droplink`` entry out of a per-side points block.

    ``arb_droplink`` is an axle-only point: the corner class does not accept it,
    so it must be removed before the block is handed to ``parse_hardpoints``.
    Returns ``(corner_points_without_arb, arb_droplink_raw_or_None)``. The lookup
    is case-insensitive to match the rest of the loader.
    """
    if not isinstance(points_raw, dict):
        return points_raw, None
    corner_points = {}
    arb_raw = None
    for key, value in points_raw.items():
        if isinstance(key, str) and key.lower() == PointID.ARB_DROPLINK.name.lower():
            arb_raw = value
        else:
            corner_points[key] = value
    return corner_points, arb_raw


def _parse_center_points(center_raw: Any) -> dict[PointID, Point3]:
    """
    Parse the shared, non-mirrored ``center`` block (ARB axis points).

    Accepts ``arb_axis_a`` and ``arb_axis_b`` (case-insensitive). Returns an empty
    dict when no center block is given (no ARB). Unknown keys are rejected so that
    typos surface at load time.
    """
    from kinematics.io.validation import coerce_point3

    if center_raw is None:
        return {}
    if not isinstance(center_raw, dict):
        raise ValueError("Axle 'center' block must be a mapping")

    valid = {
        PointID.ARB_AXIS_A.name.lower(): PointID.ARB_AXIS_A,
        PointID.ARB_AXIS_B.name.lower(): PointID.ARB_AXIS_B,
    }
    result: dict[PointID, Point3] = {}
    for key, value in center_raw.items():
        pid = valid.get(str(key).lower())
        if pid is None:
            raise ValueError(
                f"Unknown center point '{key}' (expected arb_axis_a/arb_axis_b)."
            )
        try:
            result[pid] = coerce_point3(value)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"center '{key}': {e}") from e
    return result


def _parse_mirror_hardpoints(
    raw_hardpoints: dict[str, Any],
    parse_hardpoints: Any,
) -> tuple[dict[Side, dict[PointKey, Point3]], dict[Side, Point3]]:
    """Parse mirror-mode hardpoints: one side given, the other generated."""
    from kinematics.io.validation import coerce_enum, coerce_point3

    points_raw = raw_hardpoints.get("points")
    if points_raw is None:
        raise ValueError(
            "Axle mirror-mode hardpoints require a 'points' block "
            "(or use explicit 'left'/'right' blocks)."
        )

    mirror = raw_hardpoints.get("mirror", True)
    if not mirror:
        raise ValueError(
            "Mirror-mode hardpoints with 'mirror: false' are ambiguous; "
            "give both sides explicitly under 'left' and 'right'."
        )

    source_side_str = raw_hardpoints.get("side", "left")
    source_side = coerce_enum(Side, source_side_str)
    if source_side == Side.CENTER:
        raise ValueError("Mirror source side must be 'left' or 'right'.")

    corner_points, arb_raw = _pop_arb_droplink(points_raw)
    source_hp, errors = parse_hardpoints(corner_points, DoubleWishboneSuspension)
    _require_no_errors(errors, "Axle hardpoint validation failed")
    _validate_side_signs(source_hp, source_side)

    other_side = Side.RIGHT if source_side == Side.LEFT else Side.LEFT
    side_hardpoints = {
        source_side: source_hp,
        other_side: _mirror_hardpoints(source_hp),
    }

    arb_droplinks: dict[Side, Point3] = {}
    if arb_raw is not None:
        source_arb = coerce_point3(arb_raw)
        arb_droplinks[source_side] = source_arb
        arb_droplinks[other_side] = _mirror_point(source_arb)

    return side_hardpoints, arb_droplinks


def _parse_explicit_hardpoints(
    raw_hardpoints: dict[str, Any],
    parse_hardpoints: Any,
) -> tuple[dict[Side, dict[PointKey, Point3]], dict[Side, Point3]]:
    """Parse explicit-mode hardpoints: both 'left' and 'right' given."""
    from kinematics.io.validation import coerce_point3

    if "left" not in raw_hardpoints or "right" not in raw_hardpoints:
        raise ValueError(
            "Explicit-mode axle hardpoints require both 'left' and 'right' blocks."
        )

    side_hardpoints: dict[Side, dict[PointKey, Point3]] = {}
    arb_droplinks: dict[Side, Point3] = {}
    for side, block_key in ((Side.LEFT, "left"), (Side.RIGHT, "right")):
        corner_points, arb_raw = _pop_arb_droplink(raw_hardpoints[block_key])
        hp, errors = parse_hardpoints(corner_points, DoubleWishboneSuspension)
        _require_no_errors(errors, f"Axle '{block_key}' hardpoint validation failed")
        _validate_side_signs(hp, side)
        side_hardpoints[side] = hp
        if arb_raw is not None:
            arb_droplinks[side] = coerce_point3(arb_raw)

    return side_hardpoints, arb_droplinks
