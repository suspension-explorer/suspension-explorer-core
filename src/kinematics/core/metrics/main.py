"""
Metrics public API.

Provides the top-level entry points for computing post-solve kinematic
metrics. Returns ordered mappings ready for direct export integration.

Solved geometry remains in chassis coordinates. Axle metrics share an
ISO 8855 style local or equivalent road plane reconstructed from the two tyre
wheel contact centres; standalone corners use a level local plane through their
contact centre. Metric evaluation does not use world-space vehicle placement.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, cast, overload

from kinematics.core.enums import PointID
from kinematics.core.metrics.axle_metrics import append_axle_state_metrics
from kinematics.core.metrics.catalog import (
    get_default_corner_derivative_metrics,
    get_default_corner_metrics,
    get_virtual_steering_metrics,
    physical_steering_metric_keys,
)
from kinematics.core.metrics.context import MetricContext
from kinematics.core.metrics.derivatives import evaluate_derivative_metrics
from kinematics.core.metrics.registry import flat_key
from kinematics.core.metrics.steering import SteeringAxis
from kinematics.core.primitives.point_ref import PointKey, PointRef, Side
from kinematics.core.road import RoadPlane
from kinematics.core.schema.config import SuspensionConfig
from kinematics.core.sensitivity import TangentField
from kinematics.core.state import SuspensionState

if TYPE_CHECKING:
    from kinematics.core.coordinates import ActuatorCoordinate
    from kinematics.core.steering_axis import SteeringResponseAxisResult
    from kinematics.core.suspensions.axle import AxleSuspension
    from kinematics.core.suspensions.base import Suspension
    from kinematics.core.suspensions.corner.base import CornerSuspension


MetricRow = OrderedDict[str, float | None]


@dataclass(frozen=True)
class AxleMetricRows:
    """Location-independent axle metrics plus one typed row per corner."""

    axle: MetricRow
    corners: dict[Side, MetricRow]

    def flat_row(self) -> MetricRow:
        """Render the structured rows for a flat export boundary."""
        return flatten_metric_rows(self.axle, self.corners)


def flatten_metric_rows(
    metrics: MetricRow,
    corner_metrics: Mapping[Side, MetricRow],
) -> MetricRow:
    """Flatten structural metric locations using side suffixes."""
    flat: MetricRow = OrderedDict()
    for side, row in corner_metrics.items():
        for key, value in row.items():
            flat[flat_key(key, side.name.lower())] = value
    flat.update(metrics)
    return flat


def compute_metrics_for_axle_state(
    state: SuspensionState,
    axle: AxleSuspension,
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
    steering_response_axes: "Sequence[SteeringResponseAxisResult] | None" = None,
) -> AxleMetricRows:
    """Compute corner and axle metrics against one axle-local road plane.

    The plane is reconstructed from the two wheel contact centres and expressed
    in chassis coordinates. It follows the ISO 8855 local or equivalent
    road-plane concept. Individual metric docstrings state whether they resolve
    values in chassis, road, or tyre axes. World-space presentation does not
    participate in metric calculation.
    """
    axle_row: MetricRow = OrderedDict()
    corner_rows: dict[Side, MetricRow] = {}
    road = RoadPlane.from_axle_contact_centres(
        state.get(PointRef(Side.LEFT, PointID.WHEEL_CONTACT_CENTRE)),
        state.get(PointRef(Side.RIGHT, PointID.WHEEL_CONTACT_CENTRE)),
    )
    steering_axes_by_side = _steering_axes_by_side(steering_response_axes)
    for side in (Side.LEFT, Side.RIGHT):
        corner = axle.corners[side]
        corner_state = axle.corner_state(state, side)
        corner_config = corner.config if corner.config is not None else config
        side_row = compute_metrics_for_state(
            corner_state,
            corner,
            corner_config,
            _corner_tangents(
                tangents,
                side,
                axle.required_actuator_coordinates(),
            )
            if tangents
            else None,
            road=road,
            steering_response_axis=steering_axes_by_side.get(side),
        )
        corner_rows[side] = side_row

    append_axle_state_metrics(
        axle_row,
        state,
        axle,
        road,
        axle.design_road_plane,
    )
    topology_rows = axle.topology_metric_rows(state)
    axle_row.update(topology_rows.axle)
    for side, row in topology_rows.corners.items():
        corner_rows[side].update(row)
    if tangents:
        axle_row.update(
            evaluate_derivative_metrics(
                axle.derivative_metric_definitions(),
                state,
                tangents,
            )
        )
    return AxleMetricRows(axle=axle_row, corners=corner_rows)


def _steering_axes_by_side(
    results: "Sequence[SteeringResponseAxisResult] | None",
) -> dict[Side, SteeringResponseAxisResult | None]:
    """Map axle upright results structurally through side-qualified point keys."""
    grouped: dict[Side, list[SteeringResponseAxisResult]] = {}
    for result in results or ():
        sides = {key.side for key in result.point_keys if isinstance(key, PointRef)}
        if len(sides) == 1:
            side = next(iter(sides))
            grouped.setdefault(side, []).append(result)
    return {
        side: side_results[0] if len(side_results) == 1 else None
        for side, side_results in grouped.items()
    }


def _corner_tangents(
    tangents: "Sequence[TangentField]",
    side: Side,
    required_actuator_coordinates: "Sequence[ActuatorCoordinate]",
) -> list["TangentField"]:
    """Project axle tangents into one corner, including shared actuators."""
    result: list[TangentField] = []
    for tangent in tangents:
        selector_point = tangent.target.coordinate.selector_point
        if selector_point is not None:
            local_target = _local_tangent_target(
                selector_point,
                side,
                required_actuator_coordinates,
            )
            if local_target is None:
                continue
            local_tangent_target = tangent.target.map_points(
                lambda _point: local_target
            )
        else:
            if not tangent.target.coordinate.required_points or not all(
                isinstance(point, PointRef) and point.side is side
                for point in tangent.target.coordinate.required_points
            ):
                continue
            local_tangent_target = tangent.target.map_points(
                lambda point: point.point if isinstance(point, PointRef) else point
            )
        result.append(
            TangentField(
                target_index=tangent.target_index,
                target=local_tangent_target,
                rates={
                    key.point: rate
                    for key, rate in tangent.rates.items()
                    if isinstance(key, PointRef) and key.side is side
                },
            )
        )
    return result


def _local_tangent_target(
    target_key: PointKey,
    side: Side,
    required_actuator_coordinates: "Sequence[ActuatorCoordinate]",
) -> PointID | None:
    """Resolve a side-local target or its equivalent shared actuator point."""
    if isinstance(target_key, PointRef) and target_key.side is side:
        return target_key.point
    for actuator in required_actuator_coordinates:
        if target_key not in actuator.point_keys:
            continue
        for point_id in actuator.point_keys:
            if isinstance(point_id, PointRef) and point_id.side is side:
                return point_id.point
    return None


def compute_metrics_for_state(
    state: SuspensionState,
    suspension: "CornerSuspension",
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
    *,
    road: RoadPlane | None = None,
    steering_response_axis: "SteeringResponseAxisResult | None" = None,
) -> MetricRow:
    """
    Compute all corner-level metrics for a single solved state.

    Solved positions and directions remain in chassis coordinates. When
    supplied, ``road`` is the axle's shared ISO-style local road plane,
    expressed in that same basis. A standalone corner instead receives a level
    road plane through its wheel contact centre. Individual calculations may
    use chassis, road, or tyre axes as documented, but none uses world space.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The corner suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.
        tangents: Optional solution-manifold tangents. Derivative columns are
            appended only when these are supplied.
        road: Optional shared axle road plane in chassis coordinates.
            Standalone corner callers omit this and use the horizontal plane
            through their wheel contact centre.
        steering_response_axis: Optional isolated steering-response result.
            A valid result supplies the additive virtual steering metrics;
            an invalid or unavailable result leaves those values undefined.

    Returns:
        An ordered mapping of metric column names to values. Values are
        None when the underlying geometry is undefined (e.g. parallel
        links producing an IC at infinity).
    """
    motion_axis = steering_response_axis.axis if steering_response_axis else None
    virtual_axis = (
        None
        if motion_axis is None
        else SteeringAxis.from_unoriented_line(
            motion_axis.point,
            motion_axis.direction,
        )
    )
    ctx = MetricContext(
        state=state,
        suspension=suspension,
        config=config,
        road=road,
    )

    catalog = get_default_corner_metrics()
    virtual_catalog = (
        get_virtual_steering_metrics()
        if suspension.steering_actuator_coordinate() is not None
        else ()
    )
    virtual_by_physical_key = {
        metric.column_name.removesuffix("_virtual"): metric
        for metric in virtual_catalog
    }
    virtual_ctx = (
        None
        if virtual_axis is None
        else MetricContext(
            state=state,
            suspension=suspension,
            config=config,
            road=ctx.road,
            steering_axis=virtual_axis,
        )
    )
    # An architecture without a physical steering axis omits the physical
    # steering columns entirely; its steering geometry reports only through
    # the motion-derived ``*_virtual`` family.
    omitted_physical_keys = (
        physical_steering_metric_keys()
        if suspension.steering_axis_points() is None
        else frozenset()
    )
    row: MetricRow = OrderedDict()
    for metric in catalog:
        if metric.column_name not in omitted_physical_keys:
            row[metric.column_name] = metric.compute(ctx)
        virtual_metric = virtual_by_physical_key.get(metric.column_name)
        if virtual_metric is not None:
            row[virtual_metric.column_name] = (
                None if virtual_ctx is None else virtual_metric.compute(virtual_ctx)
            )
    row.update(suspension.topology_metric_values(state))
    if tangents:
        definitions = (
            *get_default_corner_derivative_metrics(suspension),
            *suspension.derivative_metric_definitions(),
        )
        row.update(evaluate_derivative_metrics(definitions, state, tangents))
    return row


@overload
def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "AxleSuspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[AxleMetricRows]: ...


@overload
def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
) -> list[MetricRow]: ...


def compute_metrics_for_sweep(
    states: list[SuspensionState],
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents_per_state: "Sequence[Sequence[TangentField]] | None" = None,
    # The overloads narrow the element type per suspension kind. The invariant
    # list return of each overload is only assignable to a covariant Sequence
    # here, so the implementation widens to Sequence.
) -> Sequence[MetricRow | AxleMetricRows]:
    """
    Compute metrics for a sweep of solved corner or axle states.

    Reference systems are selected independently for each state using the same
    rules as :func:`compute_metrics_for_state` and
    :func:`compute_metrics_for_axle_state`. No chassis-to-world transform is
    introduced by sweep evaluation.

    Args:
        states: List of solved SuspensionStates from a parametric sweep.
        suspension: The suspension instance for type-specific geometry.
        config: Suspension configuration with vehicle parameters.
        tangents_per_state: Optional tangents aligned one-to-one with
            ``states``. Callers that already have only solved states retain
            the historical non-derivative result; high-level sweep consumers
            should use :func:`kinematics.core.sweep.compute_sweep_metrics`.

    Returns:
        One metric result per state. Corner suspensions return ordered rows;
        axle suspensions return structural axle and per-corner rows.
    """
    if tangents_per_state is not None and len(states) != len(tangents_per_state):
        raise ValueError("State/tangent row count mismatch")

    if suspension.is_axle:
        axle = cast("AxleSuspension", suspension)
        return [
            compute_metrics_for_axle_state(
                state,
                axle,
                config,
                tangents_per_state[index] if tangents_per_state is not None else None,
            )
            for index, state in enumerate(states)
        ]

    corner = cast("CornerSuspension", suspension)
    return [
        compute_metrics_for_state(
            state,
            corner,
            config,
            tangents_per_state[index] if tangents_per_state is not None else None,
        )
        for index, state in enumerate(states)
    ]


def _compute_metrics_for_suspension_state(
    state: SuspensionState,
    suspension: "Suspension",
    config: SuspensionConfig,
    tangents: "Sequence[TangentField] | None" = None,
) -> MetricRow | AxleMetricRows:
    """Dispatch calculation while preserving each metric's reference system."""
    if suspension.is_axle:
        axle = cast("AxleSuspension", suspension)
        return compute_metrics_for_axle_state(state, axle, config, tangents)
    corner = cast("CornerSuspension", suspension)
    return compute_metrics_for_state(state, corner, config, tangents)


@overload
def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "AxleSuspension",
) -> AxleMetricRows: ...


@overload
def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> MetricRow: ...


def compute_metrics_for_state_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> MetricRow | AxleMetricRows:
    """
    Compute metrics using parameters from the suspension configuration.

    Convenience wrapper that extracts config from the suspension instance.
    It delegates unchanged to the chassis and local-road calculations described
    by :func:`compute_metrics_for_state` and
    :func:`compute_metrics_for_axle_state`; it does not use world space.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension containing configuration.

    Returns:
        An ordered row for a corner suspension, or structural axle and
        per-corner rows for an axle suspension.

    Raises:
        ValueError: If the suspension has no configuration.
    """
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    return _compute_metrics_for_suspension_state(
        state,
        suspension,
        suspension.config,
    )
