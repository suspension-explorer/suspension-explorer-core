"""Generic declarative derivatives of scalar geometric responses."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.dual import DualScalar, DualVec3, dot, norm
from kinematics.core.enums import Axis
from kinematics.core.geometry import Direction3, extract_array
from kinematics.core.point_ref import PointKey
from kinematics.metrics.units import MetricUnit, MetricUnitQuotient
from kinematics.sensitivity import TangentField
from kinematics.state import SuspensionState

DualPositions = Mapping[PointKey, DualVec3]
DualSafeScalarCallable = Callable[[DualPositions], DualScalar]

_SEMANTIC_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _validate_semantics(name: str, unit: MetricUnit) -> None:
    """Validate universal scalar naming and unit metadata."""
    if _SEMANTIC_NAME.fullmatch(name) is None:
        raise ValueError(f"Scalar name must be lowercase snake-case, got {name!r}")
    if not isinstance(unit, MetricUnit):
        raise TypeError(f"Scalar unit must be a MetricUnit, got {unit!r}")


class ScalarResponse(Protocol):
    """A scalar geometric quantity evaluable on dual-number positions."""

    @property
    def name(self) -> str:
        """Semantic snake-case name used for output columns."""

    @property
    def unit(self) -> MetricUnit:
        """Physical unit of the scalar value."""

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate the response value and directional derivative."""


class ScalarDriver(ScalarResponse, Protocol):
    """A scalar input quantity used as a derivative denominator."""

    @property
    def selector_point(self) -> PointKey | None:
        """Point target used to select candidate tangent fields."""


@dataclass(frozen=True)
class PointCoordinateResponse:
    """Coordinate of a point along a normalized world or custom axis."""

    point: PointKey
    axis: Direction3
    name: str
    unit: MetricUnit

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey:
        """Select tangents targeting this coordinate's point."""
        return self.point

    @classmethod
    def from_world_axis(
        cls,
        point: PointKey,
        axis: Axis,
        *,
        name: str,
        unit: MetricUnit,
    ) -> "PointCoordinateResponse":
        """Build a coordinate response along a principal world axis."""
        direction = np.zeros(3, dtype=np.float64)
        direction[int(axis)] = 1.0
        return cls(point=point, axis=Direction3(direction), name=name, unit=unit)

    @classmethod
    def from_axis(
        cls,
        point: PointKey,
        axis: Axis | Direction3 | np.ndarray | tuple[float, float, float],
        *,
        name: str,
        unit: MetricUnit,
    ) -> "PointCoordinateResponse":
        """Build a coordinate response, normalizing the supplied axis."""
        if isinstance(axis, Axis):
            return cls.from_world_axis(point, axis, name=name, unit=unit)
        return cls(
            point=point,
            axis=Direction3(extract_array(axis)),
            name=name,
            unit=unit,
        )

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Project the point position onto the configured axis."""
        result = dot(positions[self.point], self.axis.data)
        assert isinstance(result, DualScalar)
        return result


@dataclass(frozen=True)
class PointDistanceResponse:
    """Euclidean distance between two points."""

    point_a: PointKey
    point_b: PointKey
    name: str
    unit: MetricUnit
    driving_point: PointKey | None = None

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey | None:
        """Return the explicitly declared target point when used as a driver."""
        return self.driving_point

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate ``|point_a - point_b|`` and its tangent derivative."""
        separation = positions[self.point_a] - positions[self.point_b]
        if float(np.linalg.norm(separation.val)) < EPS_GEOMETRIC:
            raise ValueError("Point-distance derivative is undefined at zero length")
        return norm(separation)


@dataclass(frozen=True)
class PointDisplacementMagnitudeResponse:
    """Magnitude of one point's displacement from a fixed reference position."""

    point: PointKey
    reference: np.ndarray
    name: str
    unit: MetricUnit

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey:
        """Select tangents targeting the displaced point."""
        return self.point

    @classmethod
    def from_reference(
        cls,
        point: PointKey,
        reference: object,
        *,
        name: str,
        unit: MetricUnit,
    ) -> "PointDisplacementMagnitudeResponse":
        """Build the response with a copied three-component reference."""
        raw_reference = extract_array(reference)
        if raw_reference.shape != (3,):
            raise ValueError(
                f"Displacement reference must have shape (3,), got "
                f"{raw_reference.shape}"
            )
        copied_reference = raw_reference.copy()
        copied_reference.flags.writeable = False
        return cls(
            point=point,
            reference=copied_reference,
            name=name,
            unit=unit,
        )

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate displacement magnitude away from its singular origin."""
        displacement = positions[self.point] - self.reference
        if float(np.linalg.norm(displacement.val)) < EPS_GEOMETRIC:
            raise ValueError(
                "Displacement magnitude derivative is undefined at zero displacement"
            )
        return norm(displacement)


@dataclass(frozen=True)
class CallableScalarResponse:
    """Adapter for an arbitrary dual-safe scalar response callable."""

    function: DualSafeScalarCallable
    name: str
    unit: MetricUnit
    driving_point: PointKey | None = None

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey | None:
        """Return the optional target point when this callable is a driver."""
        return self.driving_point

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate the wrapped callable and require a dual scalar result."""
        result = self.function(positions)
        if not isinstance(result, DualScalar):
            raise TypeError(
                "Dual-safe scalar response must return DualScalar when given "
                "dual positions"
            )
        return result


@dataclass(frozen=True)
class DerivativeMetricDefinition:
    """Declarative derivative ``d(response) / d(driver)``."""

    response: ScalarResponse
    driver: ScalarDriver
    scale: float = 1.0

    @property
    def column_name(self) -> str:
        """Universal output name derived from scalar semantics."""
        return f"deriv_{self.response.name}_wrt_{self.driver.name}"

    @property
    def unit(self) -> MetricUnitQuotient:
        """Universal quotient unit derived from scalar semantics."""
        return self.response.unit / self.driver.unit

    def evaluate(self, state: SuspensionState, tangent: TangentField) -> float:
        """Evaluate this derivative along one solution-manifold tangent."""
        positions = _dual_positions(state, tangent)
        response_rate = self.response.evaluate(positions).deriv
        driver_rate = self.driver.evaluate(positions).deriv
        if abs(driver_rate) < EPS_GEOMETRIC:
            raise ValueError("Cannot evaluate derivative for a zero-rate driver")
        return float(self.scale * response_rate / driver_rate)

    def select_tangent(
        self,
        state: SuspensionState,
        tangents: Sequence[TangentField],
    ) -> TangentField | None:
        """Select the matching tangent with the strongest nonzero driver rate."""
        selector_point = self.driver.selector_point
        if selector_point is None:
            raise ValueError(
                "Scalar driver requires an explicit driving point for tangent selection"
            )

        strongest: TangentField | None = None
        strongest_rate = 0.0
        tied = False
        for tangent in tangents:
            if tangent.target.point_id != selector_point:
                continue
            positions = _dual_positions(state, tangent)
            driver_rate = abs(self.driver.evaluate(positions).deriv)
            if driver_rate > strongest_rate + EPS_GEOMETRIC:
                strongest = tangent
                strongest_rate = driver_rate
                tied = False
            elif (
                driver_rate >= EPS_GEOMETRIC
                and abs(driver_rate - strongest_rate) <= EPS_GEOMETRIC
            ):
                tied = True
        if strongest_rate < EPS_GEOMETRIC:
            return None
        if tied:
            raise ValueError(
                f"Ambiguous derivative driver for column '{self.column_name}': "
                "multiple matching tangents have equal strength"
            )
        return strongest

    def evaluate_from_tangents(
        self,
        state: SuspensionState,
        tangents: Sequence[TangentField],
    ) -> float | None:
        """Select a driver tangent and evaluate, or return None if absent."""
        tangent = self.select_tangent(state, tangents)
        if tangent is None:
            return None
        return self.evaluate(state, tangent)


DerivativeMetricRow = OrderedDict[str, float | None]


def _dual_positions(
    state: SuspensionState,
    tangent: TangentField,
) -> dict[PointKey, DualVec3]:
    """Seed a state along one tangent field."""
    from kinematics.core.dual import seed_positions_with_tangent

    return seed_positions_with_tangent(state.positions, tangent.velocities)


def evaluate_derivative_metrics(
    definitions: Sequence[DerivativeMetricDefinition],
    state: SuspensionState,
    tangents: Sequence[TangentField],
) -> DerivativeMetricRow:
    """Evaluate declarations in order without topology-specific dispatch."""
    row: DerivativeMetricRow = OrderedDict()
    for definition in definitions:
        if definition.column_name in row:
            raise ValueError(
                f"Duplicate derivative metric column: {definition.column_name}"
            )
        row[definition.column_name] = definition.evaluate_from_tangents(
            state,
            tangents,
        )
    return row
