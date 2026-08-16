"""Generic declarative derivatives of scalar geometric responses.

Positions and custom axes are represented in chassis coordinates. Each scalar
response declares its own chassis, wheel, or road reference; differentiating
it does not change that reference system or introduce world-space dependence.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from kinematics.core.enums import Axis
from kinematics.core.metrics.units import MetricUnit, MetricUnitQuotient
from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.dual import DualScalar, DualVec3, dot, norm
from kinematics.core.primitives.geometry import Direction3, extract_array
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.sensitivity import TangentField
from kinematics.core.state import SuspensionState

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
    """A scalar quantity whose implementation defines its reference system."""

    @property
    def name(self) -> str:
        """Semantic snake-case name used for output columns."""

    @property
    def unit(self) -> MetricUnit:
        """Physical unit of the scalar value."""

    @property
    def label(self) -> str | None:
        """Optional explicitly authored display label."""

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate the value and derivative in the response's declared system."""


class ScalarDriver(ScalarResponse, Protocol):
    """A scalar input quantity used as a derivative denominator."""

    @property
    def selector_point(self) -> PointKey | None:
        """Point target used to select candidate tangent fields."""


@dataclass(frozen=True)
class PointCoordinateResponse:
    """Point coordinate along an axis expressed in chassis coordinates."""

    point: PointKey
    axis: Direction3
    name: str
    unit: MetricUnit
    label: str | None = None

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey:
        """Select tangents targeting this coordinate's point."""
        return self.point

    @classmethod
    def from_chassis_axis(
        cls,
        point: PointKey,
        axis: Axis,
        *,
        name: str,
        unit: MetricUnit,
        label: str | None = None,
    ) -> "PointCoordinateResponse":
        """Build a coordinate response along a principal chassis axis."""
        direction = np.zeros(3, dtype=np.float64)
        direction[int(axis)] = 1.0
        return cls(
            point=point,
            axis=Direction3(direction),
            name=name,
            unit=unit,
            label=label,
        )

    @classmethod
    def from_axis(
        cls,
        point: PointKey,
        axis: Axis | Direction3 | np.ndarray | tuple[float, float, float],
        *,
        name: str,
        unit: MetricUnit,
        label: str | None = None,
    ) -> "PointCoordinateResponse":
        """Build a coordinate response, normalizing the supplied axis."""
        if isinstance(axis, Axis):
            return cls.from_chassis_axis(
                point,
                axis,
                name=name,
                unit=unit,
                label=label,
            )
        return cls(
            point=point,
            axis=Direction3(extract_array(axis)),
            name=name,
            unit=unit,
            label=label,
        )

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Project a chassis-space point onto the configured chassis-space axis."""
        result = dot(positions[self.point], self.axis.data)
        assert isinstance(result, DualScalar)
        return result


@dataclass(frozen=True)
class PointDistanceResponse:
    """Euclidean distance between two chassis-coordinate points.

    The result is invariant under a rigid transformation to world space and
    does not reference the road plane.
    """

    point_a: PointKey
    point_b: PointKey
    name: str
    unit: MetricUnit
    driving_point: PointKey | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey | None:
        """Return the explicitly declared target point when used as a driver."""
        return self.driving_point

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate chassis-space separation and its tangent derivative."""
        separation = positions[self.point_a] - positions[self.point_b]
        if float(np.linalg.norm(separation.val)) < EPS_GEOMETRIC:
            raise ValueError("Point-distance derivative is undefined at zero length")
        return norm(separation)


@dataclass(frozen=True)
class PointDisplacementMagnitudeResponse:
    """Chassis-space displacement magnitude from a fixed reference position.

    The Euclidean magnitude is invariant under a common rigid world transform
    and has no road-plane reference.
    """

    point: PointKey
    reference: np.ndarray
    name: str
    unit: MetricUnit
    label: str | None = None

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
        label: str | None = None,
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
            label=label,
        )

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate chassis-space displacement away from its singular origin."""
        displacement = positions[self.point] - self.reference
        if float(np.linalg.norm(displacement.val)) < EPS_GEOMETRIC:
            raise ValueError(
                "Displacement magnitude derivative is undefined at zero displacement"
            )
        return norm(displacement)


@dataclass(frozen=True)
class CallableScalarResponse:
    """Adapter preserving the wrapped scalar's declared reference system."""

    function: DualSafeScalarCallable
    name: str
    unit: MetricUnit
    driving_point: PointKey | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        _validate_semantics(self.name, self.unit)

    @property
    def selector_point(self) -> PointKey | None:
        """Return the optional target point when this callable is a driver."""
        return self.driving_point

    def evaluate(self, positions: DualPositions) -> DualScalar:
        """Evaluate the wrapped callable without changing its reference system."""
        result = self.function(positions)
        if not isinstance(result, DualScalar):
            raise TypeError(
                "Dual-safe scalar response must return DualScalar when given "
                "dual positions"
            )
        return result


@dataclass(frozen=True)
class DerivativeMetricDefinition:
    """Declarative ``d(response) / d(driver)`` in the scalars' own systems.

    The response and driver may use different declared references, such as a
    chassis-space alignment angle with respect to chassis Z travel. The
    derivative does not transform either quantity into road or world space.
    """

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
        """Evaluate along a chassis-space solution-manifold tangent.

        The response and driver retain the reference systems declared by their
        implementations; no road or world transform is applied here.
        """
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
            if tangent.target.selector_point != selector_point:
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
        """Evaluate in the response and driver's declared reference systems."""
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
    from kinematics.core.primitives.dual import seed_positions_with_tangent

    return seed_positions_with_tangent(state.positions, tangent.rates)


def evaluate_derivative_metrics(
    definitions: Sequence[DerivativeMetricDefinition],
    state: SuspensionState,
    tangents: Sequence[TangentField],
) -> DerivativeMetricRow:
    """Evaluate declarations without changing their reference systems.

    Tangent rates and solved points are expressed in chassis coordinates.
    Each response and driver decides whether it further resolves geometry into
    wheel or local road axes; this dispatcher never introduces world space.
    """
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
