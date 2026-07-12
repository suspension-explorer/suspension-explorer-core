"""Structured units for metric values and derivative metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricUnit(Enum):
    """Supported scalar metric units."""

    MM = "mm"
    DEG = "deg"
    PERCENT = "%"

    @property
    def symbol(self) -> str:
        """Return the export symbol for this unit."""
        return self.value

    def __truediv__(self, denominator: "MetricUnit") -> "MetricUnitQuotient":
        """Compose a derivative quotient unit."""
        if not isinstance(denominator, MetricUnit):
            return NotImplemented
        return MetricUnitQuotient(self, denominator)


@dataclass(frozen=True)
class MetricUnitQuotient:
    """Unit metadata for one scalar metric differentiated by another."""

    numerator: MetricUnit
    denominator: MetricUnit

    @property
    def symbol(self) -> str:
        """Return the export symbol for this quotient."""
        return f"{self.numerator.symbol}/{self.denominator.symbol}"

    def __str__(self) -> str:
        """Render the quotient for output metadata."""
        return self.symbol
