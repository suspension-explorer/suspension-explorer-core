"""
Forward-mode automatic differentiation via dual numbers.

Provides DualScalar and DualVec3 types that propagate exact first-order
derivatives through arithmetic and key numpy operations (dot, norm).
The numpy __array_function__ protocol ensures that existing code using
np.dot and np.linalg.norm works unmodified with dual-number inputs.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, TypeVar, overload

import numpy as np

from kinematics.core.geometry import (
    ArrayBacked,
    extract_array,
    try_extract_array,
)
from kinematics.core.point_ref import PointKey


class DualScalar:
    """
    A scalar value paired with its first derivative.

    Supports arithmetic with other DualScalar, float, int, and DualVec3
    (scalar * vector). Comparisons use .val only, for epsilon guards in
    normalize_vector and similar functions.
    """

    __slots__ = ("val", "deriv")

    def __init__(self, val: float, deriv: float = 0.0):
        self.val = float(val)
        self.deriv = float(deriv)

    @overload
    def __add__(self, other: DualScalar) -> DualScalar: ...
    @overload
    def __add__(self, other: int | float) -> DualScalar: ...

    def __add__(self, other: DualScalar | int | float) -> DualScalar:
        if isinstance(other, DualScalar):
            return DualScalar(self.val + other.val, self.deriv + other.deriv)
        if isinstance(other, (int, float)):
            return DualScalar(self.val + other, self.deriv)
        return NotImplemented  # type: ignore[return-value]

    def __radd__(self, other: int | float) -> DualScalar:
        if isinstance(other, (int, float)):
            return DualScalar(other + self.val, self.deriv)
        return NotImplemented  # type: ignore[return-value]

    @overload
    def __sub__(self, other: DualScalar) -> DualScalar: ...
    @overload
    def __sub__(self, other: int | float) -> DualScalar: ...

    def __sub__(self, other: DualScalar | int | float) -> DualScalar:
        if isinstance(other, DualScalar):
            return DualScalar(self.val - other.val, self.deriv - other.deriv)
        if isinstance(other, (int, float)):
            return DualScalar(self.val - other, self.deriv)
        return NotImplemented  # type: ignore[return-value]

    def __rsub__(self, other: int | float) -> DualScalar:
        if isinstance(other, (int, float)):
            return DualScalar(other - self.val, -self.deriv)
        return NotImplemented  # type: ignore[return-value]

    @overload
    def __mul__(self, other: DualScalar) -> DualScalar: ...
    @overload
    def __mul__(self, other: int | float) -> DualScalar: ...
    @overload
    def __mul__(self, other: DualVec3) -> DualVec3: ...
    @overload
    def __mul__(self, other: np.ndarray) -> DualVec3: ...

    def __mul__(
        self, other: DualScalar | int | float | DualVec3 | np.ndarray
    ) -> DualScalar | DualVec3:
        if isinstance(other, DualScalar):
            # Product rule: d(a*b) = a'*b + a*b'
            return DualScalar(
                self.val * other.val,
                self.deriv * other.val + self.val * other.deriv,
            )
        if isinstance(other, (int, float)):
            return DualScalar(self.val * other, self.deriv * other)
        if isinstance(other, DualVec3):
            # Scalar * vector -> DualVec3 with product rule.
            return DualVec3(
                self.val * other.val,
                self.deriv * other.val + self.val * other.deriv,
            )
        if isinstance(other, np.ndarray):
            # DualScalar * plain ndarray: derivative only from scalar.
            return DualVec3(self.val * other, self.deriv * other)
        # Handle geometry types (Point3, Vector3, Direction3) by extracting
        # the raw array -- they are constants in the dual sense.
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(self.val * arr, self.deriv * arr)
        return NotImplemented  # type: ignore[return-value]

    @overload
    def __rmul__(self, other: int | float) -> DualScalar: ...
    @overload
    def __rmul__(self, other: np.ndarray) -> DualVec3: ...

    def __rmul__(self, other: int | float | np.ndarray) -> DualScalar | DualVec3:
        if isinstance(other, (int, float)):
            return DualScalar(other * self.val, other * self.deriv)
        if isinstance(other, np.ndarray):
            return DualVec3(other * self.val, other * self.deriv)
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(arr * self.val, arr * self.deriv)
        return NotImplemented  # type: ignore[return-value]

    @overload
    def __truediv__(self, other: DualScalar) -> DualScalar: ...
    @overload
    def __truediv__(self, other: int | float) -> DualScalar: ...

    def __truediv__(self, other: DualScalar | int | float) -> DualScalar:
        if isinstance(other, DualScalar):
            # Quotient rule: d(a/b) = (a'*b - a*b') / b^2
            return DualScalar(
                self.val / other.val,
                (self.deriv * other.val - self.val * other.deriv) / (other.val**2),
            )
        if isinstance(other, (int, float)):
            return DualScalar(self.val / other, self.deriv / other)
        return NotImplemented  # type: ignore[return-value]

    def __rtruediv__(self, other: int | float) -> DualScalar:
        if isinstance(other, (int, float)):
            # d(c/a) = -c * a' / a^2
            return DualScalar(
                other / self.val,
                -other * self.deriv / (self.val**2),
            )
        return NotImplemented  # type: ignore[return-value]

    def __neg__(self) -> DualScalar:
        return DualScalar(-self.val, -self.deriv)

    def __abs__(self) -> DualScalar:
        if self.val >= 0:
            return DualScalar(self.val, self.deriv)
        return DualScalar(-self.val, -self.deriv)

    def __lt__(self, other: DualScalar | float) -> bool:
        if isinstance(other, DualScalar):
            return self.val < other.val
        return self.val < other

    def __le__(self, other: DualScalar | float) -> bool:
        if isinstance(other, DualScalar):
            return self.val <= other.val
        return self.val <= other

    def __gt__(self, other: DualScalar | float) -> bool:
        if isinstance(other, DualScalar):
            return self.val > other.val
        return self.val > other

    def __ge__(self, other: DualScalar | float) -> bool:
        if isinstance(other, DualScalar):
            return self.val >= other.val
        return self.val >= other

    def __float__(self) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"DualScalar(val={self.val}, deriv={self.deriv})"


class DualVec3:
    """
    A 3D vector paired with its element-wise first derivatives.

    val and deriv are both ndarray of shape (3,). Supports arithmetic
    with other DualVec3, DualScalar, float, int, and ndarray operands.

    Dispatches np.dot and np.linalg.norm to dual-aware implementations
    via numpy's __array_function__ protocol.
    """

    __slots__ = ("val", "deriv")

    def __init__(self, val: np.ndarray, deriv: np.ndarray | None = None):
        self.val = np.asarray(val, dtype=np.float64)
        if deriv is None:
            self.deriv = np.zeros(3, dtype=np.float64)
        else:
            self.deriv = np.asarray(deriv, dtype=np.float64)

    def __add__(self, other: DualVec3 | np.ndarray) -> DualVec3:
        if isinstance(other, DualVec3):
            return DualVec3(self.val + other.val, self.deriv + other.deriv)
        if isinstance(other, np.ndarray):
            return DualVec3(self.val + other, self.deriv.copy())
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(self.val + arr, self.deriv.copy())
        return NotImplemented  # type: ignore[return-value]

    def __radd__(self, other: np.ndarray) -> DualVec3:
        if isinstance(other, np.ndarray):
            return DualVec3(other + self.val, self.deriv.copy())
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(arr + self.val, self.deriv.copy())
        return NotImplemented  # type: ignore[return-value]

    def __sub__(self, other: DualVec3 | np.ndarray) -> DualVec3:
        if isinstance(other, DualVec3):
            return DualVec3(self.val - other.val, self.deriv - other.deriv)
        if isinstance(other, np.ndarray):
            return DualVec3(self.val - other, self.deriv.copy())
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(self.val - arr, self.deriv.copy())
        return NotImplemented  # type: ignore[return-value]

    def __rsub__(self, other: np.ndarray) -> DualVec3:
        if isinstance(other, np.ndarray):
            return DualVec3(other - self.val, -self.deriv)
        arr = try_extract_array(other)
        if arr is not None:
            return DualVec3(arr - self.val, -self.deriv)
        return NotImplemented  # type: ignore[return-value]

    def __mul__(self, other: DualScalar | int | float | DualVec3) -> DualVec3:
        if isinstance(other, DualScalar):
            # Product rule: d(vec * s) = vec' * s + vec * s'
            return DualVec3(
                self.val * other.val,
                self.deriv * other.val + self.val * other.deriv,
            )
        if isinstance(other, (int, float)):
            return DualVec3(self.val * other, self.deriv * other)
        if isinstance(other, DualVec3):
            # Element-wise product rule (used by broadcasting, not dot).
            return DualVec3(
                self.val * other.val,
                self.deriv * other.val + self.val * other.deriv,
            )
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: DualScalar | int | float) -> DualVec3:
        if isinstance(other, DualScalar):
            return DualVec3(
                other.val * self.val,
                other.deriv * self.val + other.val * self.deriv,
            )
        if isinstance(other, (int, float)):
            return DualVec3(other * self.val, other * self.deriv)
        return NotImplemented  # type: ignore[return-value]

    def __truediv__(self, other: DualScalar | int | float) -> DualVec3:
        if isinstance(other, DualScalar):
            # Quotient rule: d(vec/s) = (vec'*s - vec*s') / s^2
            return DualVec3(
                self.val / other.val,
                (self.deriv * other.val - self.val * other.deriv) / (other.val**2),
            )
        if isinstance(other, (int, float)):
            return DualVec3(self.val / other, self.deriv / other)
        return NotImplemented  # type: ignore[return-value]

    def __neg__(self) -> DualVec3:
        return DualVec3(-self.val, -self.deriv)

    def __getitem__(self, idx: int) -> DualScalar:
        # idx may be an int or an IntEnum (e.g. Axis.Z).
        i = int(idx)
        return DualScalar(self.val[i], self.deriv[i])

    def __array_function__(self, func, types, args, kwargs):
        impl = _dual_implementation(func)
        if impl is not None:
            return impl(*args, **kwargs)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Handle numpy ufuncs that arise from ndarray op DualVec3 expressions.
        # Without this, numpy tries to broadcast DualVec3 into an ndarray.
        # Also handles geometry types (Point3/Vector3/Direction3) by extracting
        # their raw arrays.
        if method != "__call__" or len(inputs) != 2:
            return NotImplemented
        a, b = inputs
        # Normalize non-dual, non-ndarray operands to raw arrays.
        if not isinstance(a, (DualVec3, np.ndarray)):
            a = try_extract_array(a)
        if not isinstance(b, (DualVec3, np.ndarray)):
            b = try_extract_array(b)
        if ufunc is np.add:
            if isinstance(a, np.ndarray) and isinstance(b, DualVec3):
                return DualVec3(a + b.val, b.deriv.copy())
            if isinstance(a, DualVec3) and isinstance(b, np.ndarray):
                return DualVec3(a.val + b, a.deriv.copy())
        if ufunc is np.subtract:
            if isinstance(a, np.ndarray) and isinstance(b, DualVec3):
                return DualVec3(a - b.val, -b.deriv)
            if isinstance(a, DualVec3) and isinstance(b, np.ndarray):
                return DualVec3(a.val - b, a.deriv.copy())
        if ufunc is np.multiply:
            if isinstance(a, np.ndarray) and isinstance(b, DualVec3):
                return DualVec3(a * b.val, a * b.deriv)
            if isinstance(a, DualVec3) and isinstance(b, np.ndarray):
                return DualVec3(a.val * b, a.deriv * b)
        return NotImplemented

    def __repr__(self) -> str:
        return f"DualVec3(val={self.val}, deriv={self.deriv})"


def _dual_dot(a, b, out=None):
    """
    Dual-aware dot product.

    d(dot(a, b)) = dot(a', b) + dot(a, b')

    Handles geometry types (Point3/Vector3/Direction3) by extracting
    their raw arrays and treating them as constants (zero derivative).
    """
    if isinstance(a, DualVec3):
        a_val, a_deriv = a.val, a.deriv
    else:
        a_val = extract_array(a)
        a_deriv = np.zeros(3, dtype=np.float64)

    if isinstance(b, DualVec3):
        b_val, b_deriv = b.val, b.deriv
    else:
        b_val = extract_array(b)
        b_deriv = np.zeros(3, dtype=np.float64)

    val = np.dot(a_val, b_val)
    deriv = np.dot(a_deriv, b_val) + np.dot(a_val, b_deriv)
    return DualScalar(float(val), float(deriv))


def _dual_cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Dual-aware cross product.

    d(cross(a, b)) = cross(a', b) + cross(a, b')
    """
    if isinstance(a, DualVec3):
        a_val, a_deriv = a.val, a.deriv
    else:
        a_val = extract_array(a)
        a_deriv = np.zeros(3, dtype=np.float64)

    if isinstance(b, DualVec3):
        b_val, b_deriv = b.val, b.deriv
    else:
        b_val = extract_array(b)
        b_deriv = np.zeros(3, dtype=np.float64)

    # Cross is bilinear, so its derivative follows the product rule.
    val = np.cross(a_val, b_val)
    deriv = np.cross(a_deriv, b_val) + np.cross(a_val, b_deriv)
    return DualVec3(val, deriv)


def _dual_norm(x, ord=None, axis=None, keepdims=False):
    """
    Dual-aware vector norm (L2 only).

    d(||v||) = dot(v, v') / ||v||
    """
    if isinstance(x, DualVec3):
        n = float(np.linalg.norm(x.val))
        if n == 0.0:
            raise ValueError("Dual vector norm derivative is undefined at zero length")
        deriv = float(np.dot(x.val, x.deriv)) / n
        return DualScalar(n, deriv)
    # Fall through to standard numpy for non-dual inputs.
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def _dual_implementation(func: object) -> Callable[..., object] | None:
    """Return the dual implementation for a supported numpy function."""
    if func is np.dot:
        return _dual_dot
    if func is np.cross:
        return _dual_cross
    if func is np.linalg.norm:
        return _dual_norm
    return None


# Typed wrappers for dual-compatible numpy operations
# ----------------------------------------------------------------
# numpy stubs do not model the __array_function__ protocol, so np.dot()
# and np.linalg.norm() cannot be called with DualVec3 without a type
# error. These typed wrappers accept both ndarray and DualVec3 and
# dispatch to the correct implementation.


@overload
def dot(a: np.ndarray, b: np.ndarray) -> np.floating: ...


@overload
def dot(a: DualVec3 | np.ndarray, b: DualVec3) -> DualScalar: ...


@overload
def dot(a: DualVec3, b: DualVec3 | np.ndarray) -> DualScalar: ...


def dot(a: DualVec3 | np.ndarray, b: DualVec3 | np.ndarray) -> DualScalar | np.floating:
    """
    Typed dot product that supports DualVec3 operands.
    """
    if isinstance(a, DualVec3) or isinstance(b, DualVec3):
        return _dual_dot(a, b)
    return np.dot(a, b)


@overload
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...


@overload
def cross(a: DualVec3 | np.ndarray, b: DualVec3) -> DualVec3: ...


@overload
def cross(a: DualVec3, b: DualVec3 | np.ndarray) -> DualVec3: ...


def cross(a: DualVec3 | np.ndarray, b: DualVec3 | np.ndarray) -> DualVec3 | np.ndarray:
    """Typed cross product supporting dual-vector operands."""
    if isinstance(a, DualVec3) or isinstance(b, DualVec3):
        return _dual_cross(a, b)
    return np.cross(extract_array(a), extract_array(b))


def norm(v: DualVec3) -> DualScalar:
    """
    Typed vector norm for DualVec3.
    """
    n = float(np.linalg.norm(v.val))
    if n == 0.0:
        raise ValueError("Dual vector norm derivative is undefined at zero length")
    deriv = float(np.dot(v.val, v.deriv)) / n
    return DualScalar(n, deriv)


# Dual-aware scalar math functions.


@overload
def sqrt(x: DualScalar) -> DualScalar: ...


@overload
def sqrt(x: float) -> float: ...


def sqrt(x: DualScalar | float) -> DualScalar | float:
    """Dual-aware square root."""
    if isinstance(x, DualScalar):
        root = math.sqrt(x.val)
        # d(sqrt(x)) = x' / (2 * sqrt(x)).
        return DualScalar(root, x.deriv / (2.0 * root))
    return math.sqrt(x)


@overload
def atan2(y: DualScalar, x: DualScalar | float) -> DualScalar: ...


@overload
def atan2(y: DualScalar | float, x: DualScalar) -> DualScalar: ...


@overload
def atan2(y: float, x: float) -> float: ...


def atan2(y: DualScalar | float, x: DualScalar | float) -> DualScalar | float:
    """Dual-aware two-argument arctangent."""
    if isinstance(y, DualScalar) or isinstance(x, DualScalar):
        y_val = y.val if isinstance(y, DualScalar) else float(y)
        y_deriv = y.deriv if isinstance(y, DualScalar) else 0.0
        x_val = x.val if isinstance(x, DualScalar) else float(x)
        x_deriv = x.deriv if isinstance(x, DualScalar) else 0.0
        value = math.atan2(y_val, x_val)
        # d(atan2(y, x)) = (x * y' - y * x') / (x^2 + y^2).
        denominator = x_val**2 + y_val**2
        derivative = (x_val * y_deriv - y_val * x_deriv) / denominator
        return DualScalar(value, derivative)
    return math.atan2(y, x)


@overload
def degrees(x: DualScalar) -> DualScalar: ...


@overload
def degrees(x: float) -> float: ...


def degrees(x: DualScalar | float) -> DualScalar | float:
    """Dual-aware radians-to-degrees conversion."""
    if isinstance(x, DualScalar):
        radians_to_degrees = 180.0 / math.pi
        return DualScalar(
            x.val * radians_to_degrees,
            x.deriv * radians_to_degrees,
        )
    return math.degrees(x)


# Seeding utility
# ----------------------------------------------------------------

# Seeding preserves the caller's concrete key type: single-corner callers seed a
# PointID-keyed map, axle callers a PointRef-keyed map. Parametrizing over the key
# keeps the returned dict's key type precise instead of the invariant PointKey union.
_SeedKey = TypeVar("_SeedKey", bound=PointKey)


def seed_positions(
    positions: Mapping[_SeedKey, np.ndarray | ArrayBacked],
    seed_point: _SeedKey,
    seed_dim: int,
) -> dict[_SeedKey, DualVec3]:
    """
    Create dual-number positions seeded for differentiation.

    Wraps every position as a DualVec3 with zero derivative, except
    seed_point which gets derivative = 1.0 in coordinate seed_dim.
    This sets up forward-mode AD to compute d(output)/d(seed_point[seed_dim]).

    Handles both raw ndarray and Point3 positions transparently.

    Args:
        positions: Current point positions (PointKey -> ndarray or Point3).
        seed_point: The point whose coordinate we differentiate w.r.t.
        seed_dim: Which coordinate (0=x, 1=y, 2=z) to seed.

    Returns:
        Dictionary of DualVec3 positions ready for derived-point computation.
    """
    dual_positions: dict[_SeedKey, DualVec3] = {}
    for pid, pos in positions.items():
        raw = extract_array(pos)
        if pid == seed_point:
            d = np.zeros(3, dtype=np.float64)
            d[seed_dim] = 1.0
            dual_positions[pid] = DualVec3(raw.copy(), d)
        else:
            dual_positions[pid] = DualVec3(raw.copy())
    return dual_positions


def seed_positions_with_tangent(
    positions: Mapping[_SeedKey, np.ndarray | ArrayBacked],
    tangent: Mapping[_SeedKey, np.ndarray],
) -> dict[_SeedKey, DualVec3]:
    """
    Seed every position along an arbitrary tangent field.

    Missing tangent keys are held constant. This evaluates a
    Jacobian-vector product in one forward dual-number pass.
    """
    dual_positions: dict[_SeedKey, DualVec3] = {}
    for point_id, position in positions.items():
        raw = extract_array(position)
        tangent_vector = tangent.get(point_id)
        if tangent_vector is None:
            derivative = np.zeros(3, dtype=np.float64)
        else:
            derivative = np.asarray(tangent_vector, dtype=np.float64).copy()
        dual_positions[point_id] = DualVec3(raw.copy(), derivative)
    return dual_positions
