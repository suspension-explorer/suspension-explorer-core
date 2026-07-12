"""
Affine geometry primitives: Point3, Vector3, Direction3.

These types enforce the CGAL kernel model where points (locations), vectors
(displacements), and directions (orientations) are distinct mathematical
objects with well-defined operations between them.

Type algebra:
    Point3  - Point3    -> Vector3
    Point3  +/- Vector3 -> Point3
    Vector3 +/- Vector3 -> Vector3
    Vector3 * scalar    -> Vector3
    Direction3 * scalar -> Vector3
    -Direction3         -> Direction3

Disallowed operations (raise TypeError at runtime):
    Point3 + Point3
    Point3 * scalar
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Final, Protocol, overload

import numpy as np
from numpy.typing import NDArray

from kinematics.core.constants import EPS_GEOMETRIC

# ---------------------------------------------------------------------------
# Numpy dispatch support
# ---------------------------------------------------------------------------
# Types that our __array_function__ knows how to handle.  If any argument
# type is NOT in this set we return NotImplemented so that other protocols
# (e.g. DualVec3) can take over.
GEOM_TYPES: tuple[type, ...] = ()  # Populated after class definitions.


class ArrayBacked(Protocol):
    """
    Provide an ndarray through a data attribute.
    """

    data: np.ndarray


def extract_array(x: object) -> np.ndarray:
    """
    Extract the raw ndarray from a supported array-backed value.

    Plain ndarrays pass through unchanged. Geometry wrappers provide their
    backing array through ``data``. Lists and tuples are converted to arrays.

    Raises:
        TypeError: If the value is not array-backed or an array-like sequence.
    """
    if isinstance(x, np.ndarray):
        return x
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    raise TypeError(f"Expected an array-backed value, got {type(x).__name__}")


def try_extract_array(x: object) -> np.ndarray | None:
    """
    Return an operand's backing array when it has one.

    This dynamic probe is used by numeric protocols that must return
    ``NotImplemented`` for unsupported operand types.
    """
    if isinstance(x, np.ndarray):
        return x
    data = getattr(x, "data", None)
    return data if isinstance(data, np.ndarray) else None


# ---------------------------------------------------------------------------
# Point3
# ---------------------------------------------------------------------------


class Point3:
    """
    A location in 3D Euclidean space.

    Points support only affine-legal operations: subtracting two points
    yields a Vector3, and translating a point by a vector yields a new
    point.  Adding two points or scaling a point is disallowed.
    """

    __slots__ = ("data",)

    def __init__(self, data: NDArray[Any] | list | tuple | Point3) -> None:
        if isinstance(data, Point3):
            self.data: np.ndarray = data.data.copy()
        else:
            # np.array (not asarray) to guarantee an owned copy: if the caller
            # passes a float64 ndarray, asarray would alias and later mutations
            # of self.data would leak back into their buffer.
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(f"Point3 requires shape (3,), got {arr.shape}")
            self.data = arr

    @classmethod
    def from_trusted(cls, data: np.ndarray) -> Point3:
        """
        Wrap a (3,) float64 ndarray without copying or validation.

        The caller must ensure the array is shape (3,) and dtype float64.
        Storage is shared with the input -- mutations on either side are
        visible to the other. Use only at hot-path boundaries where the
        aliasing is intentional (e.g. binding to a solver parameter buffer).
        """
        obj = cls.__new__(cls)
        obj.data = data
        return obj

    # -- Affine arithmetic --------------------------------------------------

    @overload
    def __sub__(self, other: Point3) -> Vector3: ...
    @overload
    def __sub__(self, other: Vector3) -> Point3: ...

    def __sub__(self, other: Point3 | Vector3) -> Vector3 | Point3:
        if isinstance(other, Point3):
            return Vector3(self.data - other.data)
        if isinstance(other, Vector3):
            return Point3(self.data - other.data)
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: Vector3) -> Point3:
        if isinstance(other, Vector3):
            return Point3(self.data + other.data)
        return NotImplemented  # type: ignore[return-value]

    def __radd__(self, other: Vector3) -> Point3:
        if isinstance(other, Vector3):
            return Point3(other.data + self.data)
        return NotImplemented  # type: ignore[return-value]

    # -- Component access ---------------------------------------------------

    def __getitem__(self, idx: int) -> float:
        return float(self.data[int(idx)])

    # -- Numpy interop ------------------------------------------------------

    if TYPE_CHECKING:
        # Type-checker-only declaration so Pyright/ty treat Point3 as ArrayLike
        # for numpy function calls (np.asarray, etc.). At runtime the method is
        # NOT defined, so numpy continues to dispatch via __array_function__ /
        # __array_ufunc__ instead of falling through to a raw conversion.
        def __array__(self, dtype: Any = None) -> np.ndarray: ...

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, GEOM_TYPES) for t in types):
            return NotImplemented
        impl = _geometry_implementation(func)
        if impl is not None:
            return impl(*args, **kwargs)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

    # -- Utilities ----------------------------------------------------------

    def copy(self) -> Point3:
        return Point3(self.data.copy())

    def __eq__(self, other: object) -> bool:
        # Exact bitwise comparison via np.array_equal. Use almost_equals for
        # tolerant comparison across two arithmetic paths.
        if isinstance(other, Point3):
            return bool(np.array_equal(self.data, other.data))
        return NotImplemented

    def almost_equals(
        self, other: Point3, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """
        Tolerant elementwise comparison via np.allclose with numpy defaults.
        """
        if not isinstance(other, Point3):
            return NotImplemented
        return bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

    def __repr__(self) -> str:
        return f"Point3({self.data})"


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------


class Vector3:
    """
    A displacement in 3D Euclidean space.

    Vectors support full linear-space arithmetic: addition, subtraction,
    scalar multiplication, dot product, cross product, and normalization.
    """

    __slots__ = ("data",)

    def __init__(self, data: NDArray[Any] | list | tuple | Vector3) -> None:
        if isinstance(data, Vector3):
            self.data: np.ndarray = data.data.copy()
        else:
            # np.array (not asarray) to guarantee an owned copy; see Point3.
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(f"Vector3 requires shape (3,), got {arr.shape}")
            self.data = arr

    @classmethod
    def from_trusted(cls, data: np.ndarray) -> Vector3:
        """
        Wrap a (3,) float64 ndarray without copying or validation.

        See Point3.from_trusted for caveats. Storage is shared with the input.
        """
        obj = cls.__new__(cls)
        obj.data = data
        return obj

    # -- Vector arithmetic --------------------------------------------------

    @overload
    def __add__(self, other: Vector3) -> Vector3: ...
    @overload
    def __add__(self, other: Point3) -> Point3: ...

    def __add__(self, other: Vector3 | Point3) -> Vector3 | Point3:
        if isinstance(other, Vector3):
            return Vector3(self.data + other.data)
        if isinstance(other, Point3):
            return Point3(self.data + other.data)
        return NotImplemented  # type: ignore[return-value]

    def __radd__(self, other: object) -> Vector3 | Point3:
        if isinstance(other, Vector3):
            return Vector3(other.data + self.data)
        if isinstance(other, Point3):
            return Point3(other.data + self.data)
        return NotImplemented  # type: ignore[return-value]

    def __sub__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(self.data - other.data)
        return NotImplemented  # type: ignore[return-value]

    def __rsub__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(other.data - self.data)
        return NotImplemented  # type: ignore[return-value]

    def __mul__(self, scalar: int | float) -> Vector3:
        if isinstance(scalar, (int, float)):
            return Vector3(self.data * scalar)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, scalar: int | float) -> Vector3:
        if isinstance(scalar, (int, float)):
            return Vector3(scalar * self.data)
        return NotImplemented  # type: ignore[return-value]

    def __truediv__(self, scalar: int | float) -> Vector3:
        if isinstance(scalar, (int, float)):
            return Vector3(self.data / scalar)
        return NotImplemented  # type: ignore[return-value]

    def __neg__(self) -> Vector3:
        return Vector3(-self.data)

    # -- Component access ---------------------------------------------------

    def __getitem__(self, idx: int) -> float:
        return float(self.data[int(idx)])

    # -- Linear algebra methods ---------------------------------------------

    def dot(self, other: Vector3 | Direction3) -> float:
        """
        Dot product with another vector or direction.
        """
        return float(np.dot(self.data, other.data))

    def cross(self, other: Vector3 | Direction3) -> Vector3:
        """
        Cross product with another vector or direction.
        """
        return Vector3(np.cross(self.data, other.data))

    def norm(self) -> float:
        """
        Euclidean length of this vector.
        """
        return float(np.linalg.norm(self.data))

    def squared_norm(self) -> float:
        """
        Squared Euclidean length (avoids the sqrt).
        """
        return float(np.dot(self.data, self.data))

    def normalize(self) -> Direction3:
        """
        Return a unit-length Direction3 in the same orientation.

        Raises:
            ValueError: If the vector has zero length.
        """
        return Direction3(self.data)

    # -- Numpy interop ------------------------------------------------------

    if TYPE_CHECKING:
        # Type-checker-only ArrayLike hint; see Point3 for explanation.
        def __array__(self, dtype: Any = None) -> np.ndarray: ...

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, GEOM_TYPES) for t in types):
            return NotImplemented
        impl = _geometry_implementation(func)
        if impl is not None:
            return impl(*args, **kwargs)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or len(inputs) != 2:
            return NotImplemented
        a, b = inputs
        if ufunc is np.multiply:
            if isinstance(a, (int, float, np.floating)) and isinstance(b, Vector3):
                return Vector3(b.data * float(a))
            if isinstance(a, Vector3) and isinstance(b, (int, float, np.floating)):
                return Vector3(a.data * float(b))
        if ufunc is np.subtract:
            if isinstance(a, np.ndarray) and isinstance(b, Vector3):
                return Vector3(a - b.data)
            if isinstance(a, Vector3) and isinstance(b, np.ndarray):
                return Vector3(a.data - b)
        if ufunc is np.add:
            if isinstance(a, np.ndarray) and isinstance(b, Vector3):
                return Vector3(a + b.data)
            if isinstance(a, Vector3) and isinstance(b, np.ndarray):
                return Vector3(a.data + b)
        return NotImplemented

    # -- Utilities ----------------------------------------------------------

    def copy(self) -> Vector3:
        return Vector3(self.data.copy())

    def __eq__(self, other: object) -> bool:
        # Exact comparison; use almost_equals for tolerant comparison.
        if isinstance(other, Vector3):
            return bool(np.array_equal(self.data, other.data))
        return NotImplemented

    def almost_equals(
        self, other: Vector3, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """
        Tolerant elementwise comparison via np.allclose with numpy defaults.
        """
        if not isinstance(other, Vector3):
            return NotImplemented
        return bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

    def __repr__(self) -> str:
        return f"Vector3({self.data})"


# ---------------------------------------------------------------------------
# Direction3
# ---------------------------------------------------------------------------


class Direction3:
    """
    A unit vector representing an orientation in 3D space.

    Constructed from any non-zero vector; the input is normalized
    automatically.  Scaling a direction produces a Vector3 (not a
    direction), and negating a direction yields another direction.
    """

    __slots__ = ("data",)

    def __init__(
        self, data: NDArray[Any] | list | tuple | Vector3 | Direction3
    ) -> None:
        if isinstance(data, Direction3):
            self.data: np.ndarray = data.data.copy()
            return

        if isinstance(data, Vector3):
            raw = data.data
        else:
            raw = np.asarray(data, dtype=np.float64)

        if raw.shape != (3,):
            raise ValueError(f"Direction3 requires shape (3,), got {raw.shape}")
        magnitude = float(np.linalg.norm(raw))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("Cannot create Direction3 from zero-length vector")
        self.data = raw / magnitude

    @classmethod
    def from_trusted(cls, data: np.ndarray) -> Direction3:
        """
        Create from data known to already be unit-length (skips normalization).
        """
        obj = cls.__new__(cls)
        obj.data = data
        return obj

    # -- Scaling produces Vector3 -------------------------------------------

    def __mul__(self, scalar: int | float) -> Vector3:
        if isinstance(scalar, (int, float)):
            return Vector3(self.data * scalar)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, scalar: int | float) -> Vector3:
        if isinstance(scalar, (int, float)):
            return Vector3(scalar * self.data)
        return NotImplemented  # type: ignore[return-value]

    def __neg__(self) -> Direction3:
        return Direction3.from_trusted(-self.data)

    # -- Component access ---------------------------------------------------

    def __getitem__(self, idx: int) -> float:
        return float(self.data[int(idx)])

    # -- Linear algebra methods ---------------------------------------------

    def dot(self, other: Vector3 | Direction3) -> float:
        """
        Dot product with another vector or direction.
        """
        return float(np.dot(self.data, other.data))

    def cross(self, other: Vector3 | Direction3) -> Vector3:
        """
        Cross product with another vector or direction.
        """
        return Vector3(np.cross(self.data, other.data))

    def vector(self) -> Vector3:
        """
        Convert to a unit-length Vector3.
        """
        return Vector3(self.data.copy())

    # -- Numpy interop ------------------------------------------------------

    if TYPE_CHECKING:
        # Type-checker-only ArrayLike hint; see Point3 for explanation.
        def __array__(self, dtype: Any = None) -> np.ndarray: ...

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, GEOM_TYPES) for t in types):
            return NotImplemented
        impl = _geometry_implementation(func)
        if impl is not None:
            return impl(*args, **kwargs)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or len(inputs) != 2:
            return NotImplemented
        a, b = inputs
        if ufunc is np.multiply:
            if isinstance(a, (int, float, np.floating)) and isinstance(b, Direction3):
                return Vector3(b.data * float(a))
            if isinstance(a, Direction3) and isinstance(b, (int, float, np.floating)):
                return Vector3(a.data * float(b))
        return NotImplemented

    # -- Utilities ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        # Exact comparison; use almost_equals for tolerant comparison.
        if isinstance(other, Direction3):
            return bool(np.array_equal(self.data, other.data))
        return NotImplemented

    def almost_equals(
        self, other: Direction3, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """
        Tolerant elementwise comparison via np.allclose with numpy defaults.
        """
        if not isinstance(other, Direction3):
            return NotImplemented
        return bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

    def __repr__(self) -> str:
        return f"Direction3({self.data})"


# ---------------------------------------------------------------------------
# Populate the geometry type tuple (used by __array_function__ guards)
# ---------------------------------------------------------------------------
GEOM_TYPES = (Point3, Vector3, Direction3, np.ndarray)


# ---------------------------------------------------------------------------
# Numpy __array_function__ implementations
# ---------------------------------------------------------------------------


def geom_dot(
    a: np.ndarray | ArrayBacked,
    b: np.ndarray | ArrayBacked,
    out: object = None,
) -> float:
    """
    Geometry-aware dot product.

    Extracts raw arrays from geometry wrappers and delegates to numpy.
    """
    a_data = extract_array(a)
    b_data = extract_array(b)
    return float(np.dot(a_data, b_data))


def geom_norm(
    x: object,
    ord: object = None,
    axis: object = None,
    keepdims: bool = False,
) -> float:
    """
    Geometry-aware vector norm.
    """
    if isinstance(x, (Vector3, Direction3, Point3)):
        return float(np.linalg.norm(x.data))
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)  # ty: ignore[no-matching-overload]


def geom_cross(
    a: np.ndarray | ArrayBacked,
    b: np.ndarray | ArrayBacked,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: object = None,
) -> Vector3:
    """
    Geometry-aware cross product. Always returns Vector3.
    """
    a_data = extract_array(a)
    b_data = extract_array(b)
    return Vector3(np.cross(a_data, b_data))


def _geometry_implementation(func: object) -> Callable[..., object] | None:
    """Return the geometry implementation for a supported numpy function."""
    if func is np.dot:
        return geom_dot
    if func is np.linalg.norm:
        return geom_norm
    if func is np.cross:
        return geom_cross
    return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def midpoint(a: Point3, b: Point3) -> Point3:
    """
    Compute the midpoint of two points using affine-correct arithmetic.

    Equivalent to a + (b - a) / 2 -- avoids the illegal Point3 + Point3.
    """
    return Point3(a.data + (b.data - a.data) * 0.5)


NAN_POINT3: Final[Point3] = Point3([np.nan, np.nan, np.nan])
