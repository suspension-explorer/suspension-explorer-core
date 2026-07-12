#!/usr/bin/env python3
"""
Print common sub-expression optimized Jacobian snippets for kinematic constraints.

Uses SymPy to symbolically differentiate the residual expression for each
constraint type, applies common-subexpression elimination, and prints the
optimized CSE body (temporaries + return expression) to stdout.

The output is meant to be pasted into `src/kinematics/jacobians.py`
when a residual formula changes.

Run
---
    python tools/generate_jacobians.py
"""

from __future__ import annotations

from typing import Any, cast

import sympy as sp
from sympy.printing.pycode import pycode

# CSE temporaries use _t0, _t1, … to avoid clashing with coordinate names
# like x1, x2, …
CSE_SYMBOLS = sp.symbols(" ".join(f"t{i}" for i in range(200)))

# Softnorm regularisation: the residual uses sqrt(s + ε²) - ε, but the
# bias correction is constant and vanishes under differentiation.  The
# symbolic expressions here use sqrt(s + ε²) only — the derivatives are
# the same either way.
EPS_SQ = sp.Symbol("EPS_SQ", positive=True)


def real_symbols(names: str) -> tuple[Any, ...]:
    """
    Create a tuple of real-valued SymPy symbols.
    """
    return cast(tuple[Any, ...], sp.symbols(names, real=True))


def softnorm(sum_of_squares: Any) -> Any:
    """`sqrt(sum_of_squares + EPS_SQ)` — smooth everywhere."""
    return sp.sqrt(sum_of_squares + EPS_SQ)


def print_snippet(
    name: str,
    variables: list[Any],
    residual: Any,
) -> None:
    """Differentiate *residual* w.r.t. *variables* and print the CSE body.

    Prints only the temporaries and the return expression — not the full
    function definition.  The output is ready to paste into the hand-crafted
    module.
    """
    derivs = [sp.diff(residual, v) for v in variables]
    replacements, reduced = sp.cse(derivs, symbols=list(CSE_SYMBOLS))
    reduced_exprs = cast(list[sp.Expr], reduced)

    print(f"# === {name} ===")
    for sym, expr in replacements:
        print(f"    {sym} = {pycode(expr)}")
    rendered_values: list[str] = []
    for r in reduced_exprs:
        rendered = pycode(r)
        if isinstance(rendered, tuple):
            rendered_values.append(rendered[2])
        else:
            rendered_values.append(rendered)
    values = ", ".join(rendered_values)
    print(f"    return np.array([{values}])")
    print()


def gen_distance() -> None:
    """DistanceConstraint / SphericalJointConstraint."""
    x1, y1, z1, x2, y2, z2 = real_symbols("x1 y1 z1 x2 y2 z2")
    dist = softnorm((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    print_snippet("jac_distance", [x1, y1, z1, x2, y2, z2], dist)


def gen_angle() -> None:
    """AngleConstraint (4-point)."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = syms

    v1x, v1y, v1z = x2 - x1, y2 - y1, z2 - z1
    v2x, v2y, v2z = x4 - x3, y4 - y3, z4 - z3

    cx = v1y * v2z - v1z * v2y
    cy = v1z * v2x - v1x * v2z
    cz = v1x * v2y - v1y * v2x
    cross_mag = softnorm(cx**2 + cy**2 + cz**2)
    dot = v1x * v2x + v1y * v2y + v1z * v2z

    print_snippet("jac_angle", list(syms), sp.atan2(cross_mag, dot))


def gen_three_point_angle() -> None:
    """ThreePointAngleConstraint (vertex at p2)."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3")
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = syms

    v1x, v1y, v1z = x1 - x2, y1 - y2, z1 - z2
    v2x, v2y, v2z = x3 - x2, y3 - y2, z3 - z2

    cx = v1y * v2z - v1z * v2y
    cy = v1z * v2x - v1x * v2z
    cz = v1x * v2y - v1y * v2x
    cross_mag = softnorm(cx**2 + cy**2 + cz**2)
    dot = v1x * v2x + v1y * v2y + v1z * v2z

    print_snippet("jac_three_point_angle", list(syms), sp.atan2(cross_mag, dot))


def gen_vectors_parallel() -> None:
    """VectorsParallelConstraint."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = syms

    v1x, v1y, v1z = x2 - x1, y2 - y1, z2 - z1
    v2x, v2y, v2z = x4 - x3, y4 - y3, z4 - z3

    cx = v1y * v2z - v1z * v2y
    cy = v1z * v2x - v1x * v2z
    cz = v1x * v2y - v1y * v2x

    cross_mag = softnorm(cx**2 + cy**2 + cz**2)
    v1_mag = softnorm(v1x**2 + v1y**2 + v1z**2)
    v2_mag = softnorm(v2x**2 + v2y**2 + v2z**2)

    print_snippet("jac_vectors_parallel", list(syms), cross_mag / (v1_mag * v2_mag))


def gen_vectors_perpendicular() -> None:
    """VectorsPerpendicularConstraint."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = syms

    v1x, v1y, v1z = x2 - x1, y2 - y1, z2 - z1
    v2x, v2y, v2z = x4 - x3, y4 - y3, z4 - z3

    dot = v1x * v2x + v1y * v2y + v1z * v2z
    v1_mag = softnorm(v1x**2 + v1y**2 + v1z**2)
    v2_mag = softnorm(v2x**2 + v2y**2 + v2z**2)

    print_snippet("jac_vectors_perpendicular", list(syms), dot / (v1_mag * v2_mag))


def gen_equal_distance() -> None:
    """EqualDistanceConstraint."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = syms

    d1 = softnorm((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    d2 = softnorm((x4 - x3) ** 2 + (y4 - y3) ** 2 + (z4 - z3) ** 2)

    print_snippet("jac_equal_distance", list(syms), d1 - d2)


def gen_point_on_line() -> None:
    """PointOnLineConstraint."""
    px, py, pz = real_symbols("px py pz")
    lpx, lpy, lpz = real_symbols("lpx lpy lpz")
    ldx, ldy, ldz = real_symbols("ldx ldy ldz")

    wx, wy, wz = px - lpx, py - lpy, pz - lpz
    cx = wy * ldz - wz * ldy
    cy = wz * ldx - wx * ldz
    cz = wx * ldy - wy * ldx

    print_snippet("jac_point_on_line", [px, py, pz], softnorm(cx**2 + cy**2 + cz**2))


def gen_point_on_plane() -> None:
    """PointOnPlaneConstraint."""
    px, py, pz = real_symbols("px py pz")
    ppx, ppy, ppz = real_symbols("ppx ppy ppz")
    nx, ny, nz = real_symbols("nx ny nz")

    residual = (px - ppx) * nx + (py - ppy) * ny + (pz - ppz) * nz

    print_snippet("jac_point_on_plane", [px, py, pz], residual)


def gen_coplanar() -> None:
    """CoplanarPointsConstraint."""
    syms = real_symbols("x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = syms

    v1x, v1y, v1z = x2 - x1, y2 - y1, z2 - z1
    v2x, v2y, v2z = x3 - x1, y3 - y1, z3 - z1
    v3x, v3y, v3z = x4 - x1, y4 - y1, z4 - z1

    cx = v2y * v3z - v2z * v3y
    cy = v2z * v3x - v2x * v3z
    cz = v2x * v3y - v2y * v3x

    print_snippet("jac_coplanar", list(syms), v1x * cx + v1y * cy + v1z * cz)


def main() -> None:
    """Generate and print Jacobian snippets for all supported constraints."""
    generators = [
        ("DistanceConstraint / SphericalJointConstraint", gen_distance),
        ("AngleConstraint", gen_angle),
        ("ThreePointAngleConstraint", gen_three_point_angle),
        ("VectorsParallelConstraint", gen_vectors_parallel),
        ("VectorsPerpendicularConstraint", gen_vectors_perpendicular),
        ("EqualDistanceConstraint", gen_equal_distance),
        ("PointOnLineConstraint", gen_point_on_line),
        ("PointOnPlaneConstraint", gen_point_on_plane),
        ("CoplanarPointsConstraint", gen_coplanar),
    ]

    for label, gen_fn in generators:
        print(f"# --- {label} ---")
        gen_fn()


if __name__ == "__main__":
    main()
