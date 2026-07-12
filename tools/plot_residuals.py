"""
Plot distance constraint residual and its analytical gradient.

Generates PNG figures showing the residual function and Jacobian
for the DistanceConstraint, illustrating the softnorm regularisation
behavior near zero separation.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import mcplotlib
import numpy as np
from mcplotlib import colours

# -- Style setup -------------------------------------------------------
mcplotlib.style.use_mc_style(
    font="Geist Mono",
    font_path=Path("/Users/nickmccleery/Library/Fonts/GeistMono-VariableFont_wght.ttf"),
)
plt.rcParams.update({"font.family": "Geist Mono", "font.size": 11})

# -- Constants from soft_math ------------------------------------------
EPS = 1e-6
EPS_SQ = EPS**2

OUTPUT_DIR = Path(__file__).parent.parent

c1, c2 = colours.QUALITATIVE_2A  # pink, blue
c3 = colours.QUALITATIVE_6A[3]  # purple
c_naive = "#000000"


def softnorm(s: float) -> float:
    """sqrt(s + EPS^2) - EPS."""
    return math.sqrt(s + EPS_SQ) - EPS


def make_figure(
    x: np.ndarray,
    filename: str,
    L: float = 0.0,
    title_suffix: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Generate a residual + Jacobian side-by-side figure.

    Args:
        x: 1D array of separation values.
        filename: Output filename (saved to OUTPUT_DIR).
        L: Target distance offset. When 0, plots the raw norm shape.
        title_suffix: Optional suffix appended to plot titles.
        ylim: Optional (min, max) for the residual y-axis.
    """
    residual = np.array([softnorm(xi**2) - L for xi in x])
    grad_p2 = x / np.sqrt(x**2 + EPS_SQ)
    grad_p1 = -grad_p2

    residual_naive = np.abs(x) - L
    grad_naive = np.sign(x)

    # Label adapts to whether we're showing the full residual or just
    # the norm shape.
    if L != 0.0:
        res_label = r"softnorm(x$^2$) - L"
        naive_label = r"|x| - L (Naive)"
    else:
        res_label = r"softnorm(x$^2$)"
        naive_label = r"|x| (Naive)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Residual.
    ax1.plot(
        x,
        residual_naive,
        color=c_naive,
        linewidth=1.2,
        label=naive_label,
    )
    ax1.plot(x, residual, color=c1, linewidth=2.0, label=res_label)
    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Residual")
    ax1.set_title(f"Distance Constraint Residual{title_suffix}", fontsize=13, pad=10)
    ax1.legend(loc="upper left", fontsize=9)
    if ylim:
        ax1.set_ylim(ylim)
        ax1.ticklabel_format(axis="y", useOffset=False)

    # Jacobian -- show both partials.
    ax2.plot(
        x,
        grad_naive,
        color=c_naive,
        linewidth=1.2,
        label=r"sign(x) (Naive)",
    )
    ax2.plot(
        x,
        grad_p2,
        color=c2,
        linewidth=2.0,
        label=r"$\partial R / \partial p_{2x}$",
    )
    ax2.plot(
        x,
        grad_p1,
        color=c3,
        linewidth=2.0,
        label=r"$\partial R / \partial p_{1x}$",
    )
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Gradient")
    ax2.set_title(f"Distance Constraint Jacobian{title_suffix}", fontsize=13, pad=10)
    ax2.legend(loc="right", fontsize=9)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


# -- Naive view: [-5, 5], no softnorm comparison -----------------------
# Shows the naive |x| - L and sign(x) partials at zoom-out scale,
# illustrating the non-differentiable cusp / step.
x_wide = np.linspace(-5, 5, 2000)
L = 3.0

fig0, (ax0l, ax0r) = plt.subplots(1, 2, figsize=(14, 5.5))

ax0l.plot(x_wide, np.abs(x_wide) - L, color=c1, linewidth=2.0, label=r"|x| - L")
ax0l.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax0l.axvline(0, color="white", linewidth=0.5, alpha=0.3)
ax0l.set_xlabel("Distance")
ax0l.set_ylabel("Residual")
ax0l.set_title("Distance Constraint Residual", fontsize=13, pad=10)
ax0l.legend(loc="upper left", fontsize=9)

ax0r.plot(
    x_wide,
    np.sign(x_wide),
    color=c2,
    linewidth=2.0,
    label=r"$\partial R / \partial p_{2x}$",
)
ax0r.plot(
    x_wide,
    -np.sign(x_wide),
    color=c3,
    linewidth=2.0,
    label=r"$\partial R / \partial p_{1x}$",
)
ax0r.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax0r.axvline(0, color="white", linewidth=0.5, alpha=0.3)
ax0r.set_xlabel("Distance")
ax0r.set_ylabel("Gradient")
ax0r.set_title("Distance Constraint Jacobian", fontsize=13, pad=10)
ax0r.legend(loc="right", fontsize=9)

fig0.tight_layout(pad=2.0)
fig0.savefig(OUTPUT_DIR / "distance_constraint_naive.png", dpi=200, bbox_inches="tight")
plt.close(fig0)
print("Saved: distance_constraint_naive.png")

# -- Wide view: [-5, 5] ------------------------------------------------
make_figure(
    x=np.linspace(-5, 5, 2000),
    filename="distance_constraint.png",
    L=3.0,
)

# -- Zoomed view: +/- 5*EPS, showing softnorm smoothing ----------------
make_figure(
    x=np.linspace(-5 * EPS, 5 * EPS, 2000),
    filename="distance_constraint_zoom.png",
    L=3.0,
    title_suffix=" (Zoomed)",
    ylim=(-3.0, -2.9999948),
)
