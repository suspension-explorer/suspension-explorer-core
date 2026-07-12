"""
Softnorm regularisation utilities for the Jacobian functions.

Norm-based residuals use `sqrt(s + EPS_SQ) - EPS` instead of `sqrt(s)`
so the derivative stays finite when a constraint is exactly satisfied (s = 0),
while returning exactly zero at the solution.
"""

import math

from kinematics.core.primitives.constants import EPS_GEOMETRIC

# EPS_SQ = EPS_GEOMETRIC^2 = 1e-12. Large enough to give the solver a useful
# gradient near degeneracies, small enough that the residual bias (~1e-6) is
# invisible for suspension geometry.
EPS: float = EPS_GEOMETRIC
EPS_SQ: float = EPS**2


def softnorm(sum_of_squares: float) -> float:
    """
    Bias-corrected regularised norm: `sqrt(s + EPS_SQ) - EPS`.

    Returns exactly zero when sum_of_squares is zero, with finite derivatives
    everywhere.
    """
    return math.sqrt(sum_of_squares + EPS_SQ) - EPS
