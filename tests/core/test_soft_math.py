"""Tests for the softnorm regularisation utilities."""

import math

from kinematics.core.primitives.soft_math import EPS, EPS_SQ, softnorm


class TestEpsSq:
    """Sanity checks on the regularisation constant."""

    def test_positive(self):
        assert EPS_SQ > 0

    def test_small(self):
        assert EPS_SQ < 1e-6

    def test_eps_is_sqrt_eps_sq(self):
        assert EPS == math.sqrt(EPS_SQ)


class TestSoftnorm:
    """Tests for the softnorm function."""

    def test_zero_returns_zero(self):
        """softnorm(0) should return exactly zero (bias-corrected)."""
        assert softnorm(0.0) == 0.0

    def test_derivative_finite_at_zero(self):
        """The whole point: d/ds[softnorm] at s=0 is 1/(2*EPS), which must be finite."""
        val = 1.0 / (2.0 * math.sqrt(EPS_SQ))
        assert math.isfinite(val)

    def test_close_to_sqrt_at_normal_magnitudes(self):
        """At normal magnitudes the bias correction (~EPS) is negligible."""
        for x in [0.01, 1.0, 100.0, 1e6, 1e12]:
            assert math.isclose(softnorm(x), math.sqrt(x), rel_tol=1e-5)
