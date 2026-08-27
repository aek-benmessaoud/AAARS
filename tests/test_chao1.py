"""Tests for Chao1 estimator and frequency extraction."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.estimators.chao1 import (
    frequency_from_bits, chao_residual, chao_variance, residual_estimate,
    _POPCOUNT_LUT,
)


class TestPopcount:
    def test_popcount_lut(self):
        assert _POPCOUNT_LUT[0] == 0
        assert _POPCOUNT_LUT[1] == 1
        assert _POPCOUNT_LUT[3] == 2
        assert _POPCOUNT_LUT[7] == 3
        assert _POPCOUNT_LUT[255] == 8


class TestFrequencyFromBits:
    def test_empty(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        n, f1, f2 = frequency_from_bits(bits)
        assert n == 0
        assert f1 == 0.0
        assert f2 == 0.0

    def test_singletons(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 1  # popcount=1 -> f1
        bits[1, 1] = 1  # popcount=1 -> f1
        n, f1, f2 = frequency_from_bits(bits)
        assert n == 2
        assert f1 == 2.0
        assert f2 == 0.0

    def test_doubletons(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 3   # popcount=2 -> f2
        bits[1, 1] = 5   # popcount=2 -> f2
        bits[2, 2] = 6   # popcount=2 -> f2
        n, f1, f2 = frequency_from_bits(bits)
        assert n == 3
        assert f1 == 0.0
        assert f2 == 3.0

    def test_mixed(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 1   # popcount=1 -> f1
        bits[1, 1] = 3   # popcount=2 -> f2
        bits[2, 2] = 7   # popcount=3 -> f3 (not f1 or f2)
        n, f1, f2 = frequency_from_bits(bits)
        assert n == 3
        assert f1 == 1.0
        assert f2 == 1.0

    def test_popcount_not_bitmask_value(self):
        """Critical: f_k counts popcounts, not raw bitmask values."""
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 4  # popcount=1 -> f1, NOT f4
        bits[1, 1] = 8  # popcount=1 -> f1, NOT f8
        n, f1, f2 = frequency_from_bits(bits)
        assert n == 2
        assert f1 == 2.0
        assert f2 == 0.0


class TestChaoResidual:
    def test_basic(self):
        U = chao_residual(10, 5, 3)
        assert U == pytest.approx(5 * 4 / (2 * 4))

    def test_f2_zero(self):
        U = chao_residual(10, 5, 0)
        assert U == pytest.approx(5 * 4 / (2 * 1))

    def test_f1_zero(self):
        U = chao_residual(10, 0, 3)
        assert U == 0.0

    def test_negative_floor(self):
        U = chao_residual(10, 1, 5)
        assert U >= 0.0


class TestChaoVariance:
    def test_f2_zero(self):
        assert chao_variance(5, 0) == 0.0

    def test_positive(self):
        v = chao_variance(5, 3)
        assert v > 0.0

    def test_formula(self):
        f1, f2 = 6.0, 3.0
        r = f1 / f2
        expected = f2 * (r**2 / 4 + r**3 / 4)
        assert chao_variance(f1, f2) == pytest.approx(expected)


class TestResidualEstimate:
    def test_returns_all_keys(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 1
        bits[1, 1] = 3
        est = residual_estimate(bits)
        for key in ["n_det", "f1", "f2", "U_hat", "sd", "ci_upper",
                     "K_hat", "recall_est"]:
            assert key in est

    def test_recall_range(self):
        bits = np.zeros((10, 10), dtype=np.uint8)
        bits[0, 0] = 1
        est = residual_estimate(bits)
        assert 0.0 <= est["recall_est"] <= 1.0
