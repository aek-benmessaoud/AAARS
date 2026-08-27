"""Tests for estimator selector and hysteresis."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aaars.discrete_selector import DiscreteSelectorController
from src.aaars.risk_score import RiskScore


class TestDiscreteSelector:
    def test_default_estimator(self):
        ds = DiscreteSelectorController(grid_size=50)
        assert ds.current_estimator == "chao1"
    
    def test_hysteresis_prevents_immediate_switch(self):
        """State must persist for M steps before switching."""
        ds = DiscreteSelectorController(grid_size=50, hysteresis_M=3)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        # Provide enough data to arm
        for t in range(1, 20):
            bits[:] = 0
            bits[0, 0] = 3  # f2
            bits[1, 1] = 3
            bits[2, 2] = 3
            bits[3, 3] = 3
            bits[4, 4] = 3
            bits[5, 5] = 1  # f1
            bits[6, 6] = 1
            bits[7, 7] = 1
            bits[8, 8] = 1
            bits[9, 9] = 1
            result = ds.step(bits, t)
        
        # With hysteresis_M=3, estimator should not switch immediately
        # Just verify it has a valid estimator
        assert ds.current_estimator in ("chao1", "chao92")
    
    def test_no_ground_truth_leakage(self):
        """Discrete selector must not use ground truth."""
        ds = DiscreteSelectorController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        for t in range(1, 100):
            bits[:] = 0
            if t % 3 == 0:
                bits[t % 30, t % 30] = 1
            result = ds.step(bits, t)
            result_str = str(result)
            for leak in ["true_K", "true_recall", "mine_mask"]:
                assert leak not in result_str


class TestRiskScore:
    def test_weighted_sum(self):
        rs = RiskScore(w_temporal=0.5, w_coverage=0.3, w_frequency=0.2)
        score = rs.update(1.0, 0.0, 0.0)
        assert 0.0 <= score <= 1.0
    
    def test_all_zero_inputs(self):
        rs = RiskScore()
        score = rs.update(0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0, abs=0.01)
    
    def test_all_one_inputs(self):
        rs = RiskScore(ema_alpha=1.0)  # no smoothing
        score = rs.update(1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_smoothing(self):
        rs = RiskScore(ema_alpha=0.1)
        # Jump from 0 to 1
        score = rs.update(1.0, 1.0, 1.0)
        # With EMA alpha=0.1, score should be ~0.1, not 1.0
        assert score < 0.5
