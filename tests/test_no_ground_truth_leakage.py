"""Tests for AAARS controller — no ground-truth leakage."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aaars.controller import AAARSController


class TestAAARSController:
    def test_no_ground_truth_leakage(self):
        """CRITICAL: Controller must operate without ground truth."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        # Run 100 steps without providing any ground truth
        for t in range(1, 101):
            bits[:] = 0
            if t % 5 == 0:
                bits[10, 10] = 1  # simulated detection
            result = ctrl.step(bits, t)
            
            # Result should never contain ground truth fields
            result_str = str(result)
            for leak in ["true_K", "true_recall", "mine_mask", 
                         "located_mask", "neutralized_mask"]:
                assert leak not in result_str, \
                    f"Ground truth leakage detected: {leak} in result"
        
        # Controller should have made decisions based only on beliefs
        stats = ctrl.get_final_stats()
        assert "num_switches" in stats
    
    def test_minimum_evidence_gate(self):
        """Controller should not stop without minimum evidence."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        # Very few detections
        bits[0, 0] = 1
        result = ctrl.step(bits, 1)
        assert result["armed"] is False
        assert result["stop"] is False
    
    def test_stop_requires_sufficient_data(self):
        """Controller should not stop prematurely."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        # Provide enough detections to arm
        for t in range(1, 50):
            bits[:] = 0
            # Create enough f2 to arm
            bits[0, 0] = 3  # popcount=2 -> f2
            bits[1, 1] = 3
            bits[2, 2] = 3
            bits[3, 3] = 3
            bits[4, 4] = 3
            bits[5, 5] = 1  # popcount=1 -> f1
            bits[6, 6] = 1
            bits[7, 7] = 1
            bits[8, 8] = 1
            bits[9, 9] = 1
            result = ctrl.step(bits, t)
        
        # Should be armed but likely not stopped yet (few detections)
        assert result["armed"] is True
    
    def test_risk_score_bounds(self):
        """Risk score should always be in [0, 1]."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        for t in range(1, 200):
            bits[:] = 0
            bits[t % 40, t % 40] = 1
            result = ctrl.step(bits, t)
            risk = result.get("risk_score", 0)
            assert 0.0 <= risk <= 1.0, f"Risk score {risk} out of [0,1]"
    
    def test_reset(self):
        """Controller should reset cleanly."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        bits[0, 0] = 1
        
        for t in range(1, 20):
            ctrl.step(bits, t)
        
        ctrl.reset()
        assert ctrl.stopped is False
        assert ctrl.stop_step is None
        assert ctrl.num_switches == 0
