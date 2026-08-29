"""Tests for AAARS controller — no ground-truth leakage."""

import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aaars.controller import AAARSController
from src.percepts import PerceptStep
from src.experiments.runner import run_episode, DEFAULT_CFG
from src.percepts.io import percept_from_dict


def _percept(bits, t):
    return PerceptStep(t=t, bits=bits, fleet_coverage=0.0)


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
            result = ctrl.step(_percept(bits, t))
            
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
        result = ctrl.step(_percept(bits, 1))
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
            result = ctrl.step(_percept(bits, t))
        
        # Should be armed but likely not stopped yet (few detections)
        assert result["armed"] is True
    
    def test_risk_score_bounds(self):
        """Risk score should always be in [0, 1]."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        
        for t in range(1, 200):
            bits[:] = 0
            bits[t % 40, t % 40] = 1
            result = ctrl.step(_percept(bits, t))
            risk = result.get("risk_score", 0)
            assert 0.0 <= risk <= 1.0, f"Risk score {risk} out of [0,1]"
    
    def test_reset(self):
        """Controller should reset cleanly."""
        ctrl = AAARSController(grid_size=50)
        bits = np.zeros((50, 50), dtype=np.uint8)
        bits[0, 0] = 1
        
        for t in range(1, 20):
            ctrl.step(_percept(bits, t))
        
        ctrl.reset()
        assert ctrl.stopped is False
        assert ctrl.stop_step is None
        assert ctrl.num_switches == 0


LEAK_FREE_PERCEPT_KEYS = {
    "t", "bits_b64", "fleet_coverage",
    "new_cells_scanned", "new_finds", "coverage_frac",
}
# Ground-truth tokens that must never appear in the rule-facing stream.
GT_TOKENS = ["true_found", "n_detections", "true_K", "true_recall",
             "mine_mask", "located_mask", "neutralized_mask"]


class TestRunnerResultDictNoGroundTruth:
    """Optional D3 add-on: the runner result dict / percept stream leaks nothing.

    Extends the controller-level leak test to the runner surface: the recorded
    PerceptStep stream (what rules consume and the D5 replay consumes) must be
    structurally free of ground truth, and ground truth must live only in the
    explicit post-hoc scoring fields, never in the rule-facing stream.
    """

    def _run(self, alloc):
        cfg = {**DEFAULT_CFG, "grid_size": 40, "max_steps": 300,
               "coverage_only_threshold": 0.95}
        return run_episode(alloc, 0, env_seed=0, cfg=cfg,
                           collect_percepts=True, collect_trace=True)

    def _stream(self, res):
        return json.loads(res["_percepts"])

    def test_percept_stream_keys_are_exactly_leak_free(self):
        for alloc in ("boustro", "minerich"):
            stream = self._stream(self._run(alloc))
            assert len(stream) > 0, f"{alloc}: empty percept stream"
            for d in stream:
                assert set(d.keys()) == LEAK_FREE_PERCEPT_KEYS, \
                    f"{alloc}: unexpected key set {set(d.keys())}"

    def test_no_ground_truth_token_in_serialized_stream(self):
        for alloc in ("boustro", "minerich"):
            blob = json.dumps(self._stream(self._run(alloc)))
            for tok in GT_TOKENS:
                assert tok not in blob, \
                    f"{alloc}: ground-truth token '{tok}' leaked into percept stream"

    def test_reconstructed_percept_as_dict_is_leak_free(self):
        for alloc in ("boustro", "minerich"):
            res = self._run(alloc)
            stream = self._stream(res)
            for d in stream:
                p = percept_from_dict(d, 40)
                ad = p.as_dict()
                assert set(ad.keys()) == LEAK_FREE_PERCEPT_KEYS - {"bits_b64"}
                for tok in GT_TOKENS:
                    assert tok not in json.dumps(ad), tok

    def test_ground_truth_confined_to_scoring_fields(self):
        """true_found appears only in the scorer trace/outcome, never the stream."""
        res = self._run("minerich")
        assert "_trace" in res and "_percepts" in res
        # The rule-facing stream must not mention true_found even as a field name,
        # while the explicit evaluation trace legitimately carries it.
        assert "true_found" not in json.dumps(self._stream(res))
        assert "true_found" in res["_trace"]
        # Outcome verdicts are the only other post-hoc ground-truth surface.
        assert "chao1_ci__recall" in res

