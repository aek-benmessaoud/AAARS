"""Runner smoke test — one episode with all methods."""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.runner import run_episode, AAARS_STOP_RULES, DEFAULT_CFG


class TestRunner:
    def test_single_episode(self):
        cfg = {**DEFAULT_CFG, "max_steps": 300}
        result = run_episode("boustro", 0, env_seed=0, cfg=cfg)
        
        assert result is not None
        assert result["alloc"] == "boustro"
        assert result["run"] == 0
        
        for rule in AAARS_STOP_RULES:
            t_key = f"{rule}__t"
            r_key = f"{rule}__recall"
            assert t_key in result
            assert r_key in result
            assert 0 <= result[r_key] <= 100
    
    def test_both_allocations(self):
        cfg = {**DEFAULT_CFG, "max_steps": 300}
        for alloc in ["boustro", "minerich"]:
            result = run_episode(alloc, 0, env_seed=0, cfg=cfg)
            assert result["alloc"] == alloc
    
    def test_aaars_fields(self):
        cfg = {**DEFAULT_CFG, "max_steps": 300}
        result = run_episode("boustro", 0, env_seed=0, cfg=cfg)
        assert "aaars__switches" in result
        assert "aaars__final_risk" in result
        assert isinstance(result["aaars__switches"], int)
        assert isinstance(result["aaars__final_risk"], float)
    
    def test_trace_format(self):
        cfg = {**DEFAULT_CFG, "max_steps": 200, "trace_stride": 50}
        result = run_episode("boustro", 0, env_seed=0, cfg=cfg)
        assert "_trace" in result
        trace = json.loads(result["_trace"])
        assert len(trace) > 0
        entry = trace[0]
        assert "t" in entry
        assert "risk_score" in entry
        assert "f1" in entry
        assert "f2" in entry
