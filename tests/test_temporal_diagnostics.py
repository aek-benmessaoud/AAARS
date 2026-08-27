"""Tests for temporal diagnostics."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aaars.observation_monitor import ObservationMonitor
from src.aaars.temporal_diagnostics import TemporalDiagnostics


class TestTemporalDiagnostics:
    def test_empty_monitor(self):
        td = TemporalDiagnostics()
        mon = ObservationMonitor()
        td.update(mon)
        assert td.temporal_clustering_score() == 0.0

    def test_regular_spacing(self):
        """Regular detection intervals should have low clustering."""
        td = TemporalDiagnostics()
        mon = ObservationMonitor()
        # Simulate regular detections every 10 steps
        bits = np.zeros((20, 20), dtype=np.uint8)
        for step in range(10, 110, 10):
            bits[0, 0] = 1
            mon.update(bits, step)
        td.update(mon)
        score = td.temporal_clustering_score()
        # Regular spacing -> low CV -> low TCS
        assert score < 0.5

    def test_clustered_spacing(self):
        """Clustered detection intervals should have high clustering."""
        td = TemporalDiagnostics()
        mon = ObservationMonitor()
        # Simulate clustered detections: bursts of quick detections then long gaps
        bits = np.zeros((20, 20), dtype=np.uint8)
        steps = [1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 200, 201, 202]
        for step in steps:
            bits[0, 0] = 1
            mon.update(bits, step)
        td.update(mon)
        score = td.temporal_clustering_score()
        # Clustered spacing -> high CV -> high TCS
        assert score > 0.2

    def test_score_range(self):
        """Score should always be in [0, 1]."""
        td = TemporalDiagnostics()
        mon = ObservationMonitor()
        bits = np.zeros((20, 20), dtype=np.uint8)
        bits[0, 0] = 1
        for step in range(1, 50):
            mon.update(bits, step)
        td.update(mon)
        score = td.temporal_clustering_score()
        assert 0.0 <= score <= 1.0

    def test_get_state(self):
        td = TemporalDiagnostics()
        mon = ObservationMonitor()
        bits = np.zeros((20, 20), dtype=np.uint8)
        bits[0, 0] = 1
        mon.update(bits, 1)
        td.update(mon)
        state = td.get_state()
        assert "cv_interval" in state
        assert "cv_revisit" in state
        assert "temporal_clustering" in state
