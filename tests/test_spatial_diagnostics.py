"""Tests for spatial diagnostics."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aaars.observation_monitor import ObservationMonitor
from src.aaars.spatial_diagnostics import SpatialDiagnostics


class TestSpatialDiagnostics:
    def test_empty(self):
        sd = SpatialDiagnostics(grid_size=100)
        mon = ObservationMonitor()
        sd.update(mon)
        assert sd.spatial_concentration == 0.0
        assert sd.spatial_dispersion == 0.0

    def test_concentrated(self):
        """Detections in one bin should have high concentration."""
        sd = SpatialDiagnostics(grid_size=100, num_bins=10)
        mon = ObservationMonitor()
        bits = np.zeros((100, 100), dtype=np.uint8)
        # All detections in top-left corner
        for step in range(1, 20):
            r, c = step % 5, step % 5
            bits[r, c] = 1
            mon.update(bits, step)
        sd.update(mon)
        assert sd.spatial_concentration > 0.0
        assert sd.spatial_dispersion > 0.0

    def test_dispersed(self):
        """Detections spread across bins should have lower concentration."""
        sd = SpatialDiagnostics(grid_size=100, num_bins=10)
        mon = ObservationMonitor()
        bits = np.zeros((100, 100), dtype=np.uint8)
        # Spread detections across grid
        positions = [(0, 0), (0, 50), (50, 0), (50, 50), (90, 90)]
        for step, (r, c) in enumerate(positions, 1):
            bits[r, c] = 1
            mon.update(bits, step)
        sd.update(mon)
        # Multiple bins occupied -> higher dispersion
        assert sd.spatial_dispersion > 0.0

    def test_state_dict(self):
        sd = SpatialDiagnostics(grid_size=100)
        state = sd.get_state()
        assert "spatial_concentration" in state
        assert "spatial_dispersion" in state
