"""
spatial_diagnostics.py — Spatial concentration diagnostics for AAARS.

Measures whether detection observations are spatially dispersed or concentrated,
which affects the reliability of Chao estimators.

Two metrics:
  1. spatial_concentration: normalised Herfindahl of detection bin counts
  2. revisit_concentration: Herfindahl of revisited-cell bin counts
     (key discriminator: under minerich revisits cluster around mines,
      under boustro revisits spread across lane boundaries)
"""

import numpy as np


class SpatialDiagnostics:
    """Compute spatial concentration metrics from detection locations.

    Uses 5x5 bins (20x20 cells each) with Herfindahl normalized to [0,1]
    relative to the expected uniform value.  Under uniform distribution across
    25 bins the Herfindahl is 1/25 = 0.04; we rescale so 0 = uniform, 1 = all
    detections in one bin.
    """

    def __init__(self, grid_size=100, num_bins=5):
        """
        Args:
            grid_size: environment grid size
            num_bins: number of spatial bins per axis
        """
        self.grid_size = grid_size
        self.num_bins = num_bins
        self.hhi_uniform = 1.0 / (num_bins * num_bins)
        self.spatial_concentration = 0.0  # normalised Herfindahl [0, 1]
        self.spatial_dispersion = 0.0     # unique_bins / total_bins
        self.revisit_concentration = 0.0  # HHI of revisited-cell bins [0, 1]
        self.coverage = 0.0               # fleet coverage, set externally (leak-free)
    
    def update(self, monitor):
        """Recompute spatial diagnostics from detection history."""
        if monitor.n_det == 0:
            self.spatial_concentration = 0.0
            self.spatial_dispersion = 0.0
            self.revisit_concentration = 0.0
            return
        
        bin_size = max(1, self.grid_size // self.num_bins)
        bin_counts = np.zeros((self.num_bins, self.num_bins), dtype=float)
        revisit_bin_counts = np.zeros((self.num_bins, self.num_bins), dtype=float)
        
        for _step, (r, c) in monitor.detection_times:
            br = min(r // bin_size, self.num_bins - 1)
            bc = min(c // bin_size, self.num_bins - 1)
            bin_counts[br, bc] += 1.0
        
        # Revisit concentration: bins containing cells visited >= 2 times
        for cell, times_deque in monitor.revisit_times.items():
            if len(times_deque) >= 2:
                r, c = cell
                br = min(r // bin_size, self.num_bins - 1)
                bc = min(c // bin_size, self.num_bins - 1)
                revisit_bin_counts[br, bc] += len(times_deque)
        
        total = bin_counts.sum()
        if total <= 0:
            self.spatial_concentration = 0.0
            self.spatial_dispersion = 0.0
            self.revisit_concentration = 0.0
            return
        
        # Detection HHI
        p = bin_counts / total
        hhi = float(np.sum(p ** 2))
        
        denom = 1.0 - self.hhi_uniform
        if denom > 0:
            self.spatial_concentration = max(0.0, min(1.0,
                (hhi - self.hhi_uniform) / denom))
        else:
            self.spatial_concentration = 0.0
        
        non_empty = np.count_nonzero(bin_counts)
        self.spatial_dispersion = non_empty / (self.num_bins ** 2)
        
        # Revisit HHI
        rev_total = revisit_bin_counts.sum()
        if rev_total > 0:
            p_rev = revisit_bin_counts / rev_total
            hhi_rev = float(np.sum(p_rev ** 2))
            if denom > 0:
                self.revisit_concentration = max(0.0, min(1.0,
                    (hhi_rev - self.hhi_uniform) / denom))
            else:
                self.revisit_concentration = 0.0
        else:
            self.revisit_concentration = 0.0
    
    def get_state(self):
        """Return current diagnostic values as a dict for logging."""
        return {
            "spatial_concentration": round(self.spatial_concentration, 4),
            "spatial_dispersion": round(self.spatial_dispersion, 4),
            "revisit_concentration": round(self.revisit_concentration, 4),
            "coverage": round(self.coverage, 4),
            "coverage_deficit": round(max(0.0, 1.0 - self.coverage), 4),
        }
