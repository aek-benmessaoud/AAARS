"""Quick debug: check what the diagnostics actually see."""
import sys, os
sys.path.insert(0, r"F:\Project11-AAARS")
from src.aaars.observation_monitor import ObservationMonitor
from src.aaars.temporal_diagnostics import TemporalDiagnostics
from src.aaars.spatial_diagnostics import SpatialDiagnostics
from src.aaars.frequency_diagnostics import FrequencyDiagnostics
import numpy as np

mon = ObservationMonitor()
td = TemporalDiagnostics()
sd = SpatialDiagnostics(grid_size=100)
fd = FrequencyDiagnostics()

# Simulate boustro-like detections: regular, spread out
rng = np.random.default_rng(42)
for step in range(1, 200):
    bits = np.zeros((100, 100), dtype=np.uint8)
    # Place ~5 detections spread across the grid
    for _ in range(5):
        r, c = rng.integers(0, 100), rng.integers(0, 100)
        bits[r, c] = 1  # popcount=1
    mon.update(bits, step)
    
    if step % 50 == 0:
        td.update(mon)
        sd.update(mon)
        fd.update(mon)
        print("t=%d n_det=%d f1=%.0f f2=%.0f det_times=%d revisit_cells=%d" % (
            step, mon.n_det, mon.f1, mon.f2,
            len(mon.detection_times), len(mon.revisit_times)))
        print("  TC=%.4f SC=%.4f FI=%.4f" % (
            td.temporal_clustering_score(),
            sd.spatial_concentration,
            fd.frequency_stability_score()))
