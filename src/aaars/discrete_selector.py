"""
discrete_selector.py — Discrete Chao1/Chao92 switching (ablation variant).

This is the original prompt's design kept for ablation comparison.
It switches discretely between Chao1 and Chao92 based on observation state,
rather than the continuous blend used by the main AAARS algorithm.
"""

import numpy as np
from src.percepts import PerceptStep
from src.aaars.observation_monitor import ObservationMonitor
from src.aaars.temporal_diagnostics import TemporalDiagnostics
from src.aaars.spatial_diagnostics import SpatialDiagnostics
from src.aaars.frequency_diagnostics import FrequencyDiagnostics
from src.estimators.chao1 import residual_estimate
from src.estimators.chao92 import chao92_from_freq, full_frequencies


class DiscreteSelectorController:
    """Discrete estimator switching controller (ablation variant).
    
    Observation state -> estimator mapping:
      DISPERSED  -> Chao1-CI
      MODERATE   -> midpoint (no stop)
      CLUSTERED  -> Chao92-CI
    
    Uses hysteresis to prevent oscillation.
    """
    
    # State thresholds — calibrated so they are reachable under minerich
    T_LOW = 0.15  # temporal clustering below this = dispersed
    T_HIGH = 0.35 # temporal clustering above this = clustered
    S_LOW = 0.10  # spatial concentration below this
    S_HIGH = 0.25 # spatial concentration above this
    
    def __init__(self, window_size=8, grid_size=100, hysteresis_M=3,
                 n_min=5, f2_min=5, f2_ratio=0.08, ci_rel=0.05):
        self.window_size = window_size
        self.grid_size = grid_size
        self.hysteresis_M = hysteresis_M
        self.n_min = n_min
        self.f2_min = f2_min
        self.f2_ratio = f2_ratio
        self.ci_rel = ci_rel
        
        self.monitor = ObservationMonitor(window_size=window_size)
        self.temporal = TemporalDiagnostics()
        self.spatial = SpatialDiagnostics(grid_size=grid_size)
        
        self.stopped = False
        self.stop_step = None
        self.current_estimator = "chao1"  # default
        self._state_counter = 0
        self._proposed_state = None
        self.num_switches = 0
    
    def reset(self):
        self.monitor = ObservationMonitor(window_size=self.window_size)
        self.temporal = TemporalDiagnostics()
        self.spatial = SpatialDiagnostics(grid_size=self.grid_size)
        self.stopped = False
        self.stop_step = None
        self.current_estimator = "chao1"
        self._state_counter = 0
        self._proposed_state = None
        self.num_switches = 0
    
    def _classify_state(self, tcs, sc):
        """Classify observation process into DISPERSED/MODERATE/CLUSTERED."""
        if tcs < self.T_LOW and sc < self.S_LOW:
            return "DISPERSED"
        elif tcs > self.T_HIGH or sc > self.S_HIGH:
            return "CLUSTERED"
        else:
            return "MODERATE"
    
    def step(self, percept: "PerceptStep"):
        if self.stopped:
            return {"stop": True, "estimator": self.current_estimator,
                    "state": "STOPPED", "blended": None, "armed": True,
                    "diagnostics": {}}
        
        bits_union = percept.bits
        step = percept.t
        self.monitor.update(bits_union, step)
        n_det = self.monitor.n_det
        f2 = self.monitor.f2
        f2_floor = max(float(self.f2_min), np.ceil(self.f2_ratio * n_det))
        armed = n_det >= self.n_min and f2 >= f2_floor
        
        if not armed:
            return {"stop": False, "estimator": "chao1",
                    "state": "GATE", "blended": None, "armed": False,
                    "diagnostics": {}}
        
        self.temporal.update(self.monitor)
        self.spatial.update(self.monitor)
        
        tcs = self.temporal.temporal_clustering_score()
        sc = self.spatial.spatial_concentration
        proposed = self._classify_state(tcs, sc)
        
        # Hysteresis
        if proposed == self._proposed_state:
            self._state_counter += 1
        else:
            self._proposed_state = proposed
            self._state_counter = 1
        
        if self._state_counter >= self.hysteresis_M:
            old_est = self.current_estimator
            if proposed == "DISPERSED":
                self.current_estimator = "chao1"
            elif proposed == "CLUSTERED":
                self.current_estimator = "chao92"
            # MODERATE: keep current estimator
            if self.current_estimator != old_est:
                self.num_switches += 1
        
        # Evaluate stopping with selected estimator
        stop = False
        blended = None
        if self.current_estimator == "chao1":
            est = residual_estimate(bits_union)
            f2_floor_check = max(5.0, np.ceil(0.08 * est["n_det"]))
            if est["n_det"] >= self.n_min and est["f2"] >= f2_floor_check:
                stop = est["ci_upper"] <= self.ci_rel * est["K_hat"]
            blended = est
        else:  # chao92
            fk = full_frequencies(bits_union)
            est92 = chao92_from_freq(fk)
            f2_floor_check = max(5.0, np.ceil(0.08 * est92["n_det"]))
            if est92["n_det"] >= self.n_min and float(fk[2]) >= f2_floor_check:
                stop = (est92["U92"] >= 0 and 
                        est92["ci92_upper"] <= self.ci_rel * est92["K_hat92"])
            blended = est92
        
        if stop:
            self.stopped = True
            self.stop_step = step
        
        diag = {}
        diag.update(self.temporal.get_state())
        diag.update(self.spatial.get_state())
        diag["observation_state"] = proposed
        diag["estimator"] = self.current_estimator
        
        return {
            "stop": stop,
            "estimator": self.current_estimator,
            "state": proposed,
            "blended": blended,
            "armed": True,
            "diagnostics": diag,
        }
    
    def get_final_stats(self):
        return {
            "num_switches": self.num_switches,
            "final_estimator": self.current_estimator,
            "stop_step": self.stop_step,
        }
