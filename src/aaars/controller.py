"""
controller.py — AAARS Controller: the main online adaptive stopping algorithm.

AAARS = online observation-process diagnosis + risk-modulated estimator blending
        + adaptive confidence stopping

This is the core contribution: a genuinely new risk-aware stopping mechanism
that continuously blends Chao1 and Chao92 based on observation-process risk,
rather than switching discretely between them.
"""

import numpy as np
from src.percepts import PerceptStep
from src.aaars.observation_monitor import ObservationMonitor
from src.aaars.temporal_diagnostics import TemporalDiagnostics
from src.aaars.spatial_diagnostics import SpatialDiagnostics
from src.aaars.frequency_diagnostics import FrequencyDiagnostics
from src.aaars.risk_score import RiskScore
from src.estimators.confidence_bounds import (
    blended_estimate, should_stop, adaptive_threshold)


class AAARSController:
    """Allocation-Aware Adaptive Richness Stopping controller.
    
    At each decision interval:
    1. Collect new observations (block bits)
    2. Update observation monitor
    3. Compute temporal, spatial, frequency diagnostics
    4. Compute continuous risk score r(t) in [0,1]
    5. Blend Chao1 and Chao92 estimates using r(t)
    6. Evaluate risk-adjusted stopping condition
    7. Return STOP or CONTINUE
    
    Ground truth is NEVER used by the controller.
    """
    
    def __init__(self, 
                 window_size=8,
                 grid_size=100,
                 num_bins=5,
                 w_temporal=0.0, w_coverage=0.55, w_frequency=0.45,
                 ema_alpha=0.1,
                 base_alpha=0.05,
                 risk_lambda=1.0,
                 blend_threshold=0.30,
                 n_min=5,
                 f2_min=5,
                 f2_ratio=0.08):
        """
        Args:
            window_size: observation window for frequency stats
            grid_size: environment grid size
            num_bins: spatial bin count
            w_temporal, w_spatial, w_frequency: risk score component weights
            ema_alpha: risk score smoothing factor
            base_alpha: base stopping threshold (0.05 = 5% residual)
            risk_lambda: how much to tighten threshold under risk
            blend_threshold: risk below this → pure Chao1 (no blend)
            n_min: minimum detections before arming
            f2_min: minimum f2 before arming
            f2_ratio: minimum f2 as fraction of n_det
        """
        self.window_size = window_size
        self.grid_size = grid_size
        self.num_bins = num_bins
        self.base_alpha = base_alpha
        self.risk_lambda = risk_lambda
        self.blend_threshold = blend_threshold
        self.n_min = n_min
        self.f2_min = f2_min
        self.f2_ratio = f2_ratio
        
        # Sub-modules
        self.monitor = ObservationMonitor(window_size=window_size)
        self.temporal = TemporalDiagnostics()
        self.spatial = SpatialDiagnostics(grid_size=grid_size, num_bins=num_bins)
        self.frequency = FrequencyDiagnostics()
        self.risk = RiskScore(w_temporal=w_temporal, w_coverage=w_coverage,
                              w_frequency=w_frequency, ema_alpha=ema_alpha)
        
        # State
        self.stopped = False
        self.stop_step = None
        self.current_risk = 0.0
        self.current_alpha_adj = base_alpha
        self.num_switches = 0
        self._prev_estimator_type = None
    
    def reset(self):
        """Reset controller state for a new episode."""
        self.monitor = ObservationMonitor(window_size=self.window_size)
        self.temporal = TemporalDiagnostics()
        self.spatial = SpatialDiagnostics(grid_size=self.grid_size,
                                          num_bins=self.num_bins)
        self.frequency = FrequencyDiagnostics()
        self.risk = RiskScore(w_temporal=self.risk.w_temporal,
                              w_coverage=self.risk.w_coverage,
                              w_frequency=self.risk.w_frequency,
                              ema_alpha=self.risk.ema_alpha)
        self.stopped = False
        self.stop_step = None
        self.current_risk = 0.0
        self.current_alpha_adj = self.base_alpha
        self.num_switches = 0
        self._prev_estimator_type = None
    
    def step(self, percept: "PerceptStep"):
        """Process one decision interval from a leak-free PerceptStep.

        Args:
            percept: the fleet's leak-free observation for this step. The
                controller reads ONLY fields carried by the PerceptStep
                (scan/fusion bits and fleet coverage); it has no path to the
                environment's ground truth (true K, located/undetected masks,
                true recall) because those quantities are not represented in
                the type.
        
        Returns:
            dict with keys:
                stop: bool — should the mission stop?
                risk_score: float — current risk score
                alpha_adj: float — current adjusted threshold
                blended: dict — full blended estimate
                diagnostics: dict — all diagnostic values
                armed: bool — whether minimum evidence gate is satisfied
        """
        if self.stopped:
            return {"stop": True, "risk_score": self.current_risk,
                    "alpha_adj": self.current_alpha_adj,
                    "blended": None, "diagnostics": {}, "armed": True}
        
        bits_union = percept.bits
        step = percept.t
        fleet_coverage = percept.fleet_coverage
        
        # 1. Update observation monitor
        self.monitor.update(bits_union, step)
        
        # 2. Check minimum evidence gate
        n_det = self.monitor.n_det
        f2 = self.monitor.f2
        f2_floor = max(float(self.f2_min), 
                       np.ceil(self.f2_ratio * n_det))
        armed = n_det >= self.n_min and f2 >= f2_floor
        
        if not armed:
            return {"stop": False, "risk_score": 0.0,
                    "alpha_adj": self.base_alpha,
                    "blended": None, "diagnostics": {}, "armed": False}
        
        # 3. Update diagnostics
        self.temporal.update(self.monitor)
        self.spatial.update(self.monitor)
        self.frequency.update(self.monitor)
        self.spatial.coverage = fleet_coverage
        
        # 4. Compute risk score
        tcs = self.temporal.temporal_clustering_score()
        cov = self.spatial.coverage
        coverage_deficit = max(0.0, min(1.0, 1.0 - cov))
        fs = self.frequency.frequency_stability_score()
        self.current_risk = self.risk.update(tcs, coverage_deficit, fs)
        
        # 5. Adaptive threshold
        self.current_alpha_adj = adaptive_threshold(
            self.current_risk, self.base_alpha, self.risk_lambda)
        
        # 6. Blended estimate and stopping decision
        blended = blended_estimate(bits_union, self.current_risk,
                                   blend_threshold=self.blend_threshold)
        stop, ci = should_stop(blended, self.current_alpha_adj)
        
        # 7. Track estimator type for switch counting
        if self.current_risk < self.blend_threshold:
            est_type = "chao1"
        elif self.current_risk > 0.7:
            est_type = "chao92"
        else:
            est_type = "blended"
        if self._prev_estimator_type is not None and \
                est_type != self._prev_estimator_type:
            self.num_switches += 1
        self._prev_estimator_type = est_type
        
        if stop:
            self.stopped = True
            self.stop_step = step
        
        # 8. Collect all diagnostics for logging
        diag = {}
        diag.update(self.temporal.get_state())
        diag.update(self.spatial.get_state())
        diag.update(self.frequency.get_state())
        diag.update(self.risk.get_state())
        diag["estimator_type"] = est_type
        diag["alpha_adj"] = round(self.current_alpha_adj, 6)
        
        return {
            "stop": stop,
            "risk_score": self.current_risk,
            "alpha_adj": self.current_alpha_adj,
            "blended": blended,
            "diagnostics": diag,
            "armed": True,
        }
    
    def get_final_stats(self):
        """Return episode-level summary stats."""
        return {
            "num_switches": self.num_switches,
            "final_risk_score": round(self.current_risk, 4),
            "stop_step": self.stop_step,
        }
