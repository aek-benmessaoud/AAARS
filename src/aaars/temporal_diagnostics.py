"""
temporal_diagnostics.py — Temporal clustering diagnostics for AAARS.

Estimates whether observations are temporally dispersed (systematic allocation)
or clustered (locality-driven allocation).
"""

import numpy as np


class TemporalDiagnostics:
    """Compute temporal clustering metrics from observation monitor data."""
    
    def __init__(self):
        self.cv_interval = 0.0  # coefficient of variation of inter-detection intervals
        self.cv_revisit = 0.0   # coefficient of variation of revisit intervals
        self.short_revisit_frac = 0.0
    
    def update(self, monitor):
        """Recompute diagnostics from the observation monitor.
        
        Args:
            monitor: ObservationMonitor instance
        """
        # Inter-detection interval CV (step-level, not cell-level)
        steps = monitor._detection_steps
        if len(steps) >= 3:
            arr = np.array(list(steps), dtype=float)
            diffs = np.diff(arr)
            if diffs.mean() > 0:
                self.cv_interval = float(diffs.std() / diffs.mean())
            else:
                self.cv_interval = 0.0
        else:
            self.cv_interval = 0.0
        
        # Revisit interval CV
        rvis = monitor.get_revisit_intervals()
        if len(rvis) >= 3:
            arr = np.array(rvis, dtype=float)
            mean_rv = np.mean(arr)
            std_rv = np.std(arr)
            self.cv_revisit = float(std_rv / (mean_rv + 1e-8))
        else:
            self.cv_revisit = 0.0
        
        # Short revisit fraction
        self.short_revisit_frac = monitor.get_short_revisit_fraction()
    
    def temporal_clustering_score(self):
        """Normalized temporal clustering score in [0, 1].
        
        High TCS = observations are temporally clustered (risky for Chao1).
        Low TCS = observations are temporally dispersed (safe for Chao1).
        
        Formula: TCS = clip((cv_interval + cv_revisit + short_revisit_frac) / 3, 0, 1)
        
        This is an online diagnostic feature, NOT a theoretically optimal
        clustering measure. Each component captures a different aspect:
        - cv_interval: irregularity of detection timing
        - cv_revisit: irregularity of revisit patterns  
        - short_revisit_frac: fraction of very rapid revisits (local search signature)
        """
        raw = (self.cv_interval + self.cv_revisit + self.short_revisit_frac) / 3.0
        return float(np.clip(raw, 0.0, 1.0))
    
    def get_state(self):
        """Return current diagnostic values as a dict for logging."""
        return {
            "cv_interval": round(self.cv_interval, 4),
            "cv_revisit": round(self.cv_revisit, 4),
            "short_revisit_frac": round(self.short_revisit_frac, 4),
            "temporal_clustering": round(self.temporal_clustering_score(), 4),
        }
