"""
frequency_diagnostics.py — Frequency-spectrum diagnostics for AAARS.

Tracks f1/f2 ratio and richness-bias signal. The key insight: when f2 > f1,
allocation is causing significant revisitation (richness bias), which
biases Chao1. This is the primary signal for risk modulation.
"""

import numpy as np
from collections import deque


class FrequencyDiagnostics:
    """Track frequency spectrum stability and richness bias over time."""
    
    def __init__(self, history_window=100):
        """
        Args:
            history_window: number of recent f1/f2 snapshots to keep for trend
        """
        self.history_window = history_window
        self.f1_f2_history = deque(maxlen=history_window)
        self.singleton_fraction = 0.0
        self.doubleton_fraction = 0.0
        self.f1_f2_ratio = 0.0
        self.f1_f2_trend = 0.0
        self.richness_bias = 0.0
    
    def update(self, monitor):
        """Recompute frequency diagnostics from observation monitor.
        
        Args:
            monitor: ObservationMonitor instance
        """
        n_det = monitor.n_det
        f1 = monitor.f1
        f2 = monitor.f2
        
        if n_det > 0:
            self.singleton_fraction = f1 / n_det
            self.doubleton_fraction = f2 / n_det
        else:
            self.singleton_fraction = 0.0
            self.doubleton_fraction = 0.0
        
        if f2 > 0:
            self.f1_f2_ratio = f1 / f2
        else:
            self.f1_f2_ratio = 0.0
        
        self.f1_f2_history.append(self.f1_f2_ratio)
        
        if len(self.f1_f2_history) >= 10:
            recent = np.mean(list(self.f1_f2_history)[-10:])
            older = np.mean(list(self.f1_f2_history)[:10])
            self.f1_f2_trend = float(recent - older)
        else:
            self.f1_f2_trend = 0.0
        
        # Richness bias: positive when f2 > f1 (revisitation dominates)
        self.richness_bias = float(np.clip(
            max(0.0, f2 - f1) / (f2 + f1 + 1.0), 0.0, 1.0))
    
    def frequency_stability_score(self):
        """Combined frequency risk score in [0, 1].
        
        Two components:
        - richness_bias: how much doubletons exceed singletons (primary signal)
        - trend: how fast the f1/f2 ratio is changing (secondary signal)
        """
        trend_component = min(abs(self.f1_f2_trend) / (self.f1_f2_ratio + 1.0), 1.0)
        raw = 0.7 * self.richness_bias + 0.3 * trend_component
        return float(np.clip(raw, 0.0, 1.0))
    
    def get_state(self):
        """Return current diagnostic values as a dict for logging."""
        return {
            "f1_f2_ratio": round(self.f1_f2_ratio, 4),
            "f1_f2_trend": round(self.f1_f2_trend, 4),
            "singleton_fraction": round(self.singleton_fraction, 4),
            "doubleton_fraction": round(self.doubleton_fraction, 4),
            "richness_bias": round(self.richness_bias, 4),
            "frequency_instability": round(self.frequency_stability_score(), 4),
        }
