"""
risk_score.py — Compute the continuous risk score r(t) in [0,1] for AAARS.

r(t) = clip(w1 * TCS + w2 * CD + w3 * FS, 0, 1)

where:
  TCS = temporal clustering score (from temporal_diagnostics)
  CD  = coverage deficit = (1 - fleet_coverage)   [see spatial_diagnostics]
  FS  = frequency instability (from frequency_diagnostics)

Coverage is the key discriminate: under systematic (boustro) sweeping the
fleet covers the whole domain (low deficit -> low risk); under bias-driven
(minerich) allocation the fleet re-scans rich regions and leaves much of the
domain unscanned (high deficit -> high risk), which is exactly when Chao1's
f1^2/(2f2) richness estimate is unreliable.
"""

import numpy as np


class RiskScore:
    """Compute and maintain the continuous risk score."""

    def __init__(self, w_temporal=0.0, w_coverage=0.55, w_frequency=0.45,
                 ema_alpha=0.1):
        """
        Args:
            w_temporal: weight for temporal clustering component
            w_coverage: weight for coverage deficit component
            w_frequency: weight for frequency instability component
            ema_alpha: exponential moving average smoothing (0=no smoothing, 1=instant)
        """
        self.w_temporal = w_temporal
        self.w_coverage = w_coverage
        self.w_frequency = w_frequency
        self.ema_alpha = ema_alpha
        self.risk_score = 0.0
        self._raw_score = 0.0

    def update(self, temporal_clustering, coverage_deficit,
               frequency_instability):
        """Compute risk score from the three diagnostic streams.

        Args:
            temporal_clustering: float in [0, 1] from TemporalDiagnostics
            coverage_deficit: float in [0, 1], 1 - fleet coverage
            frequency_instability: float in [0, 1] from FrequencyDiagnostics

        Returns:
            float: smoothed risk score in [0, 1]
        """
        raw = (self.w_temporal * temporal_clustering +
               self.w_coverage * coverage_deficit +
               self.w_frequency * frequency_instability)

        raw = float(np.clip(raw, 0.0, 1.0))
        self._raw_score = raw

        # EMA smoothing to prevent oscillation
        self.risk_score = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self.risk_score
        self.risk_score = float(np.clip(self.risk_score, 0.0, 1.0))

        return self.risk_score

    def get_state(self):
        return {
            "risk_score_raw": round(self._raw_score, 4),
            "risk_score_smoothed": round(self.risk_score, 4),
        }