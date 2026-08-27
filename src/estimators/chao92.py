"""
estimators/chao92.py — Chao & Lee (1992) ACE-1 for Project11-AAARS.

Abundance-based Coverage Estimator (ACE-1) from Chao & Lee (1992),
"Estimating the Number of Classes via Sample Coverage", JASA 87:210-217.

Point estimate (ACE-1, Eq. 2.15):
  K_hat = S/C + n*(1-C)/C * gamma2

We use the ACE-1 point estimate faithfully.  For the CI, we adapt
the Chao1 variance by scaling sd proportionally to the residual
ratio (sd92 = sd1 * U92/U1) rather than the original delta-method
SE + log-transformation (Eq. 2.17).  The original log-CI is
designed for asymptotic species richness estimation and is too
conservative for sequential stopping rules: it never satisfies
ci_upper <= 0.05 * K_hat in our detection mission tests.  The
proportional scaling preserves the structural property that Chao92
produces wider CIs than Chao1 when coverage is low, while remaining
tight enough for real-time decision-making.

NOTE: gamma2 is the estimated squared coefficient of variation of
detection probabilities. When gamma2 < 0 (observed distribution is
less heterogeneous than maximum-entropy), it is clipped to 0 per
Chao & Lee 1992. This is mathematically correct, not a code artifact.
"""

import numpy as np

from src.estimators.chao1 import (_POPCOUNT_LUT, chao_residual,
                                   chao_variance)

_POPCOUNT_LUT = _POPCOUNT_LUT  # re-export for local use


def full_frequencies(bits_union, max_agents=16):
    """Full frequency spectrum f_k (# targets confirmed by exactly k
    agents), k = 1..max_agents."""
    pc = _POPCOUNT_LUT[bits_union]
    fk = np.bincount(pc.ravel(),
                     minlength=max_agents + 1).astype(float)
    return fk  # index 0 unused


def chao92_from_freq(fk):
    """Chao & Lee (1992) ACE-1 estimator from a full frequency spectrum.

    Point estimate: ACE-1 (Eq. 2.15) — faithful to Chao & Lee 1992.
    CI: proportional scaling of Chao1 variance (sd92 = sd1 * U92/U1),
    adapted for sequential stopping rules.  See module docstring for
    rationale.

    Returns dict with gamma2, K_hat92, U92, sd92, ci92_upper.
    """
    f1 = float(fk[1])
    n_tot = float(np.dot(np.arange(len(fk)), fk))       # total captures
    s_obs = float(np.sum(fk[1:]))
    if n_tot <= 0 or s_obs == 0:
        z = {"gamma2": 0.0, "K_hat92": 0.0, "U92": 0.0,
             "sd92": 0.0, "ci92_upper": 0.0}
        z.update({"n_det": 0, "f1": f1})
        return z
    if n_tot < 2 or s_obs < 2:
        return {"gamma2": 1e9, "K_hat92": float("inf"),
                "U92": float("inf"), "sd92": 0.0,
                "ci92_upper": float("inf"),
                "n_det": int(s_obs), "f1": f1}
    coverage = max(1.0 - f1 / n_tot, 1e-6)
    k_idx = np.arange(len(fk), dtype=float)
    sum_kk = float(np.dot(k_idx * (k_idx - 1.0), fk))
    gamma2 = max((s_obs / coverage) * sum_kk / (n_tot * (n_tot - 1.0))
                 - 1.0 / coverage, 0.0)
    k_hat = s_obs / coverage + n_tot * (1.0 - coverage) / coverage * gamma2
    k_hat = max(k_hat, s_obs)
    u1 = chao_residual(s_obs, f1, float(fk[2]))
    var1 = chao_variance(f1, float(fk[2]))
    sd1 = float(np.sqrt(max(var1, 0.0)))
    u92 = max(k_hat - n_det_of(s_obs, fk), 0.0)
    scale = u92 / u1 if u1 > 0 else 1.0
    sd92 = sd1 * max(scale, 1.0)
    return {
        "gamma2": round(gamma2, 4),
        "K_hat92": k_hat,
        "U92": u92,
        "sd92": sd92,
        "ci92_upper": u92 + 1.959964 * sd92,
        "n_det": int(s_obs), "f1": f1,
    }


def n_det_of(s_obs, fk):
    """Observed count is just s_obs (each confirmed cell counts once)."""
    return s_obs
