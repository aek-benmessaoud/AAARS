"""
confidence_bounds.py — Risk-modulated confidence bounds for AAARS.

The key innovation: instead of choosing between Chao1 and Chao92,
AAARS blends them continuously using a risk score r(t) in [0,1].

  â(t) = (1 - r) * U1(t) + r * U92(t)        [blended point estimate]
  σ̂²(t) = (1 - r) * Var1(t) + r * Var92(t)   [blended variance]
  α_adj(t) = 0.05 * (1 + λ * r(t))            [adaptive threshold]

Stop when: â(t) + z * σ̂(t) <= α_adj(t) * (n_det + â(t))

When r=0: behaves like Chao1 (efficient, for dispersed observations)
When r=1: behaves like Chao92 (conservative, for clustered observations)
When 0<r<1: genuinely new weighted blend
"""

import numpy as np
from src.estimators.chao1 import chao_residual, chao_variance
from src.estimators.chao92 import chao92_from_freq, full_frequencies


Z_95 = 1.959964  # z for 95% one-sided CI


def blended_estimate(bits_union, risk_score, blend_threshold=0.2):
    """Compute the AAARS risk-modulated estimate.

    Args:
        bits_union: uint8 grid of fleet block bits
        risk_score: float in [0, 1], where 0=Chao1-like, 1=Chao92-like
        blend_threshold: risk below this → pure Chao1 (no blend)

    Returns dict with:
        n_det, f1, f2: basic frequency stats
        U_blend: blended point estimate of undetected targets
        sd_blend: blended standard deviation
        ci_upper: upper confidence bound (U_blend + z * sd_blend)
        K_hat: estimated total (n_det + U_blend)
        recall_est: estimated recall (n_det / K_hat)
        U1, U92: individual estimates for logging
        gamma2: Chao92 heterogeneity measure
    """
    from src.estimators.chao1 import frequency_from_bits

    n_det, f1, f2 = frequency_from_bits(bits_union)

    # Chao1 components
    U1 = chao_residual(n_det, f1, f2)
    Var1 = chao_variance(f1, f2)

    # Chao92 components
    fk = full_frequencies(bits_union)
    est92 = chao92_from_freq(fk)
    U92 = est92["U92"]
    # Approximate Chao92 variance from its sd
    Var92 = est92["sd92"] ** 2

    # Blended estimate with dead zone
    r = float(np.clip(risk_score, 0.0, 1.0))
    if r < blend_threshold:
        r_eff = 0.0
    else:
        r_eff = (r - blend_threshold) / (1.0 - blend_threshold)
    U_blend = (1.0 - r_eff) * U1 + r_eff * U92
    Var_blend = (1.0 - r_eff) * Var1 + r_eff * Var92
    sd_blend = float(np.sqrt(max(Var_blend, 0.0)))

    ci_upper = U_blend + Z_95 * sd_blend
    K_hat = n_det + U_blend
    recall_est = n_det / K_hat if K_hat > 0 else 0.0

    return {
        "n_det": n_det, "f1": f1, "f2": f2,
        "U_blend": U_blend, "sd_blend": sd_blend,
        "ci_upper": ci_upper, "K_hat": K_hat, "recall_est": recall_est,
        "U1": U1, "U92": U92,
        "gamma2": est92.get("gamma2", 0.0),
    }


def should_stop(blended_est, alpha_adj=0.05):
    """Evaluate the AAARS stopping condition.

    Stop when: ci_upper <= alpha_adj * K_hat
    i.e., the upper residual bound is at most alpha_adj fraction of estimated total.

    Returns (should_stop: bool, adjusted_ci: float)
    """
    ci_upper = blended_est["ci_upper"]
    K_hat = blended_est["K_hat"]
    if K_hat <= 0:
        return False, ci_upper
    return ci_upper <= alpha_adj * K_hat, ci_upper


def adaptive_threshold(risk_score, base_alpha=0.05, lam=1.0):
    """Compute risk-adjusted alpha threshold.

    alpha_adj = base_alpha / (1 + lam * risk_score)

    When risk is high (clustered observations), we require tighter confidence
    (smaller residual fraction) before certifying.
    """
    return base_alpha / (1.0 + lam * risk_score)
