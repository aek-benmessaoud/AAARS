"""
estimators/chao1.py — Chao (1987) estimator for Project11-AAARS.

Capture occasions = AGENTS (model M_h): each agent sweeping an undetected
mine is an independent Bernoulli capture into that agent's list. A target
confirmed by c_j distinct agents contributes to f_{c_j}:

    f1 = # confirmed targets captured by exactly 1 agent
    f2 = # confirmed targets captured by exactly 2 agents

Chao (1987) lower-bound on unseen mass (bias-corrected form):

    U   = f1 (f1 - 1) / (2 (f2 + 1))
    Var = f2 [ (f1/f2)^2 / 4 + (f1/f2)^3 / 4 ]      (f2 > 0)

Inputs come exclusively from env.confirm_bits_union(env) (popcounts of
confirmation bitmasks shared through rendezvous fusion) -> LEAK-FREE:
missed contacts are unobservable by construction.
"""

import numpy as np


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)],
                          dtype=np.uint8)


def frequency_from_bits(bits_union):
    """bits_union: uint8 grid, bit i set if agent i confirmed the cell.

    f_k counts cells whose POPCOUNT equals k (captured by exactly k agents),
    not raw bitmask values. Returns (n_detected, f1, f2)."""
    pc = _POPCOUNT_LUT[bits_union]
    n_det = int(np.count_nonzero(pc))
    f1 = int(np.count_nonzero(pc == 1))
    f2 = int(np.count_nonzero(pc == 2))
    return n_det, float(f1), float(f2)


def chao_residual(n_det, f1, f2):
    """Bias-corrected Chao lower bound on UNDETECTED targets."""
    U = (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))
    return max(float(U), 0.0)


def chao_variance(f1, f2):
    """Chao (1987) variance of U (0 if f2 == 0)."""
    if f2 <= 0:
        return 0.0
    r = f1 / f2
    return float(f2 * (r * r / 4.0 + r ** 3 / 4.0))


def residual_estimate(bits_union):
    """Returns dict: n_det, f1, f2, U_hat, sd, ci_upper, K_hat, recall_est."""
    n_det, f1, f2 = frequency_from_bits(bits_union)
    U = chao_residual(n_det, f1, f2)
    var = chao_variance(f1, f2)
    sd = float(np.sqrt(var))
    ci_upper = U + 1.959964 * sd
    K_hat = n_det + U
    recall_est = n_det / K_hat if K_hat > 0 else 0.0
    return {
        "n_det": n_det, "f1": f1, "f2": f2,
        "U_hat": U, "sd": sd, "ci_upper": ci_upper,
        "K_hat": K_hat, "recall_est": recall_est,
    }
