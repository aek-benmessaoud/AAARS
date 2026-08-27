"""Chao1-CI stopping rule: stop when ci_upper <= alpha * K_hat."""

from src.estimators.chao1 import residual_estimate

DEFAULT_N_MIN = 5
DEFAULT_CI_REL = 0.05

def chao1_ci_stop(bits_union, n_min=DEFAULT_N_MIN, ci_rel=DEFAULT_CI_REL):
    """Evaluate Chao1-CI stopping condition.

    Requires minimum evidence gate: n_det >= n_min AND f2 >= max(5, ceil(0.08*n_det)).

    Returns:
        (should_stop: bool, estimate: dict)
    """
    est = residual_estimate(bits_union)
    n_det = est["n_det"]
    f2 = est["f2"]
    f2_floor = max(5.0, __import__('numpy').ceil(0.08 * n_det))
    armed = n_det >= n_min and f2 >= f2_floor
    if armed and est["ci_upper"] <= ci_rel * est["K_hat"]:
        return True, est
    return False, est
