"""Chao92-CI stopping rule: stop when ci92_upper <= alpha * K_hat92."""

from src.estimators.chao92 import chao92_from_freq, full_frequencies

DEFAULT_N_MIN = 5
DEFAULT_CI_REL = 0.05

def chao92_ci_stop(bits_union, n_min=DEFAULT_N_MIN, ci_rel=DEFAULT_CI_REL):
    """Evaluate Chao92-CI stopping condition.

    Same arming gate as Chao1-CI for fair comparison.

    Returns:
        (should_stop: bool, estimate: dict)
    """
    fk = full_frequencies(bits_union)
    est = chao92_from_freq(fk)
    n_det = est["n_det"]
    f1 = est["f1"]
    f2 = float(fk[2]) if len(fk) > 2 else 0.0
    f2_floor = max(5.0, __import__('numpy').ceil(0.08 * n_det))
    armed = n_det >= n_min and f2 >= f2_floor
    if armed and est["U92"] >= 0 and est["ci92_upper"] <= ci_rel * est["K_hat92"]:
        return True, est
    return False, est
