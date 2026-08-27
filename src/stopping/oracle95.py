"""Oracle-95 stopping rule: post-hoc, uses ground truth. For comparison only."""
from src.estimators.chao1 import Z_95  # not actually needed, just for consistency

def oracle_95_stop(true_found_series, num_mines, r_star=0.95):
    """First step where true_found >= r_star * num_mines.

    Args:
        true_found_series: list of (t, count) tuples
        num_mines: total number of mines
        r_star: recall threshold (default 0.95)

    Returns:
        stop_step: int or None if threshold never reached
    """
    for t, found in true_found_series:
        if found >= r_star * num_mines:
            return t
    return None
