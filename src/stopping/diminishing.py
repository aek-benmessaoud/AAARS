"""Diminishing-returns stopping rule: no new confirmation for TAU steps."""

DEFAULT_TAU = 150
DEFAULT_N_MIN = 5

class DiminishingStop:
    """Track last-find time and fire when no new mine found for TAU steps."""

    def __init__(self, tau=DEFAULT_TAU, n_min=DEFAULT_N_MIN):
        self.tau = tau
        self.n_min = n_min
        self.last_find_t = 0
        self.prev_n = 0

    def update(self, t, n_belief):
        """Call every step with current time and belief n_det.

        Returns:
            bool: True if stopping condition met
        """
        if n_belief > self.prev_n:
            self.last_find_t = t
        self.prev_n = n_belief
        return (t - self.last_find_t >= self.tau and n_belief >= self.n_min)

    def reset(self):
        self.last_find_t = 0
        self.prev_n = 0
