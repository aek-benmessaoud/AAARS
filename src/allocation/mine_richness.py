"""
mine_richness.py — Allocation policies for detection missions (V5).

AL-Richness  : bounded-BFS candidates scored by window scan-gap density +
               lambda * window confirmed-mine density (spatial clustering prior),
               / (BFS layer + eps). V4 lesson transplanted: signal lives in
               the target scoring, not in behavior switching.

LEAK-FREE: only local_seen/local_obs/local_scan_count/local_confirm_bits[aid].
"""

import numpy as np

from src.allocation.common import bounded_bfs, box_sum, explore_action
from src.analysis_utils import frontier_mask, select_target


# ======================================================================
# AL-Richness (proposed)
# ======================================================================

class MineRichnessPolicy:
    """utility(c) = [gap_w(c) + lam * susp_w(c)] / (D(c) + eps);

    gap_w  = # believed-free cells with zero local scans in window w
    susp_w = # confirmed mines in window w (clustering prior)
    """

    def __init__(self, seed=None, fov_radius=5, horizon=8, window=None,
                 eps=1.0, lam=0.5, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.lam = lam
        self.tie_eps = tie_eps
        self.fallback = 0
        self.random_walk = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        scan = env.local_scan_count[agent_id]
        bits = env.local_confirm_bits[agent_id]

        unknown = (~info["known"]).astype(np.float64)
        frontier = frontier_mask(info["known"], info["obs"], unknown)

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        free_known = info["known"] & ~info["obs"]
        gap = box_sum(free_known & (scan == 0), self.window)
        susp = box_sum(bits > 0, self.window)
        gain = gap + self.lam * susp

        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        utility[pos] = gain[pos] / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, frontier, self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            self.fallback += 1
            act, mode = explore_action(env, agent_id, self.rng)
            if mode == "explore_random":
                self.random_walk += 1
            return act, "Mine+Richness", mode, None
        return int(action), "Mine+Richness", "mine_richness", None
