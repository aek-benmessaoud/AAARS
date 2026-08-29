"""
realistic_policies.py — Natural, non-adversarial biased allocation policies.

These produce reward/spatial bias from the allocation algorithm itself
(rather than from knowing true target locations), to test AAARS when bias
emerges "for real". All are LEAK-FREE: they use only per-agent local
memory / scan counts / confirmed-mine evidence, never ground truth.

Policy A  FrontierPoorCoord : agents crowd the same frontier blobs
                              (poor cross-agent coordination).
Policy B  GreedyCoverage     : each agent selfishly maximizes its OWN new
                              coverage, ignoring the others (duplication).
Policy C  HotspotPatrol      : reward-based patrolling — agents revisit
                              local regions where mines were confirmed.
"""

import numpy as np

from src.allocation.common import (
    bounded_bfs, box_sum, explore_action, valid_neighbors_local)
from src.analysis_utils import frontier_mask, select_target


# ======================================================================
# Policy A — frontier exploration with poor coordination
# ======================================================================
class FrontierPoorCoordPolicy:
    """Each agent BFS-targets the frontier cell with the largest local
    unvisited "gap" in a window. Because local memory is similar near other
    agents (limited comm) and all agents score the same large frontier blobs,
    they converge onto the same blobs instead of spreading -> under-coverage
    elsewhere (spatial, detection-independent bias)."""

    def __init__(self, seed=None, fov_radius=5, horizon=10, window=None,
                 eps=1.0, non_frontier_penalty=0.5, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.non_frontier_penalty = non_frontier_penalty
        self.tie_eps = tie_eps
        self.fallback = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        scan = env.local_scan_count[agent_id]
        unknown = (~info["known"]).astype(np.float64)

        frontier = frontier_mask(info["known"], info["obs"], unknown)
        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)

        free_known = info["known"] & ~info["obs"]
        gap = box_sum(free_known & (scan == 0), self.window)

        # Score: frontier cells priority (their window-gap), non-frontier
        # devalued so agents stay at the frontier rather than filling gaps.
        base = gap + (1.0 - frontier) * (-self.non_frontier_penalty)
        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        utility[pos] = base[pos] / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, frontier, self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            self.fallback += 1
            act, mode = explore_action(env, agent_id, self.rng)
            return act, "Frontier", mode, None
        return int(action), "Frontier", "frontier_poorcoord", None


# ======================================================================
# Policy B — selfish greedy coverage (info-gain, no coordination)
# ======================================================================
class GreedyCoveragePolicy:
    """Each agent greedily moves to the nearby cell that maximizes the
    number of NEW cells it would scan (its own local scan gaps in a window),
    ignoring all other agents. Overlapping FOVs lead to duplicated coverage
    and rich regions being re-scanned while others wait."""

    def __init__(self, seed=None, fov_radius=5, horizon=4, window=None,
                 eps=1.0, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.tie_eps = tie_eps
        self.stagnate = 0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        scan = env.local_scan_count[agent_id]

        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)
        free_known = info["known"] & ~info["obs"]
        gap = box_sum(free_known & (scan == 0), self.window)

        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        utility[pos] = gap[pos] / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, pos, self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            act, mode = explore_action(env, agent_id, self.rng)
            return act, "GreedyCov", mode, None
        return int(action), "GreedyCov", "greedy_cover", None



# ======================================================================
# Policy C — reward-based hotspot patrolling
# ======================================================================
class HotspotPatrolPolicy:
    """Reward-based patrolling: each agent moves toward the nearest local
    region with confirmed mines (local_confirm_bits), i.e. it patrols back to
    rich areas. Detection-dependent but uses only LOCAL confirmed evidence
    (no ground truth), emulating an operator re-checking hot zones."""

    def __init__(self, seed=None, fov_radius=5, horizon=10, window=None,
                 eps=1.0, lam=0.8, tie_eps=1e-3):
        self.rng = np.random.default_rng(seed)
        self.fov_radius = fov_radius
        self.horizon = horizon
        self.window = window if window is not None else fov_radius
        self.eps = eps
        self.lam = lam
        self.tie_eps = tie_eps
        self.explored_frac = 0.0

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov_radius)
        info = env.get_local_info(agent_id)
        bits = env.local_confirm_bits[agent_id]
        unknown = (~info["known"]).astype(np.float64)

        frontier = frontier_mask(info["known"], info["obs"], unknown)
        D, curdir = bounded_bfs(env, agent_id, max_depth=self.horizon)

        # Hotspot reward: window count of locally-confirmed mines.
        hot = box_sum(bits > 0, self.window)

        utility = np.full_like(D, -np.inf, dtype=np.float64)
        pos = D > 0
        # Tie frontier exploration with hotspot patrolling.
        utility[pos] = (hot[pos] * self.lam + frontier[pos]) / (D[pos] + self.eps)

        action = select_target(utility, D, curdir, (D > 0), self.rng,
                               tie_eps=self.tie_eps)
        if action is None:
            act, mode = explore_action(env, agent_id, self.rng)
            return act, "Hotspot", mode, None
        return int(action), "Hotspot", "hotspot_patrol", None
