"""
grid_env.py — Grid environment for UAV swarm coverage. Project11-AAARS.
V3 (comm_limited info model).

INFO MODEL
----------
comm_limited  (V3 default): limited-range communication with proximity-triggered
  (rendezvous) fusion. Each agent keeps a private memory:
  - agent_own_visit_count[agent_id]: counts only cells THIS agent visited.
  - local_visit_count[agent_id]    : what the agent "knows" — initialized to
    its own counts, updated by FUSION when agents meet within COMM_RANGE.
  - local_seen_mask[agent_id]      : cells ever seen inside its FOV (or a
    neighbor's FOV via fusion) — occupancy mask.
  - local_obstacle_map[agent_id]   : obstacles ever seen (FOV + fusion).
  Fusion is symmetric: both agents keep the combined knowledge (visit = MAX,
  seen/obstacle = union). No global visit_count / obstacle_map is ever fed to
  a policy. Collision resolution is handled by the environment from REAL agent
  positions (position-only channel), fully independent of COMM_RANGE/fusion.

pure_local  (V2 reference): zero communication — each agent perceives ONLY its
  own history. Same structure, no fusion ever fires.

fov_perfect (ablation): perfect knowledge of the whole grid (global obstacle
  map + global visit counts). Used as the upper bound of the spectrum.

The canonical spectrum: pure_local <= comm_limited <= fov_perfect.
"""

import numpy as np


class GridEnv:
    """
    2-D grid world shared by all UAV agents.

    Action encoding: 0=up, 1=down, 2=left, 3=right, 4=stay
    """

    def __init__(self, grid_size=100, num_agents=6, obstacle_ratio=0.05,
                 seed=0, info_model="pure_local", p_miss=0.0, sigma_loc=0.0,
                 comm_range=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacle_ratio = obstacle_ratio
        self.seed = seed
        self.info_model = info_model
        self.p_miss = p_miss
        self.sigma_loc = sigma_loc
        if info_model == "comm_limited" and comm_range is None:
            comm_range = 5.0
        self.comm_range = comm_range

        self.rng = np.random.default_rng(seed)
        self._build_obstacle_map()

        # --- GLOBAL ground truth (metrics ONLY, never fed to policies) ---
        self.visit_count = np.zeros((grid_size, grid_size), dtype=np.int32)

        # --- PER-AGENT OWN MEMORY ---
        # Only incremented when the agent itself steps onto the cell.
        self.agent_own_visit_count = [
            np.zeros((grid_size, grid_size), dtype=np.int32)
            for _ in range(num_agents)
        ]
        # What the agent "knows": own visits, augmented by rendezvous fusion
        # (comm_limited). For pure_local it stays == own counts.
        self.local_visit_count = [
            np.zeros((grid_size, grid_size), dtype=np.int32)
            for _ in range(num_agents)
        ]
        # Cells ever seen inside FOV (and, for comm_limited, via fusion).
        self.local_seen_mask = [
            np.zeros((grid_size, grid_size), dtype=bool)
            for _ in range(num_agents)
        ]
        # Obstacles ever seen inside FOV (unknown cells are treated as free).
        self.local_obstacle_map = [
            np.zeros((grid_size, grid_size), dtype=bool)
            for _ in range(num_agents)
        ]

        # Number of symmetric pair-merges performed (comm_limited only).
        self.fusion_events_count = 0
        self.last_rendezvous_pairs = 0

        # Frontier BFS exhaustion cache (per-agent). When a policy's BFS
        # exhausts (no locally-unvisited cell reachable), the reachable
        # component can only shrink while the unvisited/obstacle knowledge is
        # unchanged, so the "no reachable target" result stays valid. The
        # masks are snapshots used to invalidate the cache. Behavior-identical.
        self.explore_exhausted = [False] * num_agents
        self.explore_exhausted_unvisited = [None] * num_agents
        self.explore_exhausted_obs = [None] * num_agents

        # --- Noise bookkeeping ---
        self.noise_rngs = [
            np.random.default_rng(seed + 1000 + i) for i in range(num_agents)
        ]
        self.loc_error_sum = 0.0
        self.loc_error_count = 0

        self.traversable = int(np.sum(~self.obstacle_map))
        self._place_agents()
        self.last_collision_count = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_obstacle_map(self):
        """Random obstacles. Outer edge ring left free for spawn safety."""
        total = self.grid_size * self.grid_size
        inner = [(r, c) for r in range(1, self.grid_size - 1)
                 for c in range(1, self.grid_size - 1)]
        n_obs = int(total * self.obstacle_ratio)
        flat = np.zeros(total, dtype=bool)
        cand = self.rng.choice(len(inner), size=n_obs, replace=False)
        for idx in cand:
            r, c = inner[idx]
            flat[r * self.grid_size + c] = True
        self.obstacle_map = flat.reshape(self.grid_size, self.grid_size)

    def _place_agents(self):
        free = list(zip(*np.where(~self.obstacle_map)))
        replace_flag = len(free) < self.num_agents
        chosen = self.rng.choice(len(free), size=self.num_agents,
                                 replace=replace_flag)
        self.agent_positions = [list(free[i]) for i in chosen]
        for aid, (r, c) in enumerate(self.agent_positions):
            self.visit_count[r, c] += 1
            self.agent_own_visit_count[aid][r, c] += 1
            self.local_visit_count[aid][r, c] += 1

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def step(self, actions):
        """Move each agent. No stacking: contested cells keep one random
        winner, losers stay in place."""
        intended = []
        for agent_id, action in enumerate(actions):
            dr, dc = self.DELTAS[action]
            r, c = self.agent_positions[agent_id]
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and not self.obstacle_map[nr, nc]):
                intended.append((nr, nc))
            else:
                intended.append((r, c))

        from collections import defaultdict
        claimants = defaultdict(list)
        for agent_id, cell in enumerate(intended):
            claimants[cell].append(agent_id)

        self.last_collision_count = 0
        for cell, agents in claimants.items():
            if len(agents) > 1:
                self.last_collision_count += len(agents) - 1
                winner = int(self.rng.choice(agents))
                for loser in agents:
                    if loser != winner:
                        intended[loser] = self.agent_positions[loser][:]

        for agent_id in range(self.num_agents):
            nr, nc = intended[agent_id]
            self.agent_positions[agent_id] = [nr, nc]
            self.visit_count[nr, nc] += 1
            self.agent_own_visit_count[agent_id][nr, nc] += 1
            self.local_visit_count[agent_id][nr, nc] += 1

        return [pos[:] for pos in self.agent_positions]

    # ------------------------------------------------------------------
    # Communication (comm_limited): proximity-triggered fusion
    # ------------------------------------------------------------------

    def distance(self, i, j):
        """Euclidean distance between agents i and j (units = grid cells)."""
        return float(np.linalg.norm(
            np.asarray(self.agent_positions[i], dtype=float)
            - np.asarray(self.agent_positions[j], dtype=float)))

    def _within_comm_range(self, i, j):
        if self.comm_range is None:
            return False
        return self.distance(i, j) <= float(self.comm_range)

    def check_and_merge(self, agent_id_list=None):
        """Check all pairs of agents within COMM_RANGE and fuse their maps.
        Called ONCE per decision step, BEFORE FOV perception (§5.3 order).
        No-op for info models other than comm_limited."""
        if self.info_model != "comm_limited":
            return 0
        import itertools
        if agent_id_list is None:
            agent_id_list = list(range(self.num_agents))
        n_fusions = 0
        for i, j in itertools.combinations(agent_id_list, 2):
            if self._within_comm_range(i, j):
                self.merge_maps(i, j)
                n_fusions += 1
        self.last_rendezvous_pairs = n_fusions
        return n_fusions

    def merge_maps(self, i, j):
        """
        Symmetric merge: both agents leave with the COMBINED knowledge.
        - Visits  : element-wise MAX (never a sum, to avoid inflating F1/F2).
        - Seen    : boolean union.
        - Obstacles: boolean union (occupancy is shared geometry).
        The merge is idempotent per pair; the counter counts pair-merges.
        """
        merged_visit = np.maximum(self.local_visit_count[i],
                                  self.local_visit_count[j])
        merged_seen = self.local_seen_mask[i] | self.local_seen_mask[j]
        merged_obs = self.local_obstacle_map[i] | self.local_obstacle_map[j]

        self.local_visit_count[i] = merged_visit.copy()
        self.local_visit_count[j] = merged_visit.copy()
        self.local_seen_mask[i] = merged_seen
        self.local_seen_mask[j] = merged_seen.copy()
        self.local_obstacle_map[i] = merged_obs
        self.local_obstacle_map[j] = merged_obs.copy()

        self.fusion_events_count += 1

    # ------------------------------------------------------------------
    # FOV / memory
    # ------------------------------------------------------------------

    def get_fov_mask(self, r, c, fov_radius):
        """Square FOV window centred at (r, c), clamped to grid bounds."""
        gs = self.grid_size
        mask = np.zeros((gs, gs), dtype=bool)
        r0 = max(0, r - fov_radius)
        r1 = min(gs, r + fov_radius + 1)
        c0 = max(0, c - fov_radius)
        c1 = min(gs, c + fov_radius + 1)
        mask[r0:r1, c0:c1] = True
        return mask

    def update_local_memory(self, agent_id, fov_radius):
        """
        Merge visible cells into the agent's local memory.
        pure_local: only marks seen + obstacles (NEVER copies visit counts).
        fov_perfect: full knowledge (ablation).
        """
        if self.info_model == "fov_perfect":
            self.local_seen_mask[agent_id][:] = ~self.obstacle_map
            self.local_obstacle_map[agent_id][:] = self.obstacle_map
            return

        x_obs, y_obs = self._observed_position(agent_id)
        fov = self.get_fov_mask(x_obs, y_obs, fov_radius)

        if self.p_miss > 0:
            rng = self.noise_rngs[agent_id]
            noise = rng.random(fov.shape) > self.p_miss
            fov = fov & noise

        self.local_seen_mask[agent_id][fov] = True
        self.local_obstacle_map[agent_id][fov] = self.obstacle_map[fov]

    def _observed_position(self, agent_id):
        """True position possibly corrupted by localisation noise."""
        r, c = self.agent_positions[agent_id]
        if self.sigma_loc <= 0:
            return int(r), int(c)
        rng = self.noise_rngs[agent_id]
        x_f = r + rng.normal(0, self.sigma_loc)
        y_f = c + rng.normal(0, self.sigma_loc)
        self.loc_error_sum += (x_f - r) ** 2 + (y_f - c) ** 2
        self.loc_error_count += 1
        return int(round(x_f)), int(round(y_f))

    def get_rmse_2d(self):
        if self.loc_error_count == 0:
            return 0.0
        return np.sqrt(self.loc_error_sum / self.loc_error_count)

    # ------------------------------------------------------------------
    # Knowledge interface (single access point for policies)
    # ------------------------------------------------------------------

    def get_visit_knowledge(self, agent_id):
        """Visit counts the agent may legitimately use."""
        if self.info_model == "fov_perfect":
            return self.visit_count
        if self.info_model == "comm_limited":
            return self.local_visit_count[agent_id]
        return self.agent_own_visit_count[agent_id]

    def get_known_mask(self, agent_id):
        """Cells the agent knows about (traversable-candidate set)."""
        if self.info_model == "fov_perfect":
            return ~self.obstacle_map
        return self.local_seen_mask[agent_id]

    def get_obstacle_knowledge(self, agent_id):
        """Obstacles the agent knows about."""
        if self.info_model == "fov_perfect":
            return self.obstacle_map
        return self.local_obstacle_map[agent_id]

    def get_total_unknown(self, agent_id):
        """Number of cells the agent believes are still to be covered.
        Used to cap ACE-U. Always expressed in the agent's frame."""
        if self.info_model == "fov_perfect":
            return int(np.sum((self.visit_count == 0) & ~self.obstacle_map))
        known = self.local_seen_mask[agent_id]
        obs = self.local_obstacle_map[agent_id]
        own = self.get_visit_knowledge(agent_id)
        traversable_known = known & ~obs
        return int(np.sum(traversable_known & (own == 0)))

    def get_local_info(self, agent_id):
        """Convenience bundle used by all estimators/policies."""
        return {
            "visit": self.get_visit_knowledge(agent_id),
            "known": self.get_known_mask(agent_id),
            "obs": self.get_obstacle_knowledge(agent_id),
        }

    def get_local_unvisited_mask(self, agent_id):
        """Targets for frontier exploration: known traversable cells the
        agent has not itself visited."""
        info = self.get_local_info(agent_id)
        return info["known"] & ~info["obs"] & (info["visit"] == 0)

    # ------------------------------------------------------------------
    # Coverage helpers (metrics use GLOBAL truth)
    # ------------------------------------------------------------------

    def global_coverage(self):
        visited = int(np.sum((self.visit_count > 0) & ~self.obstacle_map))
        return 100.0 * visited / self.traversable if self.traversable else 0.0

    def local_coverage(self, agent_id):
        """Fraction of the agent's known traversable cells it has visited."""
        info = self.get_local_info(agent_id)
        tr = info["known"] & ~info["obs"]
        total = int(np.sum(tr))
        if total == 0:
            return 0.0
        visited = int(np.sum((info["visit"][tr] > 0)))
        return 100.0 * visited / total

    def reset(self):
        self.visit_count[:] = 0
        for i in range(self.num_agents):
            self.agent_own_visit_count[i][:] = 0
            self.local_visit_count[i][:] = 0
            self.local_seen_mask[i][:] = False
            self.local_obstacle_map[i][:] = False
        self.fusion_events_count = 0
        self.last_rendezvous_pairs = 0
        self.explore_exhausted = [False] * self.num_agents
        self.explore_exhausted_unvisited = [None] * self.num_agents
        self.explore_exhausted_obs = [None] * self.num_agents
        self.loc_error_sum = 0.0
        self.loc_error_count = 0
        self._place_agents()


class NoisyGridEnv(GridEnv):
    """Alias kept for backward compatibility with run scripts."""
