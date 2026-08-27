"""
mine_env.py — MineGridEnv: detection-mission extension of GridEnv (V5).

Scenario (PLAN_V5 §4, amended §4b): K mines hidden on free cells. Capture
occasions are SURVEY BLOCKS (fixed-width step windows = independent passes),
not agents — free-roaming allocations produce almost no cross-AGENT overlap
(f2 ~ 0), which makes agent-wise capture-recapture vacuous. Block-wise
occasions mirror real multi-pass demining clearance.

Verification protocol: a mine is LOCATED on first detection; it is
NEUTRALIZED only after hits in >= conf_blocks distinct blocks (independent
re-detection). Until then it stays capturable -> temporal f2 > 0 emerges
naturally once coverage repeats.

LEAK RULES (enforced by tests):
  - Policies/estimators may ONLY read: local_seen/local_obs (inherited),
    local_scan_count, local_found, local_confirm_bits, block_bits.
  - mine_mask / located_mask / detect_prob / pass_count / neutralized_mask
    are ground truth for metrics only.

Belief arrays per agent (fusion merges them at rendezvous):
  local_scan_count[aid]  int32 : # times this agent's FOV covered the cell
  local_found[aid]       bool  : cells THIS agent confirmed (own only)
  local_confirm_bits[aid] uint16: bit i set if agent i confirmed that cell
  block_bits             uint8 : FLEET-level; bit b set if mine hit in block b
                                   (hits are observable events; fusion shares)
"""

import numpy as np

from src.environment.grid_env import GridEnv

_PC_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class MineGridEnv(GridEnv):

    def __init__(self, grid_size=100, num_agents=6, obstacle_ratio=0.05,
                 seed=0, info_model="comm_limited", p_miss=0.0,
                 sigma_loc=0.0, comm_range=None,
                 num_mines=60, detectability="homogeneous",
                 p_bar=0.7, strata=(0.9, 0.6, 0.3), persistent=False,
                 conf_blocks=2, occ_cooldown=50):
        self.num_mines = num_mines
        self.detectability = detectability
        self.p_bar = p_bar
        self.strata = tuple(strata)
        self.persistent = persistent
        # Verification protocol: neutralize a mine only when it was hit in
        # >= conf_blocks DISTINCT occasions (re-detections separated by at
        # least occ_cooldown steps).
        self.conf_blocks = conf_blocks
        self.occ_cooldown = occ_cooldown
        super().__init__(grid_size=grid_size, num_agents=num_agents,
                         obstacle_ratio=obstacle_ratio, seed=seed,
                         info_model=info_model, p_miss=p_miss,
                         sigma_loc=sigma_loc, comm_range=comm_range)

    # ------------------------------------------------------------------
    # Setup (after super() built obstacles + placed agents)
    # ------------------------------------------------------------------

    def _build_belief_arrays(self):
        gs = self.grid_size
        self.local_scan_count = [
            np.zeros((gs, gs), dtype=np.int32) for _ in range(self.num_agents)
        ]
        self.local_found = [
            np.zeros((gs, gs), dtype=bool) for _ in range(self.num_agents)
        ]
        self.local_confirm_bits = [
            # uint16: bit i set if agent i confirmed that cell (supports
            # fleets up to 16 agents; fusion ORs these masks).
            np.zeros((gs, gs), dtype=np.uint16) for _ in range(self.num_agents)
        ]

    def _place_mines(self):
        free = np.argwhere(~self.obstacle_map)
        agent_cells = {tuple(p) for p in self.agent_positions}
        cand = np.array([c for c in free if tuple(c) not in agent_cells])
        idx = self.rng.choice(len(cand), size=self.num_mines, replace=False)
        cells = cand[idx]
        self.mine_mask = np.zeros((self.grid_size,) * 2, dtype=bool)
        self.mine_mask[cells[:, 0], cells[:, 1]] = True
        # Per-mine detectability strata (M_h heterogeneity)
        self.detect_prob = np.zeros((self.grid_size,) * 2, dtype=np.float64)
        if self.detectability == "homogeneous":
            self.detect_prob[self.mine_mask] = self.p_bar
        else:
            k = len(self.strata)
            assign = self.rng.integers(0, k, size=len(cells))
            probs = np.asarray(self.strata, dtype=np.float64)[assign]
            self.detect_prob[cells[:, 0], cells[:, 1]] = probs
        # Ground truth (metrics only)
        self.located_mask = np.zeros((self.grid_size,) * 2, dtype=bool)
        self.neutralized_mask = np.zeros((self.grid_size,) * 2, dtype=bool)
        # Capture-recapture over OCCASIONS (observable beliefs):
        # bit b of block_bits[r,c] = mine at (r,c) hit in occasion b
        # (= detection event separated from the previous by occ_cooldown).
        self.block_bits = np.zeros((self.grid_size,) * 2, dtype=np.uint8)
        self.occasion_idx = np.full((self.grid_size,) * 2, -1, dtype=np.int8)
        self.last_occasion_step = np.full((self.grid_size,) * 2, -10 ** 9,
                                          dtype=np.int64)
        self.pass_count = np.zeros((self.grid_size,) * 2, dtype=np.int32)
        self.contact_count = np.zeros((self.grid_size,) * 2, dtype=np.int32)
        self.detection_step = {}          # (r,c) -> step of first confirmation
        self.n_detections = 0
        self.false_positives = 0

    # GridEnv.__init__ calls _place_agents(); we hook AFTER it by overriding
    # the tail of __init__ via this wrapper (called once, at construction).

    def __init_post__(self):
        pass

    # ------------------------------------------------------------------
    # Perception with detection opportunities (overrides GridEnv)
    # ------------------------------------------------------------------

    def update_local_memory(self, agent_id, fov_radius):
        super().update_local_memory(agent_id, fov_radius)
        x, y = self._observed_position(agent_id)
        fov = self.get_fov_mask(x, y, fov_radius)
        rng = self.noise_rngs[agent_id]

        self.local_scan_count[agent_id][fov] += 1

        # A located mine stays ACTIVE until verified (hit in conf_blocks
        # DISTINCT occasions). A new occasion opens for a mine when it is
        # swept again after OCC_COOLDOWN steps since its previous occasion
        # — repeated-survey capture design; hovering does not fabricate
        # independent evidence.
        active = self.mine_mask & fov & ~self.neutralized_mask
        if active.any():
            rs, cs = np.nonzero(active)
            p = self.detect_prob[rs, cs]
            hits = rng.random(len(rs)) < p
            self.pass_count[rs, cs] += 1
            self.contact_count[rs, cs] += 1
            hit_cells = (rs[hits], cs[hits])
            for k in range(len(hit_cells[0])):
                r, c = int(hit_cells[0][k]), int(hit_cells[1][k])
                if not self.located_mask[r, c]:
                    self.located_mask[r, c] = True
                    self.n_detections += 1
                    self.detection_step[(r, c)] = self.step_index
                    self.occasion_idx[r, c] = 0
                    self.last_occasion_step[r, c] = self.step_index
                elif self.step_index - self.last_occasion_step[r, c] \
                        >= self.occ_cooldown and \
                        self.occasion_idx[r, c] < 7:
                    self.occasion_idx[r, c] += 1
                    self.last_occasion_step[r, c] = self.step_index
                if not self.persistent:
                    self.local_found[agent_id][r, c] = True
                    self.local_confirm_bits[agent_id][r, c] |= (1 << agent_id)
                self.block_bits[r, c] |= (1 << self.occasion_idx[r, c])

        # Clearance authority: neutralize mines hit in >= conf_blocks
        # DISTINCT occasions (beliefs only — leak-free).
        pc = _PC_LUT[self.block_bits]
        newly_done = self.mine_mask & ~self.neutralized_mask & \
            (pc >= self.conf_blocks)
        if newly_done.any():
            self.neutralized_mask |= newly_done

    def step(self, actions):
        out = super().step(actions)
        # Base GridEnv does NOT maintain step_index; the mission clock and
        # occasion cooldowns derive from it.
        self.step_index = getattr(self, "step_index", 0) + 1
        return out

    # ------------------------------------------------------------------
    # Fusion overrides (union of confirmations + MAX scan counts)
    # ------------------------------------------------------------------

    def merge_maps(self, i, j):
        super().merge_maps(i, j)
        np.maximum(self.local_scan_count[i], self.local_scan_count[j],
                   out=self.local_scan_count[i])
        self.local_scan_count[j] = self.local_scan_count[i].copy()
        fi = self.local_found[i] | self.local_found[j]
        bi = self.local_confirm_bits[i] | self.local_confirm_bits[j]
        self.local_found[i] = fi
        self.local_found[j] = fi.copy()
        self.local_confirm_bits[i] = bi
        self.local_confirm_bits[j] = bi.copy()

    # ------------------------------------------------------------------
    # Fleet-level belief summaries (leak-free: beliefs only)
    # ------------------------------------------------------------------

    def fleet_confirm_bits(self):
        bits = self.local_confirm_bits[0].copy()
        for a in range(1, self.num_agents):
            bits |= self.local_confirm_bits[a]
        return bits

    def fleet_block_bits(self):
        """Observable per-mine occasion-capture bitmasks."""
        return self.block_bits

    def fleet_found(self):
        found = self.local_found[0].copy()
        for a in range(1, self.num_agents):
            found |= self.local_found[a]
        return found

    def fleet_scan_max(self):
        sc = self.local_scan_count[0].copy()
        for a in range(1, self.num_agents):
            np.maximum(sc, self.local_scan_count[a], out=sc)
        return sc

    def fleet_known_free(self):
        known = self.local_seen_mask[0].copy()
        obs = self.local_obstacle_map[0].copy()
        for a in range(1, self.num_agents):
            known |= self.local_seen_mask[a]
            obs |= self.local_obstacle_map[a]
        return known & ~obs

    def fleet_area_domain(self):
        """Cells NOT believed to be obstacles — the plausible mission area
        (includes never-seen cells; a clearance pass must cover them)."""
        obs = self.local_obstacle_map[0].copy()
        for a in range(1, self.num_agents):
            obs |= self.local_obstacle_map[a]
        return ~obs

    # ------------------------------------------------------------------
    # Ground-truth metrics helpers (never fed to policies)
    # ------------------------------------------------------------------

    def true_residual(self):
        """Unlocated mines (never confirmed by any agent)."""
        return int(self.num_mines - np.count_nonzero(self.located_mask))

    def true_recall(self):
        return float(np.count_nonzero(self.located_mask)) / \
            float(self.num_mines)

    def true_neutralized(self):
        return int(np.count_nonzero(self.neutralized_mask))

    def reset(self):
        super().reset()
        self._build_belief_arrays()
        self.step_index = 0
        self._place_mines()

    # Construction order fix: GridEnv.__init__ ends with _place_agents();
    # we re-run the belief/mine setup here right after super().__init__.
    # (reset() and __init__ share the same path)


def build_mine_env(**kwargs):
    env = MineGridEnv(**kwargs)
    env._build_belief_arrays()
    env.step_index = 0
    env._place_mines()
    return env
