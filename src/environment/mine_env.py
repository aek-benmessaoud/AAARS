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
                 conf_blocks=2, occ_cooldown=50,
                 band_pd=(0.5, 0.7, 0.9), comm_delay=0):
        self.num_mines = num_mines
        self.detectability = detectability
        self.p_bar = p_bar
        self.strata = tuple(strata)
        self.band_pd = tuple(band_pd)
        self.persistent = persistent
        # Heterogeneous communication latency: per-agent staleness (in decision
        # steps) applied to rendezvous fusion. comm_delay=0 -> no latency
        # (exact symmetric merge, unchanged behaviour). Per-agent lags are drawn
        # deterministically from a dedicated RNG so the schedule is exactly
        # reproducible per agent for a given seed (D4 / advisor 2d).
        self.comm_delay = int(comm_delay)
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
        # Heterogeneous comm latency (leak-free: only observables are lagged).
        if self.comm_delay > 0:
            self._delay_rng = np.random.default_rng(self.seed + 5000)
            # Per-agent lag in [1, comm_delay], deterministic per seed.
            self._agent_delay = [
                int(self._delay_rng.integers(1, self.comm_delay + 1))
                for _ in range(self.num_agents)
            ]
            maxd = max(self._agent_delay)
            # Circular history of belief snapshots (list of list-of-arrays);
            # _snap[k] holds the state from (maxd - k) steps ago.
            self._snap = []
            self._record_snapshot()
        else:
            self._agent_delay = [0] * self.num_agents
            self._snap = None

    def _record_snapshot(self):
        """Push a deep copy of every agent's observable belief arrays onto the
        circular snapshot history (used only when comm_delay > 0)."""
        gs = self.grid_size
        entry = []
        for a in range(self.num_agents):
            entry.append({
                "visit": self.local_visit_count[a].copy(),
                "seen": self.local_seen_mask[a].copy(),
                "obs": self.local_obstacle_map[a].copy(),
                "scan": self.local_scan_count[a].copy(),
                "found": self.local_found[a].copy(),
                "cfb": self.local_confirm_bits[a].copy(),
            })
        self._snap.append(entry)
        maxd = max(self._agent_delay)
        if len(self._snap) > maxd + 1:
            self._snap.pop(0)

    def _snapshot_at(self, agent, lag):
        """Beliefs of `agent` from `lag` decision-steps ago (0 = now)."""
        want = len(self._snap) - 1 - lag
        entry = self._snap[max(0, want)]
        s = entry[agent]
        return s["visit"], s["seen"], s["obs"], s["scan"], s["found"], s["cfb"]

    def _place_mines(self):
        gs = self.grid_size
        free = np.argwhere(~self.obstacle_map)
        agent_cells = {tuple(p) for p in self.agent_positions}
        cand = np.array([c for c in free if tuple(c) not in agent_cells])

        nb = len(self.band_pd)
        if self.detectability in ("bands_hetero", "bands_rich"):
            # 3 vertical strips by column index. Each free cell belongs to the
            # strip that contains its column.
            bounds = [i * gs // nb for i in range(nb + 1)]
            bands = np.zeros(len(cand), dtype=int)
            for b in range(nb):
                m = (cand[:, 1] >= bounds[b]) & (cand[:, 1] < bounds[b + 1])
                bands[m] = b
            band_pd = np.asarray(self.band_pd, dtype=np.float64)

            if self.detectability == "bands_hetero":
                # H1: mines uniform over free cells; p_d varies only by strip.
                idx = self.rng.choice(len(cand), size=self.num_mines,
                                      replace=False)
            else:
                # H2: K stays = num_mines (LOCK), only the *distribution* moves.
                # Concentrate the mines in the STRIP WITH LOWEST p_d (the hardest
                # zone to detect), so richness is richest exactly where detection
                # is weakest. Total mines unchanged -> comparability with H0/H1
                # and tab:confirm is preserved (redistribution, not inflation).
                low = int(np.argmin(band_pd))
                weights = np.ones(nb, dtype=np.float64) * 0.2
                weights[low] = 1.0 - 0.2 * (nb - 1)   # leftover mass on hard zone
                counts = np.bincount(bands, minlength=nb).astype(np.float64)
                per_cell = weights[bands] / np.maximum(counts[bands], 1.0)
                per_cell /= per_cell.sum()
                idx = self.rng.choice(len(cand), size=self.num_mines,
                                      replace=False, p=per_cell)
            cells = cand[idx]
            cell_bands = bands[idx]
            self.mine_mask = np.zeros((gs, gs), dtype=bool)
            self.mine_mask[cells[:, 0], cells[:, 1]] = True
            self.detect_prob = np.zeros((gs, gs), dtype=np.float64)
            self.detect_prob[cells[:, 0], cells[:, 1]] = band_pd[cell_bands]
        else:
            idx = self.rng.choice(len(cand), size=self.num_mines, replace=False)
            cells = cand[idx]
            self.mine_mask = np.zeros((gs, gs), dtype=bool)
            self.mine_mask[cells[:, 0], cells[:, 1]] = True
            # Per-mine detectability strata (M_h heterogeneity)
            self.detect_prob = np.zeros((gs, gs), dtype=np.float64)
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
        if self.comm_delay > 0:
            # Record this step's fused/observed beliefs for async fusion read
            # by the next step's check_and_merge.
            self._record_snapshot()
        return out

    # ------------------------------------------------------------------
    # Fusion overrides (union of confirmations + MAX scan counts)
    # ------------------------------------------------------------------

    def merge_maps(self, i, j):
        if self.comm_delay > 0:
            # Heterogeneous (async) fusion: each receiver keeps its OWN current
            # maps and unions in the partner's beliefs as of `d_receiver` steps
            # ago. Different receivers see different map ages -> the realistic
            # async-network stress the advisor asked for. Leak-free: only the
            # observable belief arrays are lagged; ground truth is untouched.
            vj, sj, oj, scj, foj, cfbj = self._snapshot_at(j, self._agent_delay[i])
            self._fuse_in(i, (vj, sj, oj, scj, foj, cfbj))
            vi, si, oi, sci, foi, cfbi = self._snapshot_at(i, self._agent_delay[j])
            self._fuse_in(j, (vi, si, oi, sci, foi, cfbi))
            self.fusion_events_count += 1
            return
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

    def _fuse_in(self, receiver, src):
        """Union/max the SOURCE's (possibly stale) beliefs into RECEIVER's
        own current beliefs. Symmetric-merge-free: receiver keeps its own."""
        r, a = receiver, src
        self.local_visit_count[r] = np.maximum(self.local_visit_count[r], a[0])
        self.local_seen_mask[r] |= a[1]
        self.local_obstacle_map[r] |= a[2]
        np.maximum(self.local_scan_count[r], a[3], out=self.local_scan_count[r])
        self.local_found[r] |= a[4]
        self.local_confirm_bits[r] |= a[5]

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
