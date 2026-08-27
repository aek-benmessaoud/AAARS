"""
boustro_lanes.py — Allocation policies for detection missions (V5).

AL-Boustro   : static lane partition + serpentine sweep (demining practice),
               lane_passes legs, obstacle-aware greedy transit, explore fallback.

LEAK-FREE: only local_seen/local_obs/local_scan_count/local_confirm_bits[aid].
"""

import numpy as np

from src.allocation.common import explore_action


# ======================================================================
# AL-Boustro
# ======================================================================

class BoustroLanesPolicy:
    """Agent i sweeps vertical lane [i*W, (i+1)*W) in horizontal serpentine
    legs centered on row bands of height (2*fov+1)."""

    def __init__(self, seed=None, fov_radius=5, grid_size=100, num_agents=6,
                 agent_id=0, lane_passes=2):
        self.rng = np.random.default_rng(seed)
        self.fov = fov_radius
        self.gs = grid_size
        self.lane_passes = lane_passes
        W = int(np.ceil(grid_size / num_agents))
        self.c0 = agent_id * W
        self.c1 = min(self.c0 + W, grid_size)
        band_h = 2 * fov_radius + 1
        # Tile rows into disjoint bands of width band_h; each band center
        # covers its full band (width <= 2*fov+1). Guarantees edge rows
        # (0 and gs-1) belong to some band -> no unswept border strip.
        centers = []
        start = 0
        while start < grid_size:
            end = min(start + band_h - 1, grid_size - 1)
            centers.append((start + end) // 2)
            start += band_h
        self.waypoints = []          # (row, col, required_scan_count)
        for p in range(lane_passes):
            req = p + 1
            rows = centers[::-1] if p % 2 else centers
            for bi, r in enumerate(rows):
                # enough sweep points per band to tile the lane WIDTH
                n_pts = max(1, int(np.ceil((self.c1 - self.c0) / band_h)))
                if n_pts == 1:
                    cols_pt = [(self.c0 + self.c1) // 2]
                else:
                    lo, hi = self.c0 + fov_radius, self.c1 - 1 - fov_radius
                    step = (hi - lo) / (n_pts - 1)
                    cols_pt = [int(round(lo + k * step))
                               for k in range(n_pts)]
                    cols_pt = [int(np.clip(cc, self.c0, self.c1 - 1))
                               for cc in cols_pt]
                if (r // band_h) % 2 == 1:
                    cols_pt = cols_pt[::-1]
                for cc in cols_pt:
                    self.waypoints.append((int(r), cc, req))
        self.wp_i = 0
        self.stalls = 0
        self._path = None            # cached BFS cell path to active wp
        self._path_wp_i = -1
        self._path_age = 0
        self._wp_fails = 0
        self._adj_key = -1
        self._adj_rc = None

    def _bfs_path(self, env, agent_id, tr, tc):
        """BFS shortest path over BELIEVED-free cells, restricted to this
        agent's lane strip (+margin) for speed; falls back to the full
        grid when the strip is disconnected by obstacle clusters.
        None if unreachable."""
        path = self._bfs_path_limited(env, agent_id, tr, tc,
                                      max(0, self.c0 - 3),
                                      min(self.gs, self.c1 + 3))
        if path is None:
            path = self._bfs_path_limited(env, agent_id, tr, tc, 0, self.gs)
        return path

    def _bfs_path_limited(self, env, agent_id, tr, tc, lo, hi):
        from collections import deque
        gs = self.gs
        obs = env.get_obstacle_knowledge(agent_id)
        sr, sc_ = env.agent_positions[agent_id]
        prev = {}
        seen = {(sr, sc_)}
        q = deque([(sr, sc_)])
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) == (tr, tc):
                found = True
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and lo <= nc < hi and \
                        (nr, nc) not in seen and not obs[nr, nc]:
                    seen.add((nr, nc))
                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        if not found:
            return None
        path = []
        node = (tr, tc)
        while node != (sr, sc_):
            path.append(node)
            node = prev.get(node)
            if node is None:
                return None
        path.reverse()
        return [(sr, sc_)] + path

    def _nearest_free(self, env, agent_id, tr, tc):
        """If the waypoint cell itself is (believed) obstacle, pick the
        nearest free cell within FOV+reach so its scan box still covers
        the intended band segment."""
        gs = self.gs
        obs = env.get_obstacle_knowledge(agent_id)
        if 0 <= tr < gs and 0 <= tc < gs and not obs[tr, tc]:
            return tr, tc
        lo_c, hi_c = max(0, self.c0 - 2), min(gs, self.c1 + 2)
        for d in range(1, self.fov + 3):
            best, best_key = None, None
            for nr in range(max(0, tr - d), min(gs, tr + d + 1)):
                for nc in range(max(lo_c, tc - d), min(hi_c, tc + d + 1)):
                    if obs[nr, nc]:
                        continue
                    key = (max(abs(nr - tr), abs(nc - tc)),
                           abs(nr - tr), abs(nc - tc))
                    if best_key is None or key < best_key:
                        best_key, best = key, (nr, nc)
            if best:
                return best
        return tr, tc

    def _coverage_recovery_target(self, env, agent_id, tr, tc, req):
        """Free cell whose FOV box covers the most still-deficit band
        cells (used only when the nominal waypoint is unusable)."""
        gs = self.gs
        sc = env.local_scan_count[agent_id]
        obs = env.get_obstacle_knowledge(agent_id)
        r0, r1 = max(0, tr - self.fov), min(gs, tr + self.fov + 1)
        c0, c1 = self.c0, self.c1
        deficit = sc[r0:r1, c0:c1] < req
        if not deficit.any():
            return None
        best, best_key = None, None
        for nr in range(max(0, r0 - self.fov), min(gs, r1 + self.fov)):
            for nc in range(c0, c1):
                if obs[nr, nc]:
                    continue
                rr0 = max(r0, nr - self.fov); rr1 = min(r1, nr + self.fov + 1)
                cc0 = max(c0, nc - self.fov); cc1 = min(c1, nc + self.fov + 1)
                cov = int(deficit[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0].sum())
                key = (-cov, abs(nr - tr) + abs(nc - tc))
                if best_key is None or key < best_key:
                    best_key, best = key, (nr, nc)
        return best

    def _transit_step(self, env, agent_id, tr, tc):
        """Next action toward (tr, tc): cached-BFS path following with
        revalidation; greedy fallback when no path exists."""
        r, c = env.agent_positions[agent_id]
        if self._path_wp_i != self.wp_i or self._path_age > 80:
            self._path = self._bfs_path(env, agent_id, tr, tc)
            self._path_wp_i = self.wp_i
            self._path_age = 0
            self._pi = 0
        if self._path:
            # resync: locate current position along the cached path
            try:
                self._pi = self._path.index((r, c), max(0, self._pi - 2))
            except ValueError:
                self._path = self._bfs_path(env, agent_id, tr, tc)
                self._path_age = 0
                self._pi = 0
        if self._path and self._pi + 1 < len(self._path):
            nr_, nc_ = self._path[self._pi + 1]
            obs = env.get_obstacle_knowledge(agent_id)
            if obs[nr_, nc_]:
                self._path = self._bfs_path(env, agent_id, tr, tc)
                self._path_age = 0
                self._pi = 0
            if self._path and self._pi + 1 < len(self._path):
                nr_, nc_ = self._path[self._pi + 1]
                dr_, dc_ = nr_ - r, nc_ - c
                act = {( -1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get(
                    (dr_, dc_))
                if act is not None:
                    self._path_age += 1
                    self._pi += 1
                    return act
        self._path = None
        return None

    def _band_done(self, env, agent_id, wp_r, req):
        sc = env.local_scan_count[agent_id]
        r0 = max(0, wp_r - self.fov)
        r1 = min(self.gs, wp_r + self.fov + 1)
        if r1 <= r0:
            return True
        return bool((sc[r0:r1, self.c0:self.c1] >= req).all())

    def select_action(self, env, agent_id):
        env.update_local_memory(agent_id, self.fov)
        r, c = env.agent_positions[agent_id]

        while self.wp_i < len(self.waypoints):
            tr, tc, req = self.waypoints[self.wp_i]
            if (abs(r - tr) + abs(c - tc)) == 0 or \
                    self._band_done(env, agent_id, tr, req):
                self.wp_i += 1
                continue
            break

        if self.wp_i >= len(self.waypoints):
            # All passes done. Deliberately idle: further evidence must come
            # from UNIFORM full-lane passes (scheduled via lane_passes), not
            # from targeted singleton chasing — chasing promotes seen mines'
            # occasion counts independently of their detectability and
            # biases the Chao estimator toward premature certification.
            return 4, "Boustro", "idle", None

        tr, tc, req = self.waypoints[self.wp_i][:3]
        if self._adj_key != self.wp_i:
            self._adj_key = self.wp_i
            obs0 = env.get_obstacle_knowledge(agent_id)
            if not (0 <= tr < self.gs and 0 <= tc < self.gs) or \
                    obs0[tr, tc]:
                adj = self._coverage_recovery_target(
                    env, agent_id, tr, tc, req)
                self._adj_rc = adj if adj else \
                    self._nearest_free(env, agent_id, tr, tc)
            else:
                self._adj_rc = (tr, tc)
        tr, tc = self._adj_rc
        act = self._transit_step(env, agent_id, tr, tc)
        if act is not None:
            self.stalls = 0
            return int(act), "Boustro", "sweep", None
        # unreachable within believed map: after repeated failures skip wp
        self._wp_fails += 1
        if self._wp_fails > 5:
            self.wp_i += 1
            self._wp_fails = 0
            self._path = None
        obs = env.get_obstacle_knowledge(agent_id)
        gs = self.gs
        best, best_d, cands = None, None, []
        for a, dr, dc in ((0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < gs and 0 <= nc < gs) or obs[nr, nc]:
                continue
            d = abs(nr - tr) + abs(nc - tc)
            if best_d is None or d < best_d:
                best_d, best, cands = d, a, [a]
            elif d == best_d:
                cands.append(a)
        if best is None:
            act, mode = explore_action(env, agent_id, self.rng)
            return act, "Boustro", mode, None
        # stall guard: oscillation around obstacle -> explore fallback
        if len(cands) == 1 and self.stalls > 8:
            self.stalls = 0
            act, mode = explore_action(env, agent_id, self.rng)
            return act, "Boustro", mode, None
        self.stalls = self.stalls + 1 if len(cands) == 1 else 0
        return int(self.rng.choice(cands)), "Boustro", "sweep", None
