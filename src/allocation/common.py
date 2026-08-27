"""
policies/_common.py — Shared low-level helpers used by all policies.

These helpers are LEAK-FREE: they only use env.local_* / knowledge
interface. Unknown cells (not marked as obstacles) are treated as free.
"""

import numpy as np


def valid_neighbors_local(env, agent_id, r, c, include_stay=False):
    """
    Valid moves using the agent's LOCAL obstacle memory.
    Unknown cells are considered free unless marked obstacle.
    Returns list of (action, nr, nc).
    """
    obs = env.get_obstacle_knowledge(agent_id)
    gs = env.grid_size
    moves = [
        (0, -1, 0),   # up
        (1, 1, 0),    # down
        (2, 0, -1),   # left
        (3, 0, 1),    # right
    ]
    if include_stay:
        moves.append((4, 0, 0))

    out = []
    for action, dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < gs and 0 <= nc < gs and not obs[nr, nc]:
            out.append((action, nr, nc))
    return out


def explore_action(env, agent_id, rng):
    """
    BFS towards the nearest locally-unvisited traversable cell.
    Fallback: random valid move (unknown cells are walkable).
    Returns (action, mode).

    Exhaustion cache: if a previous BFS exhausted (no reachable target) and
    neither the unvisited mask nor the obstacle knowledge changed, the
    component can only have shrunk, so the result is still "no reachable
    target" -> reuse the cached fallback (behavior-identical, rng stream
    unchanged because the fallback consumes rng exactly as the BFS path does).
    """
    r, c = env.agent_positions[agent_id]
    unvisited = env.get_local_unvisited_mask(agent_id)
    obs = env.get_obstacle_knowledge(agent_id)

    if env.explore_exhausted[agent_id]:
        cached_u = env.explore_exhausted_unvisited[agent_id]
        cached_o = env.explore_exhausted_obs[agent_id]
        if (cached_u is not None and cached_o is not None
                and np.array_equal(cached_u, unvisited)
                and np.array_equal(cached_o, obs)):
            neigh = valid_neighbors_local(env, agent_id, r, c, include_stay=True)
            if neigh:
                return int(rng.choice([a for a, _, _ in neigh])), "explore_random"
            return 4, "explore_stay"

    if unvisited[r, c]:
        return 4, "explore"
    gs = env.grid_size

    visited = np.zeros((gs, gs), dtype=bool)
    curdir = np.full((gs, gs), 4, dtype=np.int8)
    visited[r, c] = True
    cur = np.zeros((gs, gs), dtype=bool)
    cur[r, c] = True
    depth = 0

    while True:
        hit = cur & unvisited
        if hit.any():
            tr, tc = np.argwhere(hit)[0]
            return int(curdir[tr, tc]), "explore"

        depth += 1
        nxt = np.zeros((gs, gs), dtype=bool)
        nxt[:-1, :] |= cur[1:, :]   # up
        nxt[1:, :] |= cur[:-1, :]   # down
        nxt[:, :-1] |= cur[:, 1:]   # left
        nxt[:, 1:] |= cur[:, :-1]   # right
        nxt &= ~obs & ~visited
        if not nxt.any():
            break

        # Per-direction masks of newly reached cells (child -> parent).
        mu = np.zeros((gs, gs), dtype=bool); mu[:-1, :] = cur[1:, :]; mu &= nxt
        md = np.zeros((gs, gs), dtype=bool); md[1:, :] = cur[:-1, :]; md &= nxt
        ml = np.zeros((gs, gs), dtype=bool); ml[:, :-1] = cur[:, 1:]; ml &= nxt
        mr = np.zeros((gs, gs), dtype=bool); mr[:, 1:] = cur[:, :-1]; mr &= nxt

        if depth == 1:
            cd = np.full((gs, gs), 4, dtype=np.int8)
            cd[mu] = 0
            cd[md] = np.where(cd[md] == 4, 1, cd[md])
            cd[ml] = np.where(cd[ml] == 4, 2, cd[ml])
            cd[mr] = np.where(cd[mr] == 4, 3, cd[mr])
            curdir = cd
        else:
            nd = np.full((gs, gs), 4, dtype=np.int8)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:-1, :] = np.where(mu[:-1, :], curdir[1:, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[1:, :] = np.where(md[1:, :], curdir[:-1, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, :-1] = np.where(ml[:, :-1], curdir[:, 1:], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, 1:] = np.where(mr[:, 1:], curdir[:, :-1], 4)
            nd = np.minimum(nd, cand)
            curdir = nd

        visited |= nxt
        cur = nxt

    env.explore_exhausted[agent_id] = True
    env.explore_exhausted_unvisited[agent_id] = unvisited.copy()
    env.explore_exhausted_obs[agent_id] = obs.copy()

    neigh = valid_neighbors_local(env, agent_id, r, c, include_stay=True)
    if neigh:
        return int(rng.choice([a for a, _, _ in neigh])), "explore_random"
    return 4, "explore_stay"


def exploit_action(env, agent_id, rng):
    """
    Move to the KNOWN neighbor with the lowest OWN visit count.
    Unknown cells are skipped for exploitation. Fallback: random move.
    Returns (action, mode).
    """
    r, c = env.agent_positions[agent_id]
    info = env.get_local_info(agent_id)
    visit = info["visit"].copy()
    obs = info["obs"]
    known = info["known"]

    visit[obs] = 999999
    best_actions = []
    best_score = -1.0

    for a, nr, nc in valid_neighbors_local(env, agent_id, r, c, include_stay=True):
        if not known[nr, nc]:
            continue
        score = 1.0 / (1.0 + visit[nr, nc])
        if a != 4:
            score += 0.1
        if score > best_score:
            best_score = score
            best_actions = [a]
        elif abs(score - best_score) < 1e-12:
            best_actions.append(a)

    if best_actions:
        return int(rng.choice(best_actions)), "exploit"

    neigh = valid_neighbors_local(env, agent_id, r, c, include_stay=True)
    if neigh:
        return int(rng.choice([a for a, _, _ in neigh])), "exploit"
    return 4, "exploit"


def box_sum(mask, w):
    """Sum of `mask` over every square window of radius `w` (size 2w+1),
    fully vectorized via two sliding cumsum passes. Returns a float grid."""
    f = np.asarray(mask, dtype=np.float64)

    def axis_sum(a, axis):
        n = a.shape[axis]
        A = np.pad(a, [(w, w) if ax == axis else (0, 0) for ax in range(2)],
                   mode="constant")
        zshape = list(A.shape)
        zshape[axis] = 1
        C = np.concatenate([np.zeros(zshape), np.cumsum(A, axis)], axis=axis)
        lo = np.arange(n)
        hi = lo + 2 * w + 1
        return np.take(C, hi, axis=axis) - np.take(C, lo, axis=axis)

    return axis_sum(axis_sum(f, 1), 0)


def bounded_bfs(env, agent_id, max_depth=8):
    """Bounded BFS from the agent in the V4 movement frame.

    Paths may pass through UNKNOWN cells and KNOWN-UNVISITED free cells, but
    NOT through already-visited cells. Every step into a target therefore
    enters fresh territory. Unknown cells are treated as free (codebase
    convention). The expansion is capped at `max_depth` layers, so the cost is
    bounded and independent of target distance.

    Returns (D, curdir):
      D[i, j]      = BFS layer (1..max_depth) or -1 if unreachable / beyond
                     the horizon
      curdir[i, j] = first-step action toward (i, j) (4 = stay/unset)
    """
    r, c = env.agent_positions[agent_id]
    known = env.get_known_mask(agent_id)
    obs = env.get_obstacle_knowledge(agent_id)
    visit = env.get_visit_knowledge(agent_id)
    gs = env.grid_size

    known_visited = known & ~obs & (visit > 0)
    free_path = ~obs & ~known_visited   # unknown + known-unvisited free cells

    visited = np.zeros((gs, gs), dtype=bool)
    visited[r, c] = True
    curdir = np.full((gs, gs), 4, dtype=np.int8)
    D = np.full((gs, gs), -1, dtype=np.int16)
    D[r, c] = 0
    cur = np.zeros((gs, gs), dtype=bool)
    cur[r, c] = True
    depth = 0

    while True:
        depth += 1
        if depth > max_depth:
            break
        nxt = np.zeros((gs, gs), dtype=bool)
        nxt[:-1, :] |= cur[1:, :]
        nxt[1:, :] |= cur[:-1, :]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[:, 1:] |= cur[:, :-1]
        nxt &= free_path & ~visited
        if not nxt.any():
            break

        mu = np.zeros((gs, gs), dtype=bool); mu[:-1, :] = cur[1:, :]; mu &= nxt
        md = np.zeros((gs, gs), dtype=bool); md[1:, :] = cur[:-1, :]; md &= nxt
        ml = np.zeros((gs, gs), dtype=bool); ml[:, :-1] = cur[:, 1:]; ml &= nxt
        mr = np.zeros((gs, gs), dtype=bool); mr[:, 1:] = cur[:, :-1]; mr &= nxt

        if depth == 1:
            cd = np.full((gs, gs), 4, dtype=np.int8)
            cd[mu] = 0
            cd[md] = np.where(cd[md] == 4, 1, cd[md])
            cd[ml] = np.where(cd[ml] == 4, 2, cd[ml])
            cd[mr] = np.where(cd[mr] == 4, 3, cd[mr])
            curdir = cd
        else:
            nd = curdir.copy()
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:-1, :] = np.where(mu[:-1, :], curdir[1:, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[1:, :] = np.where(md[1:, :], curdir[:-1, :], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, :-1] = np.where(ml[:, :-1], curdir[:, 1:], 4)
            nd = np.minimum(nd, cand)
            cand = np.full((gs, gs), 4, dtype=np.int8)
            cand[:, 1:] = np.where(mr[:, 1:], curdir[:, :-1], 4)
            nd = np.minimum(nd, cand)
            curdir = nd

        D[nxt] = depth
        visited |= nxt
        cur = nxt

    return D, curdir
