"""analysis_utils.py — Functions needed by allocation policies, extracted from old compute_entropy.py."""

import numpy as np


def frontier_mask(known, obs, unknown):
    """Known free cells adjacent (4-neighborhood) to >= 1 unknown cell."""
    unknown = np.asarray(unknown, dtype=bool)
    gs = known.shape[0]
    adj = np.zeros_like(unknown)
    adj[:-1, :] |= unknown[1:, :]
    adj[1:, :] |= unknown[:-1, :]
    adj[:, :-1] |= unknown[:, 1:]
    adj[:, 1:] |= unknown[:, :-1]
    return (known & ~obs & adj).astype(np.float64)


def select_target(utility, D, curdir, candidate_mask, rng, tie_eps=1e-3):
    """Pick the argmax utility cell among candidates with D>0."""
    reachable = (np.asarray(candidate_mask, dtype=bool)) & (D > 0)
    if not reachable.any():
        return None
    vals = utility[reachable].copy()
    if tie_eps > 0:
        vals = vals + rng.uniform(-tie_eps, tie_eps, size=vals.shape)
    j = int(np.argmax(vals))
    idx = np.argwhere(reachable)[j]
    tr, tc = int(idx[0]), int(idx[1])
    return int(curdir[tr, tc])
