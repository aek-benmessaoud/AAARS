"""
utils/seed_manager.py — Deterministic, paired seed handling.

RULES (locked decisions):
  - Env seed for run index r:  env_seed = BASE_SEED + r * SEED_STRIDE.
    All methods share the SAME env seed per run index (paired at map level).
  - Per-agent policy seed: derived deterministically from env_seed + agent_id,
    so each agent has its own independent RNG stream.
  - Policy seeds are NOT identical across methods: internal policy randomness
    (tie-breaking etc.) is free to differ, as agreed (limitation D1).
"""

BASE_SEED = 0
SEED_STRIDE = 1000


def env_seed_for_run(run_index):
    return BASE_SEED + run_index * SEED_STRIDE


def policy_seed_for(env_seed, agent_id):
    """Deterministic per-agent policy RNG seed."""
    return env_seed * 1000 + agent_id + 1
