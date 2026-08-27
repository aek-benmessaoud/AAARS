"""
runner.py — Unified episode runner for Project11-AAARS.

One episode per (allocation, seed) pair. Evaluates ALL methods on the same
detection trace. AAARS and discrete_aaars run their controllers in-loop;
baselines are evaluated post-hoc from the trace.

Methods:
  chao1_ci     : Chao1-CI stopping (baseline)
  chao92_ci    : Chao92-CI stopping (baseline)
  aaars        : AAARS continuous risk-modulated estimator (main contribution)
  discrete_aaars : Discrete Chao1/Chao92 switching (ablation)
  oracle_95    : Post-hoc oracle (ground truth)
  fixed_2      : Fixed-2 sweep (post-hoc from scan counts)
  diminishing  : Diminishing returns (post-hoc from trace)
"""

import os
import sys
import time
import json

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.environment.mine_env import build_mine_env
from src.estimators.chao1 import residual_estimate
from src.estimators.chao92 import chao92_from_freq, full_frequencies
from src.allocation.boustro_lanes import BoustroLanesPolicy
from src.allocation.mine_richness import MineRichnessPolicy
from src.aaars.controller import AAARSController
from src.aaars.discrete_selector import DiscreteSelectorController
from src.stopping.diminishing import DiminishingStop
from src.utils.seed_manager import env_seed_for_run, policy_seed_for


# ---- Default configuration ----
DEFAULT_CFG = {
    "grid_size": 100,
    "num_agents": 6,
    "fov_radius": 5,
    "obstacle_ratio": 0.05,
    "num_mines": 60,
    "p_bar": 0.7,
    "strata": (0.9, 0.6, 0.3),
    "detectability": "homogeneous",
    "info_model": "comm_limited",
    "comm_range": 5.0,
    "max_steps": 6000,
    "trace_stride": 10,
}

AAARS_STOP_RULES = [
    "chao1_ci", "chao92_ci", "aaars", "discrete_aaars",
    "oracle_95", "fixed_2", "diminishing",
]


def build_allocation(name, seed_i, agent_id, cfg):
    """Build an allocation policy instance."""
    gs = cfg["grid_size"]
    na = cfg["num_agents"]
    fov = cfg["fov_radius"]
    if name == "boustro":
        return BoustroLanesPolicy(seed=seed_i, fov_radius=fov,
                                  grid_size=gs, num_agents=na,
                                  agent_id=agent_id, lane_passes=3)
    if name == "minerich":
        return MineRichnessPolicy(seed=seed_i, fov_radius=fov)
    raise ValueError(f"Unknown allocation: {name}")


def run_episode(alloc_name, run_index, env_seed, cfg=None,
                collect_trace=True):
    """Run one episode and evaluate all stopping methods.
    
    Args:
        alloc_name: 'boustro' or 'minerich'
        run_index: episode index (for seed derivation)
        env_seed: explicit environment seed
        cfg: dict of configuration overrides
        collect_trace: whether to record per-step trace
    
    Returns:
        dict with per-method results and metadata
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    
    # Build environment
    env = build_mine_env(
        grid_size=cfg["grid_size"],
        num_agents=cfg["num_agents"],
        obstacle_ratio=cfg["obstacle_ratio"],
        seed=env_seed,
        info_model=cfg["info_model"],
        comm_range=cfg["comm_range"],
        num_mines=cfg["num_mines"],
        detectability=cfg["detectability"],
        p_bar=cfg["p_bar"],
        strata=cfg["strata"],
    )
    
    # Build allocation policies
    policies = [
        build_allocation(alloc_name,
                         policy_seed_for(env_seed, i), i, cfg)
        for i in range(cfg["num_agents"])
    ]
    
    # Build AAARS controllers (params from cfg["aaars"] if provided)
    aaars_kw = cfg.get("aaars", {})
    aaars = AAARSController(
        window_size=aaars_kw.get("window_size", 8),
        grid_size=cfg["grid_size"],
        num_bins=aaars_kw.get("num_bins", 5),
        w_temporal=aaars_kw.get("w_temporal", 0.0),
        w_coverage=aaars_kw.get("w_coverage", 0.55),
        w_frequency=aaars_kw.get("w_frequency", 0.45),
        ema_alpha=aaars_kw.get("ema_alpha", 0.1),
        base_alpha=aaars_kw.get("base_alpha", 0.05),
        risk_lambda=aaars_kw.get("risk_lambda", 1.0),
        blend_threshold=aaars_kw.get("blend_threshold", 0.30),
    )
    discrete = DiscreteSelectorController(
        window_size=aaars_kw.get("window_size", 8),
        grid_size=cfg["grid_size"],
    )
    diminishing = DiminishingStop(tau=150, n_min=5)
    
    # State tracking
    stop_t = {r: None for r in AAARS_STOP_RULES}
    true_found_series = []
    trace = []
    last_chao1_est = None
    last_chao92_est = None
    
    t0 = time.perf_counter()
    
    for t in range(1, cfg["max_steps"] + 1):
        # Fleet fusion before perception
        if cfg["info_model"] == "comm_limited":
            env.check_and_merge()
        
        # Agent actions + environment step
        actions = [policies[i].select_action(env, i)[0]
                   for i in range(cfg["num_agents"])]
        env.step(actions)
        
        # Get fleet-level observations (belief only, leak-free)
        bits = env.fleet_block_bits()
        
        # Ground truth (evaluation only)
        true_found = int(env.n_detections)
        true_found_series.append((t, true_found))
        
        # --- Trace-based baselines ---
        
        # Chao1-CI
        est1 = residual_estimate(bits)
        last_chao1_est = est1
        f2_floor = max(5.0, np.ceil(0.08 * est1["n_det"]))
        armed1 = est1["n_det"] >= 5 and est1["f2"] >= f2_floor
        if stop_t["chao1_ci"] is None and armed1:
            if est1["ci_upper"] <= 0.05 * est1["K_hat"]:
                stop_t["chao1_ci"] = t
        
        # Chao92-CI
        fk = full_frequencies(bits)
        est92 = chao92_from_freq(fk)
        last_chao92_est = est92
        f2_floor92 = max(5.0, np.ceil(0.08 * est92["n_det"]))
        armed92 = est92["n_det"] >= 5 and float(fk[2]) >= f2_floor92
        if stop_t["chao92_ci"] is None and armed92:
            if est92["U92"] >= 0 and est92["ci92_upper"] <= 0.05 * est92["K_hat92"]:
                stop_t["chao92_ci"] = t
        
        # Fixed-2
        if stop_t["fixed_2"] is None:
            sc = env.fleet_scan_max()
            dom = env.fleet_area_domain()
            if dom.any() and sc[dom].min() >= 2:
                stop_t["fixed_2"] = t
        
        # Diminishing
        if stop_t["diminishing"] is None:
            if diminishing.update(t, est1["n_det"]):
                stop_t["diminishing"] = t
        
        # --- AAARS (continuous blend) ---
        # Leak-free fleet coverage: fraction of traversable cells scanned >= 1
        sc_fleet = env.fleet_scan_max()
        traversable = env.traversable  # total traversable count (leak-free)
        cover = float(np.count_nonzero((sc_fleet > 0) & ~env.obstacle_map)) / traversable if traversable else 0.0
        aaars_result = aaars.step(bits, t, fleet_coverage=cover)
        if stop_t["aaars"] is None and aaars_result["stop"]:
            stop_t["aaars"] = t
        
        # --- Discrete selector (ablation) ---
        discrete_result = discrete.step(bits, t)
        if stop_t["discrete_aaars"] is None and discrete_result["stop"]:
            stop_t["discrete_aaars"] = t
        
        # --- Trace collection ---
        if collect_trace and (t % cfg["trace_stride"] == 0 or t == cfg["max_steps"]):
            diag = {}
            diag.update(aaars_result.get("diagnostics", {}))
            entry = {
                "t": t,
                "true_found": true_found,
                "belief_ndet": est1["n_det"],
                "U1": round(est1["U_hat"], 3),
                "U92": round(est92.get("U92", 0), 3),
                "risk_score": round(aaars_result.get("risk_score", 0), 4),
                "aaars_alpha_adj": round(aaars_result.get("alpha_adj", 0.05), 6),
                "aaars_stop": aaars_result["stop"],
                "discrete_state": discrete_result.get("state", ""),
                "f1": est1["f1"], "f2": est1["f2"],
                "fk": [int(x) for x in fk[:10]],
            }
            entry.update(diag)
            trace.append(entry)
        
        # Early exit: all rules fired
        if t >= 2000:
            pending = any(stop_t[k] is None for k in
                         ("chao1_ci", "chao92_ci", "diminishing", "fixed_2"))
            if not pending and stop_t["aaars"] is not None \
                    and stop_t["discrete_aaars"] is not None:
                break
    
    wall = time.perf_counter() - t0
    T_total = t
    
    # Post-hoc oracle
    oracle_t = None
    for tt, f in true_found_series:
        if f >= 0.95 * cfg["num_mines"]:
            oracle_t = tt
            break
    stop_t["oracle_95"] = oracle_t or T_total
    
    # Outcome helper
    def outcome(stop_at):
        s = min(stop_at if stop_at is not None else T_total, T_total)
        f = next((f_ for tt, f_ in reversed(true_found_series) if tt <= s), 0)
        return f, 100.0 * f / cfg["num_mines"], cfg["num_mines"] - f
    
    # Build result dict
    res = {
        "alloc": alloc_name,
        "run": run_index,
        "env_seed": env_seed,
        "detectability": cfg["detectability"],
        "num_mines": cfg["num_mines"],
        "grid_size": cfg["grid_size"],
        "num_agents": cfg["num_agents"],
        "T_total": T_total,
        "wall_time_s": round(wall, 2),
    }
    
    for rule in AAARS_STOP_RULES:
        s = stop_t[rule]
        found, rec, miss = outcome(s)
        res[f"{rule}__t"] = s
        res[f"{rule}__found"] = found
        res[f"{rule}__recall"] = round(rec, 2)
        res[f"{rule}__misses"] = miss
    
    # Add AAARS-specific stats
    aaars_stats = aaars.get_final_stats()
    discrete_stats = discrete.get_final_stats()
    res["aaars__switches"] = aaars_stats["num_switches"]
    res["aaars__final_risk"] = aaars_stats["final_risk_score"]
    res["discrete__switches"] = discrete_stats["num_switches"]
    
    if collect_trace:
        res["_trace"] = json.dumps(trace)
    
    return res


if __name__ == "__main__":
    # Quick smoke test: 1 episode, boustro, seed 0
    result = run_episode("boustro", 0, env_seed=0,
                         cfg={**DEFAULT_CFG, "max_steps": 500})
    print(f"Alloc: {result['alloc']}, Run: {result['run']}")
    for rule in AAARS_STOP_RULES:
        t = result[f"{rule}__t"]
        rec = result[f"{rule}__recall"]
        print(f"  {rule:20s}: t={str(t):>6s}  recall={rec:.1f}%")
    print(f"  aaars switches: {result['aaars__switches']}")
    print(f"  wall time: {result['wall_time_s']:.1f}s")
