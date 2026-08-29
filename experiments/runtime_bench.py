#!/usr/bin/env python
"""
runtime_bench.py — per-step incremental cost of each stopping step.

Times the per-step UPDATE of each method on the SAME detection trace of one
episode, isolating the controller/estimator cost per agent-step. The point is
to show AAARS's risk-modulated controller adds only O(1)-per-step overhead
relative to the plain Chao1 baseline (i.e. no meaningful computational tax),
NOT that it stops earlier.

Times per method (mirrors the exact calls in src/experiments/runner.py):
  chao1    : residual_estimate(bits) + arming check
  aaars    : AAARSController.step(PerceptStep)  (full controller)
  gap_sprt : GapSPRT.update(PerceptStep)
  rate_cs  : RateCS.update(PerceptStep)

Output: for each episode, accumulated total wall-seconds and step count per
method; mean per-step microseconds is derived downstream.
"""
import os
import sys
import time
import json
import argparse

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import DEFAULT_CFG, build_allocation
from src.environment.mine_env import build_mine_env
from src.utils.seed_manager import policy_seed_for
from src.estimators.chao1 import residual_estimate
from src.aaars.controller import AAARSController
from src.percepts import PerceptStep
from src.stopping.sequential import RateCS, GapSPRT

METHODS = ["chao1", "aaars", "gap_sprt"]


def bench_episode(alloc_name, seed_idx, env_seed, cfg_over=None):
    cfg = {**DEFAULT_CFG, **(cfg_over or {})}
    env = build_mine_env(
        grid_size=cfg["grid_size"], num_agents=cfg["num_agents"],
        obstacle_ratio=cfg["obstacle_ratio"], seed=env_seed,
        info_model=cfg["info_model"], comm_range=cfg["comm_range"],
        num_mines=cfg["num_mines"], detectability=cfg["detectability"],
        p_bar=cfg["p_bar"], strata=cfg["strata"])
    policies = [build_allocation(alloc_name, policy_seed_for(env_seed, i), i, cfg)
                for i in range(cfg["num_agents"])]

    ak = cfg.get("aaars", {})
    aaars = AAARSController(
        window_size=ak.get("window_size", 8), grid_size=cfg["grid_size"],
        num_bins=ak.get("num_bins", 5), w_temporal=ak.get("w_temporal", 0.0),
        w_coverage=ak.get("w_coverage", 0.55), w_frequency=ak.get("w_frequency", 0.45),
        ema_alpha=ak.get("ema_alpha", 0.1), base_alpha=ak.get("base_alpha", 0.05),
        risk_lambda=ak.get("risk_lambda", 1.0),
        blend_threshold=ak.get("blend_threshold", 0.30))
    gap_sprt = GapSPRT(p0=0.6, p1=0.03, alpha=0.05, beta=0.05)

    acc = {m: 0.0 for m in METHODS}
    nsteps = {m: 0 for m in METHODS}
    aaars_stop_n = None
    chao1_stop_n = None

    prev_n_det = -1
    prev_scanmask = None

    t_last = 0
    for t in range(1, cfg["max_steps"] + 1):
        t_last = t
        if cfg["info_model"] == "comm_limited":
            env.check_and_merge()
        actions = [policies[i].select_action(env, i)[0]
                   for i in range(cfg["num_agents"])]
        env.step(actions)
        bits = env.fleet_block_bits()

        _t0 = time.perf_counter()
        est1 = residual_estimate(bits)
        f2_floor = max(5.0, float(np.ceil(0.08 * est1["n_det"])))
        if chao1_stop_n is None and est1["n_det"] >= 5 and est1["f2"] >= f2_floor:
            if est1["ci_upper"] <= 0.05 * est1["K_hat"]:
                chao1_stop_n = t
        acc["chao1"] += time.perf_counter() - _t0
        nsteps["chao1"] += 1

        sc_fleet = env.fleet_scan_max()
        traversable = env.traversable
        cover = float(np.count_nonzero((sc_fleet > 0) & ~env.obstacle_map)) / \
            traversable if traversable else 0.0

        cur_n_det = est1["n_det"]
        new_find = max(0, cur_n_det - prev_n_det)
        sc_now = env.fleet_scan_max()
        dom = env.fleet_area_domain()
        cov_mask = sc_now > 0
        cov_frac = float(np.count_nonzero(cov_mask & dom)) / \
            max(float(np.count_nonzero(dom)), 1.0)
        new_cells = int(np.count_nonzero((sc_now > 0) & ~(prev_scanmask > 0))) \
            if prev_scanmask is not None else int(np.count_nonzero(cov_mask & dom))
        prev_scanmask = sc_now
        prev_n_det = cur_n_det

        percept = PerceptStep(
            t=t, bits=bits, fleet_coverage=cover,
            new_cells_scanned=new_cells, new_finds=new_find,
            coverage_frac=cov_frac,
        )

        _t0 = time.perf_counter()
        ar = aaars.step(percept)
        acc["aaars"] += time.perf_counter() - _t0
        nsteps["aaars"] += 1
        if aaars_stop_n is None and ar["stop"]:
            aaars_stop_n = t

        _t0 = time.perf_counter()
        gap_sprt.update(percept)
        acc["gap_sprt"] += time.perf_counter() - _t0
        nsteps["gap_sprt"] += 1

    return {
        "alloc": alloc_name, "seed": seed_idx, "env_seed": env_seed,
        "T_total": t_last,
        "aaars_stop_t": aaars_stop_n, "chao1_stop_t": chao1_stop_n,
        "total_s": {m: round(acc[m], 6) for m in METHODS},
        "n_steps": {m: nsteps[m] for m in METHODS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    env_seed = a.seed * 1000
    res = bench_episode(a.alloc, a.seed, env_seed)
    path = os.path.join(_PROJECT_ROOT, "results", "raw", a.out)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    us = {m: 1e6 * res["total_s"][m] / max(res["n_steps"][m], 1) for m in METHODS}
    print(f"saved {a.out}: alloc={res['alloc']} seed={res['seed']} "
          f"T={res['T_total']} | us/step chao1={us['chao1']:.1f} "
          f"aaars={us['aaars']:.1f} gap_sprt={us['gap_sprt']:.1f}", flush=True)


if __name__ == "__main__":
    main()
