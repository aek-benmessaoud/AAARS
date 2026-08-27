#!/usr/bin/env python
"""
power_sweep.py — Higher-powered KxN sweep.

20 seeds per (K,N,alloc) cell, K in {30,60,120}, N in {3,6,12}.
Saves raw to results/raw/power_sweep.json and per-cell binomial CI on FC%.
"""

import os
import sys
import time
import json
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import run_episode, DEFAULT_CFG

SEEDS = list(range(10, int(os.environ.get("POWER_SWEEP_SEEDS", 20)) + 10))
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"]

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}

K_VALUES = [30, 60, 120]
N_VALUES = [3, 6, 12]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def main():
    configs = [(k, n) for k in K_VALUES for n in N_VALUES]
    total = len(configs) * len(SEEDS) * len(ALLOCATIONS)

    print("=" * 70)
    print("AAARS POWER SWEEP")
    print(f"  K in {K_VALUES}, N in {N_VALUES}")
    print(f"  {len(SEEDS)} seeds x {len(ALLOCATIONS)} allocs x {len(configs)} = {total} episodes")
    print("=" * 70, flush=True)

    all_results = []
    t_start = time.perf_counter()
    done = 0

    for k, n in configs:
        cfg = {**DEFAULT_CFG, "num_mines": k, "num_agents": n,
               "max_steps": 6000, "aaars": AAARS_PARAMS}
        print(f"\n--- K={k}, N={n} ---", flush=True)
        for seed_idx in SEEDS:
            env_seed = seed_idx * SEED_STRIDE
            for alloc in ALLOCATIONS:
                done += 1
                t0 = time.perf_counter()
                r = run_episode(alloc, seed_idx, env_seed,
                                cfg=cfg, collect_trace=False)
                elapsed = time.perf_counter() - t0
                total_elapsed = time.perf_counter() - t_start
                eta = (total_elapsed / done) * (total - done) if done > 0 else 0
                aaars_t = r.get("aaars__t")
                chao1_t = r.get("chao1_ci__t")
                print(f"  [{done:>3d}/{total}] K={k:<3d} N={n:<2d} {alloc:8s} "
                      f"aaars={str(aaars_t):>5s} chao1={str(chao1_t):>5s} "
                      f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]",
                      flush=True)
                all_results.append(r)

    total_time = time.perf_counter() - t_start

    raw_path = os.path.join(_PROJECT_ROOT, "results", "raw", "power_sweep.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw saved to {raw_path} ({total_time/60:.1f} min)")

    print("\n" + "=" * 80)
    print("SWEEP: FC% with WILSON 95% CI (MineRich)")
    print("=" * 80)
    print(f"{'K':>4s} {'N':>3s} {'Chao1 FC%':>22s} {'AAARS FC%':>22s}")
    for k in K_VALUES:
        for n in N_VALUES:
            subset = [r for r in all_results if r["alloc"] == "minerich"
                      and r["num_mines"] == k and r["num_agents"] == n]
            stops_a = [r for r in subset if r.get("aaars__t") is not None]
            stops_c = [r for r in subset if r.get("chao1_ci__t") is not None]
            fa = sum(1 for r in stops_a if r["aaars__recall"] < 95.0)
            fc = sum(1 for r in stops_c if r["chao1_ci__recall"] < 95.0)
            la, ha = wilson_ci(fa, len(subset))
            lc, hc = wilson_ci(fc, len(subset))
            print(f"{k:>4d} {n:>3d} "
                  f"{fc:>2d}/{len(subset):<2d} ({100.0*fc/len(subset):.0f}% [{lc:.0f},{hc:.0f}]) "
                  f"{fa:>2d}/{len(subset):<2d} ({100.0*fa/len(subset):.0f}% [{la:.0f},{ha:.0f}])")


if __name__ == "__main__":
    main()
