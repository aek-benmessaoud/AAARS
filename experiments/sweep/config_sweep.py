#!/usr/bin/env python
"""
config_sweep.py — Vary K (mine count) and N (agent count) to test robustness.

Sweep matrix:
  K ∈ {30, 60, 120}   (mine counts)
  N ∈ {3, 6, 12}      (agent counts)
  Grid: 100×100 fixed
  Allocations: boustro, minerich
  Seeds: 10-14 (5 seeds per config)
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

SEEDS = list(range(10, 15))
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"]

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}

K_VALUES = [30, 60, 120]
N_VALUES = [3, 6, 12]


def main():
    configs = [(k, n) for k in K_VALUES for n in N_VALUES]
    total = len(configs) * len(SEEDS) * len(ALLOCATIONS)

    print("=" * 70)
    print("AAARS CONFIG SWEEP")
    print(f"  K in {K_VALUES}, N in {N_VALUES}, Grid=100x100")
    print(f"  {len(SEEDS)} seeds × {len(ALLOCATIONS)} allocs × {len(configs)} configs = {total} episodes")
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
                r = run_episode(alloc, seed_idx, env_seed, cfg=cfg,
                                collect_trace=False)
                elapsed = time.perf_counter() - t0
                total_elapsed = time.perf_counter() - t_start
                eta = (total_elapsed / done) * (total - done) if done > 0 else 0

                aaars_t = r.get("aaars__t")
                aaars_r = r.get("aaars__recall", 0)
                chao1_t = r.get("chao1_ci__t")

                print(f"  [{done:>3d}/{total}] K={k:<3d} N={n:<2d} {alloc:8s} "
                      f"aaars={str(aaars_t):>5s}/{aaars_r:5.1f}% "
                      f"chao1={str(chao1_t):>5s} "
                      f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]",
                      flush=True)

                all_results.append(r)

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"All {total} episodes done in {total_time/60:.1f} min")

    # Save raw
    raw_dir = os.path.join(_PROJECT_ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, "config_sweep.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary by K and N
    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY: AAARS vs Chao1 by K and N")
    print(f"{'=' * 70}")
    header = f"{'K':>4s} {'N':>3s} {'Alloc':8s} {'AAARS Stops':>12s} {'AAARS FC':>9s} {'AAARS FC%':>9s} {'Chao1 Stops':>12s} {'Chao1 FC':>9s}"
    print(f"\n{header}")
    print("-" * len(header))

    for k in K_VALUES:
        for n in N_VALUES:
            for alloc in ALLOCATIONS:
                subset = [r for r in all_results
                          if r["alloc"] == alloc
                          and r["num_mines"] == k
                          and r["num_agents"] == n]

                # AAARS
                aaars_stops = [r for r in subset
                               if r.get("aaars__t") is not None]
                aaars_fc = sum(1 for r in aaars_stops
                               if r.get("aaars__recall", 100) < 95.0)
                aaars_n = len(subset)
                aaars_fc_pct = (100.0 * aaars_fc / len(aaars_stops)
                                if aaars_stops else 0)

                # Chao1
                chao1_stops = [r for r in subset
                               if r.get("chao1_ci__t") is not None]
                chao1_fc = sum(1 for r in chao1_stops
                               if r.get("chao1_ci__recall", 100) < 95.0)

                print(f"{k:>4d} {n:>3d} {alloc:8s} "
                      f"{len(aaars_stops):>4d}/{aaars_n:<2d} "
                      f"{aaars_fc:>5d} {aaars_fc_pct:>7.1f}% "
                      f"{len(chao1_stops):>4d}/{aaars_n:<2d} "
                      f"{chao1_fc:>5d}")

    print(f"\nTotal time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
