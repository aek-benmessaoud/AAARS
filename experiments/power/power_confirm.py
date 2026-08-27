#!/usr/bin/env python
"""
power_confirm.py — Higher-powered confirmation campaign.

80 seeds per allocation (seeds 10-89), K=60, N=6, both allocations.
Saves raw per-episode outcomes to results/raw/power_confirm.json
and computes binomial CIs + Fisher's exact on FC rates.

Runs WITHOUT per-step traces (outcome-only) for speed.
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

SEED_OFFSET = 10
NUM_SEEDS = int(os.environ.get("POWER_SEEDS", 80))
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"]

BASE_CFG = {**DEFAULT_CFG, "max_steps": 6000}

AAARS_PARAMS = {
    "w_temporal": 0.0,
    "w_coverage": 0.55,
    "w_frequency": 0.45,
    "ema_alpha": 0.1,
    "base_alpha": 0.05,
    "risk_lambda": 1.0,
    "blend_threshold": 0.30,
    "num_bins": 5,
}


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def fisher_exact(a, b, c, d):
    """Two-tailed Fisher's exact test p-value (hypergeometric)."""
    from math import lgamma, exp
    table = [[a, b], [c, d]]
    n = a + b + c + d
    row_tot = [a + b, c + d]
    col_tot = [a + c, b + d]

    def log_hypergeom(x):
        # x = top-left cell count
        return (lgamma(row_tot[0] + 1) + lgamma(row_tot[1] + 1)
                + lgamma(col_tot[0] + 1) + lgamma(col_tot[1] + 1)
                - lgamma(n + 1)
                - lgamma(x + 1) - lgamma(a + b - x + 1)
                - lgamma(a + c - x + 1) - lgamma(b + d - (a + b - x) + 1))

    lo = max(0, a - min(row_tot[0], col_tot[0]))
    hi = min(a + b, a + c)
    p_obs = exp(log_hypergeom(a))
    total = 0.0
    for x in range(lo, hi + 1):
        p = exp(log_hypergeom(x))
        if p <= p_obs * (1 + 1e-12):
            total += p
    return total


def main():
    print("=" * 70)
    print(f"AAARS POWER CONFIRMATION — {NUM_SEEDS} seeds")
    print("  100x100 grid, K=60, N=6, Tmax=6000")
    print(f"  Seeds {SEED_OFFSET}-{SEED_OFFSET+NUM_SEEDS-1}")
    print("=" * 70, flush=True)

    all_results = []
    total_episodes = NUM_SEEDS * len(ALLOCATIONS)
    done = 0
    t_start = time.perf_counter()

    for seed_idx in range(SEED_OFFSET, SEED_OFFSET + NUM_SEEDS):
        env_seed = seed_idx * SEED_STRIDE
        for alloc in ALLOCATIONS:
            done += 1
            t0 = time.perf_counter()
            cfg_run = {**BASE_CFG, "aaars": AAARS_PARAMS}
            result = run_episode(alloc, seed_idx, env_seed,
                                 cfg=cfg_run, collect_trace=False)
            elapsed = time.perf_counter() - t0
            all_results.append(result)

            aaars_t = result.get("aaars__t")
            chao1_t = result.get("chao1_ci__t")
            total_elapsed = time.perf_counter() - t_start
            eta = (total_elapsed / done) * (total_episodes - done)

            print(f"  [{done:>3d}/{total_episodes}] {alloc:8s} seed={seed_idx:<3d} "
                  f"aaars={str(aaars_t):>5s} chao1={str(chao1_t):>5s} "
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]",
                  flush=True)

    total_time = time.perf_counter() - t_start

    raw_path = os.path.join(_PROJECT_ROOT, "results", "raw",
                            "power_confirm.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw saved to {raw_path} ({total_time/60:.1f} min)")

    # ---- Statistical summary ----
    print("\n" + "=" * 70)
    print("FC RATE WITH BINOMIAL CI + FISHER EXACT (AAARS vs Chao1)")
    print("=" * 70)

    methods = ["chao1_ci", "aaars", "discrete_aaars", "chao92_ci",
               "oracle_95", "fixed_2", "diminishing"]

    for alloc in ALLOCATIONS:
        subset = [r for r in all_results if r["alloc"] == alloc]
        print(f"\n--- {alloc.upper()} ({len(subset)} episodes) ---")
        for method in methods:
            stops = [r for r in subset if r[f"{method}__t"] is not None]
            n = len(subset)
            ns = len(stops)
            if ns == 0:
                print(f"  {method:16s}: 0/{n} stops")
                continue
            fc = sum(1 for r in stops if r[f"{method}__recall"] < 95.0)
            lo, hi = wilson_ci(fc, n)
            times = [r[f"{method}__t"] for r in stops]
            med_t = np.median(times)
            print(f"  {method:16s}: {ns}/{n} stops, FC={fc} "
                  f"({100.0*fc/n:.1f}% [{lo:.1f},{hi:.1f}]) "
                  f"med_t={med_t:.0f}")

        # Fisher exact AAARS vs Chao1 on FC counts
        aars = [r for r in subset if r.get("aaars__t") is not None]
        chao = [r for r in subset if r.get("chao1_ci__t") is not None]
        if aars and chao:
            ma = sum(1 for r in aars if r["aaars__recall"] < 95.0)
            mch = sum(1 for r in chao if r["chao1_ci__recall"] < 95.0)
            nA = len(aars); nC = len(chao)
            p = fisher_exact(ma, nA - ma, mch, nC - mch)
            print(f"  FISHER(AAARS vs Chao1, FC): AAARS {ma}/{nA} vs "
                  f"Chao1 {mch}/{nC}  p={p:.4f}")


if __name__ == "__main__":
    main()
