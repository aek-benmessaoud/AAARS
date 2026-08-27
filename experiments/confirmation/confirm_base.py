#!/usr/bin/env python
"""
confirmation_campaign.py — Base config confirmation for AAARS.

Seeds 10-29 (20 seeds), 100x100 grid, K=60, N=6, both allocations.
Uses thresholds from discovery campaign.
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

from src.experiments.runner import run_episode, AAARS_STOP_RULES, DEFAULT_CFG

# Confirmation set: seeds 10-29
SEED_OFFSET = 10
NUM_SEEDS = 20
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"]

# Base config: 100x100, K=60, N=6
BASE_CFG = {**DEFAULT_CFG, "max_steps": 6000}

# Tuned thresholds from discovery
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


def run_all_methods(alloc, seed_idx, env_seed, cfg):
    """Run one episode, return result dict."""
    cfg_run = {**cfg, "aaars": AAARS_PARAMS}
    return run_episode(alloc, seed_idx, env_seed, cfg=cfg_run, collect_trace=True)


def main():
    print("=" * 70)
    print("AAARS CONFIRMATION CAMPAIGN — BASE CONFIG")
    print("  100x100 grid, K=60 mines, N=6 agents, Tmax=6000")
    print("  Seeds 10-%d (%d seeds)" % (SEED_OFFSET + NUM_SEEDS - 1, NUM_SEEDS))
    print("  Allocations: boustro, minerich")
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
            result = run_all_methods(alloc, seed_idx, env_seed, BASE_CFG)
            elapsed = time.perf_counter() - t0
            all_results.append(result)

            # Progress
            aaars_t = result.get("aaars__t")
            aaars_rec = result.get("aaars__recall", 0)
            chao1_t = result.get("chao1_ci__t")
            oracle_t = result.get("oracle_95__t")
            risk = result.get("aaars__final_risk", 0)
            total_elapsed = time.perf_counter() - t_start
            eta = (total_elapsed / done) * (total_episodes - done)

            print(f"  [{done:>3d}/{total_episodes}] {alloc:8s} seed={seed_idx:<3d} "
                  f"aaars={str(aaars_t):>5s}/{aaars_rec:5.1f}% "
                  f"chao1={str(chao1_t):>5s} "
                  f"oracle={str(oracle_t):>5s} "
                  f"risk={risk:.3f} "
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]",
                  flush=True)

    # ---- Aggregate results ----
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"All {total_episodes} episodes done in {total_time/60:.1f} min")
    print(f"{'=' * 70}")

    # Save raw results
    raw_path = os.path.join(_PROJECT_ROOT, "results", "raw",
                            "confirmation_base.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved to {raw_path}")

    # ---- Summary tables ----
    print(f"\n{'=' * 70}")
    print("SUMMARY BY METHOD AND ALLOCATION")
    print(f"{'=' * 70}")

    methods = ["chao1_ci", "chao92_ci", "aaars", "discrete_aaars",
               "oracle_95", "fixed_2", "diminishing"]

    header = f"{'Method':18s} {'Alloc':10s} {'Stops':>6s} {'FC':>5s} {'FC%':>6s} " \
             f"{'Med_t':>7s} {'Med_R':>7s} {'Min_R':>7s}"
    print(f"\n{header}")
    print("-" * len(header))

    for method in methods:
        for alloc in ALLOCATIONS:
            subset = [r for r in all_results if r["alloc"] == alloc]
            stops = [r for r in subset if r[f"{method}__t"] is not None]
            n_stops = len(stops)
            n_total = len(subset)

            if n_stops > 0:
                times = [r[f"{method}__t"] for r in stops]
                recalls = [r[f"{method}__recall"] for r in stops]
                fc = sum(1 for r in stops if r[f"{method}__recall"] < 95.0)
                med_t = int(np.median(times))
                med_r = round(float(np.median(recalls)), 1)
                min_r = round(float(min(recalls)), 1)
                fc_pct = 100.0 * fc / n_stops
                print(f"{method:18s} {alloc:10s} {n_stops:>4d}/{n_total:<2d} "
                      f"{fc:>3d} {fc_pct:>5.1f}% "
                      f"{med_t:>7d} {med_r:>6.1f}% {min_r:>6.1f}%")
            else:
                print(f"{method:18s} {alloc:10s} {n_stops:>4d}/{n_total:<2d} "
                      f"{'---':>5s} {'---':>6s} {'---':>7s} {'---':>7s} {'---':>7s}")

    # ---- Key comparison: AAARS vs Chao1 under minerich ----
    print(f"\n{'=' * 70}")
    print("KEY COMPARISON: AAARS vs Chao1 under minerich (the problem case)")
    print(f"{'=' * 70}")

    minerich = [r for r in all_results if r["alloc"] == "minerich"]
    for method in ["chao1_ci", "aaars", "discrete_aaars"]:
        stops = [r for r in minerich if r[f"{method}__t"] is not None]
        if stops:
            fc = sum(1 for r in stops if r[f"{method}__recall"] < 95.0)
            times = [r[f"{method}__t"] for r in stops]
            print(f"  {method:18s}: {fc}/{len(stops)} FC, "
                  f"median_t={int(np.median(times))}")

    # ---- Risk score analysis ----
    print(f"\n{'=' * 70}")
    print("AAARS RISK SCORE ANALYSIS")
    print(f"{'=' * 70}")
    for alloc in ALLOCATIONS:
        subset = [r for r in all_results if r["alloc"] == alloc]
        risks = [r.get("aaars__final_risk", 0) for r in subset]
        print(f"  {alloc:10s}: mean_risk={np.mean(risks):.4f}, "
              f"median_risk={np.median(risks):.4f}, "
              f"std={np.std(risks):.4f}")

    # Save summary
    summary = {
        "config": "base_100x100_K60_N6",
        "seeds": f"{SEED_OFFSET}-{SEED_OFFSET+NUM_SEEDS-1}",
        "aaars_params": AAARS_PARAMS,
        "total_episodes": total_episodes,
        "total_time_min": round(total_time / 60, 1),
    }
    summary_path = os.path.join(_PROJECT_ROOT, "results",
                                "confirmation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
