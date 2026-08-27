#!/usr/bin/env python
"""
ablation_campaign.py — Component ablation for AAARS.

Tests which component contributes most to performance:
  1. full          — AAARS with all components
  2. no_blend      — AAARS without Chao92 blend (blend_threshold=1.0)
  3. no_alpha      — AAARS without alpha adaptation (risk_lambda=0)
  4. discrete      — Discrete switching (ablation baseline)
  5. chao1_ci      — Pure Chao1 CI (no adaptation)
  6. chao92_ci     — Pure Chao92 CI (conservative baseline)
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
NUM_SEEDS = 10
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"]

BASE_CFG = {**DEFAULT_CFG, "max_steps": 6000}

# Variant configurations
VARIANTS = {
    "full": {
        "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
        "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
        "blend_threshold": 0.30, "num_bins": 5,
    },
    "no_blend": {
        "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
        "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
        "blend_threshold": 1.0, "num_bins": 5,
    },
    "no_alpha": {
        "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
        "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 0.0,
        "blend_threshold": 0.30, "num_bins": 5,
    },
}


def main():
    print("=" * 70)
    print("AAARS ABLATION CAMPAIGN")
    print("  100x100 grid, K=60 mines, N=6 agents, Tmax=6000")
    print("  Seeds 10-%d (%d seeds)" % (SEED_OFFSET + NUM_SEEDS - 1, NUM_SEEDS))
    print("=" * 70, flush=True)

    all_results = []
    t_start = time.perf_counter()
    total = NUM_SEEDS * len(ALLOCATIONS)
    done = 0

    for seed_idx in range(SEED_OFFSET, SEED_OFFSET + NUM_SEEDS):
        env_seed = seed_idx * SEED_STRIDE
        for alloc in ALLOCATIONS:
            done += 1
            t0 = time.perf_counter()

            # Run each variant
            variant_results = {}
            for vname, vparams in VARIANTS.items():
                cfg_run = {**BASE_CFG, "aaars": vparams}
                r = run_episode(alloc, seed_idx, env_seed, cfg=cfg_run,
                                collect_trace=True)
                variant_results[vname] = r

            # Also run chao1_ci and chao92_ci (they don't use aaars params)
            for baseline in ["chao1_ci", "chao92_ci", "oracle_95", "fixed_2",
                             "diminishing"]:
                r = run_episode(alloc, seed_idx, env_seed, cfg=BASE_CFG,
                                collect_trace=False)
                variant_results[baseline] = r

            elapsed = time.perf_counter() - t0
            total_elapsed = time.perf_counter() - t_start
            eta = (total_elapsed / done) * (total - done)

            # Merge all results
            merged = {"alloc": alloc, "seed_idx": seed_idx, "env_seed": env_seed}
            for vname, vr in variant_results.items():
                for k, v in vr.items():
                    merged[f"{vname}__{k}"] = v

            all_results.append(merged)

            # Progress
            aaars_t = merged.get("full__aaars__t")
            aaars_fc = merged.get("full__aaars__recall", 0)
            no_blend_t = merged.get("no_blend__aaars__t")
            no_alpha_t = merged.get("no_alpha__aaars__t")
            chao1_t = merged.get("chao1_ci__chao1_ci__t")
            print(f"  [{done:>3d}/{total}] {alloc:8s} seed={seed_idx:<3d} "
                  f"full={str(aaars_t):>5s}/{aaars_fc:5.1f}% "
                  f"no_blend={str(no_blend_t):>5s} "
                  f"no_alpha={str(no_alpha_t):>5s} "
                  f"chao1={str(chao1_t):>5s} "
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]",
                  flush=True)

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"All {len(all_results)} episodes done in {total_time/60:.1f} min")

    # Save raw
    raw_dir = os.path.join(_PROJECT_ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, "ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    methods = ["full", "no_blend", "no_alpha"]
    baselines = ["chao1_ci", "chao92_ci", "discrete_aaars", "oracle_95",
                 "fixed_2", "diminishing"]

    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")

    header = (f"{'Variant':18s} {'Alloc':10s} {'Stops':>6s} {'FC':>5s} "
              f"{'FC%':>6s} {'Med_t':>7s} {'Med_R':>7s}")
    print(f"\n{header}")
    print("-" * len(header))

    for method in methods + baselines:
        for alloc in ALLOCATIONS:
            subset = [r for r in all_results if r["alloc"] == alloc]

            if method in methods:
                key_t = f"{method}__aaars__t"
                key_r = f"{method}__aaars__recall"
            else:
                key_t = f"{method}__{method}__t"
                key_r = f"{method}__{method}__recall"

            stops = [r for r in subset if r.get(key_t) is not None]
            n_stops = len(stops)
            n_total = len(subset)

            if n_stops > 0:
                times = [r[key_t] for r in stops]
                recalls = [r[key_r] for r in stops]
                fc = sum(1 for r in stops if r[key_r] < 95.0)
                med_t = int(np.median(times))
                med_r = round(float(np.median(recalls)), 1)
                fc_pct = 100.0 * fc / n_stops
                print(f"{method:18s} {alloc:10s} {n_stops:>4d}/{n_total:<2d} "
                      f"{fc:>3d} {fc_pct:>5.1f}% "
                      f"{med_t:>7d} {med_r:>6.1f}%")
            else:
                print(f"{method:18s} {alloc:10s} {n_stops:>4d}/{n_total:<2d} "
                      f"{'---':>5s} {'---':>6s} {'---':>7s} {'---':>7s}")

    print(f"\nTotal time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
