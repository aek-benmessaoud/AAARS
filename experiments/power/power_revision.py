#!/usr/bin/env python
"""
power_revision.py — 80-seed confirmation re-run with the two new
isolation baselines (threshold_aaars, coverage_only).

Mirrors power_confirm.py (seeds 10-89, K=60, N=6, both allocations) but the
runner now also computes:
  - threshold_aaars : Chao1 + AAARS adaptive alpha, NO blending
  - coverage_only   : stop when coverage >= coverage_only_threshold
Each episode stores a full coverage series (coverage_series) so the coverage
threshold can be swept post-hoc in analysis.
"""
import os
import sys
import time
import json
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import run_episode, DEFAULT_CFG

parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=int, default=10)
parser.add_argument("--seeds", type=int, default=80)
parser.add_argument("--cov", type=float, default=0.95)
parser.add_argument("--out", type=str, default="power_revision.json")
parser.add_argument("--alloc", type=str, default=None,
                    help="single policy name; if set, run only this allocation")
args = parser.parse_args()

SEED_OFFSET = args.offset
NUM_SEEDS = args.seeds
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"] if args.alloc is None else [args.alloc]
COVERAGE_THRESHOLD = args.cov
OUT_NAME = args.out

BASE_CFG = {**DEFAULT_CFG, "max_steps": 6000,
            "coverage_only_threshold": COVERAGE_THRESHOLD}

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}


def main():
    print(f"AAARS POWER REVISION - {NUM_SEEDS} seeds, cov_thr={COVERAGE_THRESHOLD}")
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
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]", flush=True)

    print(f"\nAll {total_episodes} episodes done in "
          f"{(time.perf_counter()-t_start)/60:.1f} min")

    raw_path = os.path.join(_PROJECT_ROOT, "results", "raw", OUT_NAME)
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw saved to {raw_path}")


if __name__ == "__main__":
    main()
