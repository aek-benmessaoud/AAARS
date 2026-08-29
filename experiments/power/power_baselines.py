#!/usr/bin/env python
"""
power_baselines.py — 80-seed run for the modern anytime-valid stopping
baselines (rate_cs, gap_sprt).

Mirrors power_revision.py (seeds 10-89, K=60, N=6, both allocations) but the
runner now also computes, for every episode in the same shared trace:
  - rate_cs   : coverage-aware silent-run rule (safe / coverage-bearing)
  - gap_sprt  : Wald SPRT on the geometric inter-discovery gap (naive rate rule)

Because these are POST-HOC additions evaluated on the same shared detection
trace as the other baselines (all rules read leak-free per-step observations
from the loop), no re-run would be needed if results were regenerated from the
same trace; this produces a fresh, consistent set containing all rule keys.

Ratings use the standard convention elsewhere in the paper: an episode is a
FALSE-CERTIFY / missed find if the rule stopped at recall < 95%.
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
parser.add_argument("--out", type=str, default="power_baselines.json")
parser.add_argument("--alloc", type=str, default=None,
                    help="single policy name; if set, run only this allocation")
args = parser.parse_args()

SEED_OFFSET = args.offset
NUM_SEEDS = args.seeds
SEED_STRIDE = 1000
ALLOCATIONS = ["boustro", "minerich"] if args.alloc is None else [args.alloc]

BASE_CFG = {**DEFAULT_CFG, "max_steps": 6000,
            "coverage_only_threshold": args.cov}

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}


def main():
    print(f"P8 POWER BASELINES - {NUM_SEEDS} seeds, seeds {SEED_OFFSET}.."
          f"{SEED_OFFSET + NUM_SEEDS - 1}, allocs={ALLOCATIONS}", flush=True)
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
            total_elapsed = time.perf_counter() - t_start
            eta = (total_elapsed / done) * (total_episodes - done)
            rs = str(result.get("rate_cs__t"))
            gs = str(result.get("gap_sprt__t"))
            print(f"  [{done:>3d}/{total_episodes}] {alloc:8s} seed={seed_idx:<3d} "
                  f"rate_cs={rs:>5s} gap_sprt={gs:>5s} "
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}m]", flush=True)

    print(f"\nAll {total_episodes} episodes done in "
          f"{(time.perf_counter()-t_start)/60:.1f} min")

    raw_path = os.path.join(_PROJECT_ROOT, "results", "raw", args.out)
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw saved to {raw_path}")


if __name__ == "__main__":
    main()
