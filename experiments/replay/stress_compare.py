#!/usr/bin/env python
"""
stress_compare.py — D5 stress test at full manuscript scale.

Runs a live 100x100 / 6000-step episode for each allocation, records the full
PerceptStep stream (collect_percepts=True), replays it through every reproduced
stopping rule (see replay_episode.py), and diffs replay vs. live stop times.

This is the "full-scale --compare": it exercises late-arming rules (chao92_ci,
diminishing) and the horizon-edge case (a rule whose stop time stays None) that
smaller-grid smoke tests cannot reach, and confirms bit-for-bit reproduction at
the exact configuration the manuscript reports.

The rules are pure functions of the PerceptStep stream, so parallelism is capped
at the number of independent live episodes (one full episode per allocation); a
faithful compare cannot split a single episode across processes.

Usage:
    python experiments/replay/stress_compare.py [--grid 100 --max-steps 6000]
                                                [--seed 10 --workers N]
Exit code 0 iff every reproduced rule matches replay==live for every allocation.
"""

import json
import os
import sys
import time
import argparse
import multiprocessing as mp

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.experiments.runner import run_episode, DEFAULT_CFG          # noqa: E402
from experiments.replay.replay_episode import replay, REPRODUCED_RULES  # noqa: E402

ALLOCATIONS = ["boustro", "minerich"]


def _worker(args):
    alloc, grid, max_steps, seed, cov = args
    cfg = {**DEFAULT_CFG, "grid_size": grid, "max_steps": max_steps,
           "coverage_only_threshold": cov}
    env_seed = seed * 1000
    t0 = time.time()
    live = run_episode(alloc, seed, env_seed, cfg=cfg,
                       collect_percepts=True, collect_trace=False)
    stream = json.loads(live["_percepts"])
    stop = replay(stream, cfg=cfg)
    dt = time.time() - t0
    ok = {r: (live.get(f"{r}__t") == stop[r]) for r in REPRODUCED_RULES}
    live_t = {r: live.get(f"{r}__t") for r in REPRODUCED_RULES}
    return alloc, dt, len(stream), ok, stop, live_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--cov", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: capped at n allocations)")
    args = ap.parse_args()

    tasks = [(a, args.grid, args.max_steps, args.seed, args.cov)
             for a in ALLOCATIONS]
    workers = args.workers or min(mp.cpu_count(), len(tasks))
    print(f"grid={args.grid}x{args.grid} max_steps={args.max_steps} "
          f"workers={workers} (capped at {len(tasks)} episodes)", flush=True)

    t_start = time.time()
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    wall = time.time() - t_start

    all_ok = True
    for alloc, dt, n, ok, stop, live_t in results:
        all_ok &= all(ok.values())
        tag = {r: ("OK " if ok[r] else "DIFF") for r in REPRODUCED_RULES}
        print(f"\n== {alloc}: {n} percepts, live+replay {dt:.1f}s ==")
        for r in REPRODUCED_RULES:
            print(f"  [{tag[r]}] {r:16s} live={str(live_t[r]):>6s} "
                  f"replay={str(stop[r]):>6s}")
    print(f"\nwall {wall:.1f}s | "
          f"RESULT: {'ALL MATCH — STRESS PASS' if all_ok else 'MISMATCH'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
