#!/usr/bin/env python
"""
gamma_extract.py — extract leak-free spatial scalars for the realistic biased
policies (frontier / greedy / hotspot; 80 seeds each) by re-simulating with
collect_percepts and decoding the bits grid at the AAARS stop time.

Per episode we store:
  n_prime  : total detection hits in the populated cells == sum_k k*f_k
             (cumulative-draws denominator already used by d1_prime_extract)
  n_cells  : distinct populated (scanned) cells == sum_k f_k
  gamma2   : squared coefficient of variation of the per-cell hit counts over
             the populated cells (true spatial concentration / revisit metric,
             mirroring Chao92's gamma2) = (1/(s-1)*sum (fi-mean)^2)/mean^2
  revisit  : n_prime / n_cells  (mean revisits per scanned cell)
plus the AAARS stop, recall and FC label.

Never touches ground truth: only the leak-free PerceptStep bits (scan/fusion
hits). Parallelisation follows the project's subprocess-per-chunk pattern
(multiprocessing.Pool deadlocks on this Python 3.14/Windows setup).

Run:
    python experiments/power/gamma_extract.py --workers 8 --allocs greedy hotspot frontier
"""
import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}
THIS = os.path.abspath(__file__)


def gamma2_of_bits(bits):
    fi = bits.ravel()
    fi = fi[fi > 0].astype(float)
    if fi.size == 0:
        return None, 0, 0, None, None, None
    s = int(fi.size)
    n = float(fi.sum())
    mean = fi.mean()
    g2 = 0.0
    if mean > 0:
        var = ((fi - mean) ** 2).sum() / (s - 1)
        g2 = var / (mean * mean)
    # spatial concentration of *visits* (distinct from capture heterogeneity):
    p = fi / fi.sum()
    entropy = float(-(p * np.log(p)).sum()) / np.log(s) if s > 1 else 0.0
    k = max(1, int(round(0.10 * s)))
    order = np.sort(fi)[::-1]
    topk = float(order[:k].sum() / fi.sum())
    return float(g2), int(n), s, float(n / s), entropy, topk


def extract_one(cfg, seed_idx, env_seed, stop_t, grid, alloc):
    from src.experiments.runner import run_episode
    from src.percepts.io import percept_from_dict
    t0 = time.perf_counter()
    res = run_episode(alloc, seed_idx, env_seed, cfg=cfg,
                      collect_percepts=True, collect_trace=False)
    stream = json.loads(res["_percepts"])
    s = min(stop_t, len(stream)) if stop_t is not None else len(stream)
    idx = (s - 1) if (s and 0 <= s - 1 < len(stream)) else 0
    pp = percept_from_dict(stream[idx], grid)
    g2, n, nc, rev, entr, topk = gamma2_of_bits(pp.bits)
    return {"alloc": alloc, "run": seed_idx, "env_seed": env_seed,
            "aaars_t": stop_t, "n_prime": n, "n_cells": nc,
            "gamma2": g2, "revisit": rev, "entropy": entr, "topk": topk,
            "recall": res.get("aaars__recall"),
            "fc": int(res.get("aaars__t") is not None
                      and res.get("aaars__recall", 0) < RECALL_THR),
            "wall_s": time.perf_counter() - t0}


def run_chunk(chunk_ids, allocs, chunk_out):
    from src.experiments.runner import DEFAULT_CFG
    cohort = json.load(open(os.path.join(RAW, "policies_results.json")))
    by = {}
    for a in allocs:
        by[a] = {r["run"]: r for r in cohort if r.get("alloc") == a}
    cfg = {**DEFAULT_CFG, "max_steps": 6000, "coverage_only_threshold": 0.95,
           "aaars": AAARS_PARAMS}
    grid = cfg["grid_size"]
    outs = []
    for a in allocs:
        for seed_idx in chunk_ids:
            rec = by[a].get(seed_idx)
            if not rec:
                continue
            o = extract_one(cfg, seed_idx, rec["env_seed"],
                            rec.get("aaars__t"), grid, a)
            outs.append(o)
            print(f"chunk {os.path.basename(chunk_out)} {a} seed={seed_idx} "
                  f"n={o['n_prime']} nc={o['n_cells']} g2={o['gamma2']} "
                  f"fc={o['fc']}", flush=True)
    with open(chunk_out, "w") as f:
        json.dump(outs, f, indent=2)
    return outs


def main():
    import multiprocessing  # noqa: F401  (importing is fine)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allocs", nargs="+",
                    default=["greedy", "hotspot", "frontier"])
    ap.add_argument("--chunk-idx", type=int, default=None)
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--out", type=str, default="gamma_policies.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip chunks whose tmp.json already exists")
    ap.add_argument("--merge", action="store_true",
                    help="collect any existing chunk tmp.json files into the final output and exit")
    args = ap.parse_args()

    cohort = json.load(open(os.path.join(RAW, "policies_results.json")))
    runs = sorted({r["run"] for r in cohort
                   if r.get("alloc") in args.allocs})
    print(f"{len(runs)} runs x {len(args.allocs)} allocs -> {args.allocs}",
          flush=True)

    if args.chunk_idx is not None:
        nchunks = args.n_chunks or args.workers
        chunks = [runs[i::nchunks] for i in range(nchunks)]
        chunk_out = os.path.join(RAW, f"{args.out}.{args.chunk_idx}.tmp.json")
        run_chunk(chunks[args.chunk_idx], args.allocs, chunk_out)
        print(f"worker {args.chunk_idx} done -> {chunk_out}", flush=True)
        sys.exit(0)

    n = args.workers

    if args.merge:
        merged = []
        for i in range(n):
            cp = os.path.join(RAW, f"{args.out}.{i}.tmp.json")
            if os.path.exists(cp):
                merged += json.load(open(cp))
        final = os.path.join(RAW, args.out)
        json.dump(merged, open(final, "w"), indent=2)
        print(f"MERGED {len(merged)} episodes -> {final}", flush=True)
        sys.exit(0)

    DETACH = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags = DETACH  # child survives parent-shell kill without a console window

    launched = 0
    for i in range(n):
        base = os.path.join(RAW, f"{args.out}.{i}")
        cp = base + ".tmp.json"
        if args.resume and os.path.exists(cp):
            print(f"chunk {i}: exists, skipped (resume)", flush=True)
            continue
        for suf in (".tmp.json", ".log", ".err.log"):
            p = base + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, THIS, "--chunk-idx", str(i),
               "--n-chunks", str(n), "--out", args.out,
               "--allocs"] + args.allocs
        with open(base + ".log", "w") as lf, open(base + ".err.log", "w") as ef:
            p = subprocess.Popen(cmd, cwd=_ROOT, stdout=lf, stderr=ef,
                                 creationflags=flags)
        launched += 1
        print(f"launched chunk {i} pid={p.pid} (detached)", flush=True)

    print(f"launched {launched} chunk worker(s); run '--merge' to collect "
          f"results when finished", flush=True)


if __name__ == "__main__":
    main()
