#!/usr/bin/env python
"""
checkpoint_n.py — advisor circularity test for the Step-3 D1' covariate.

Risk: n (D1', cumulative detections) measured *at the AAARS stop time* is the
moment the rule decides to certify, so low n(stop) ~ "the rule was tricked into
stopping early" ~ FC itself. The clean test measures n at a FIXED, EXTERNAL
checkpoint t0 chosen independently of the stop decision, then re-tests whether
FC vs n(t0) persists.

If FC vs n(t0) keeps high AUC (rule that stops with little cumulative discovery
at an equalized early age is the one that false-certifies later) -> real result.
If the relationship collapses to ~0.5-0.6 -> it was a stop-time tautology.

We sample n(t0) for a grid of checkpoints (all well before the typical stop) on
the CLEAN group (minerich, frontier). Reuses the subprocess-per-chunk pattern.
    python experiments/power/checkpoint_n.py --workers 8 --allocs minerich frontier
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
CHECKPOINTS = [100, 200, 300, 400]

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}
THIS = os.path.abspath(__file__)


def extract_one(cfg, alloc, seed_idx, env_seed, stop_t, grid):
    from src.experiments.runner import run_episode
    from src.percepts.io import percept_from_dict
    res = run_episode(alloc, seed_idx, env_seed, cfg=cfg,
                      collect_percepts=True, collect_trace=False)
    stream = json.loads(res["_percepts"])
    n_at = {}
    for t0 in CHECKPOINTS:
        idx = min(t0, len(stream)) - 1
        idx = max(0, min(idx, len(stream) - 1)) if stream else 0
        if stream:
            n_at[t0] = int(percept_from_dict(stream[idx], grid).bits.sum())
        else:
            n_at[t0] = 0
    s = min(stop_t, len(stream)) if stop_t is not None else len(stream)
    sidx = (s - 1) if (s and 0 <= s - 1 < len(stream)) else 0
    n_stop = int(percept_from_dict(stream[sidx], grid).bits.sum()) if stream else 0
    return {"alloc": alloc, "run": seed_idx,
            "aaars_t": stop_t, "n_stop": n_stop, "n_at": n_at,
            "fc": int(res.get("aaars__t") is not None
                      and res.get("aaars__recall", 0) < RECALL_THR)}


def run_chunk(chunk_ids, allocs, chunk_out):
    from src.experiments.runner import DEFAULT_CFG
    src = {}
    for a in allocs:
        if a == "minerich":
            cohort = json.load(open(os.path.join(RAW, "power_revision.json")))
            src[a] = {r["run"]: r for r in cohort if r.get("alloc") == a}
        else:
            cohort = json.load(open(os.path.join(RAW, "policies_results.json")))
            src[a] = {r["run"]: r for r in cohort if r.get("alloc") == a}
    cfg = {**DEFAULT_CFG, "max_steps": 6000, "coverage_only_threshold": 0.95,
           "aaars": AAARS_PARAMS}
    grid = cfg["grid_size"]
    outs = []
    for a in allocs:
        for seed_idx in chunk_ids:
            rec = src[a].get(seed_idx)
            if not rec:
                continue
            o = extract_one(cfg, a, seed_idx, rec["env_seed"],
                            rec.get("aaars__t"), grid)
            outs.append(o)
            print(f"chunk {os.path.basename(chunk_out)} {a} seed={seed_idx} "
                  f"n_stop={o['n_stop']} n@200={o['n_at'].get(200)} fc={o['fc']}",
                  flush=True)
    with open(chunk_out, "w") as f:
        json.dump(outs, f, indent=2)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allocs", nargs="+", default=["minerich", "frontier"])
    ap.add_argument("--chunk-idx", type=int, default=None)
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--out", type=str, default="checkpoint_n_clean.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    runs = sorted({r["run"] for r in json.load(
        open(os.path.join(RAW, "power_revision.json")))
        if r.get("alloc") in args.allocs})
    print(f"{len(runs)} runs x {args.allocs} allocs", flush=True)

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
        json.dump(merged, open(os.path.join(RAW, args.out), "w"), indent=2)
        print(f"MERGED {len(merged)} -> {args.out}", flush=True)
        sys.exit(0)

    DETACH = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for i in range(n):
        base = os.path.join(RAW, f"{args.out}.{i}")
        cp = base + ".tmp.json"
        if args.resume and os.path.exists(cp):
            print(f"chunk {i}: exists, skipped", flush=True)
            continue
        for suf in (".tmp.json", ".log", ".err.log"):
            p = base + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, THIS, "--chunk-idx", str(i),
               "--n-chunks", str(n), "--out", args.out,
               "--allocs"] + args.allocs
        with open(base + ".log", "w") as lf, open(base + ".err.log", "w") as ef:
            subprocess.Popen(cmd, cwd=_ROOT, stdout=lf, stderr=ef,
                             creationflags=DETACH)
        print(f"launched chunk {i}", flush=True)
    print("launched; run '--merge' to collect when finished", flush=True)


if __name__ == "__main__":
    main()
