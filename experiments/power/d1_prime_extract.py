#!/usr/bin/env python
"""
d1_prime_extract.py — (re)run the canonical 80-minerich confirmation cohort and
extract, for each episode, the cumulative-detection denominator n = sum_k k*f_k
(D1'), plus the secondary "distinct scanned cells" n_cells and n_det.

Context (advisor Step-3 prep): a missing-mass / Good-Turing concentration bound
uses n = total draws = cumulative detections (D1': sum_k k f_k), not spatial
coverage. That is NOT persisted in power_revision.json (only coverage_series),
so it must be re-derived. The bits grid is recorded per step via
collect_percepts=True; at the AAARS stop time we decode it and, since each set
bit is one detection hit, n = bits.sum() == sum_k k f_k exactly (full spectrum,
no truncation).

Authoritative FC labels / stop times come from the committed power_revision.json
so the re-sim only regenerates the bits needed for the denominator (not the
rule outputs). Same seeds & config as power_revision (seeds 10-89, minerich,
grid=100, max_steps=6000, cov_thr=0.95, AAARS risk_lambda=1.0).

Parallelisation: multiprocessing.Pool is avoided (it deadlocks on this Python
3.14/Windows setup); the launcher instead spawns independent worker processes
via subprocess (one per chunk), matching the project's run_*_parallel pattern.

    # launcher (merges all chunks):
    python experiments/power/d1_prime_extract.py --workers 8
    # direct single-chunk worker mode:
    python experiments/power/d1_prime_extract.py --chunk-idx 0 --n-chunks 8
"""
import os
import sys
import json
import time
import argparse
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
RAW = os.path.join(_ROOT, "results", "raw")

AAARS_PARAMS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}
SEED_STRIDE = 1000
THIS = os.path.abspath(__file__)


def extract_one(cfg, seed_idx, env_seed, stop_t, grid):
    """Run one episode with collect_percepts and extract n at the AAARS stop."""
    from src.experiments.runner import run_episode
    from src.percepts.io import percept_from_dict
    import numpy as np
    t0 = time.perf_counter()
    res = run_episode("minerich", seed_idx, env_seed, cfg=cfg,
                      collect_percepts=True, collect_trace=False)
    stream = json.loads(res["_percepts"])
    s = min(stop_t, len(stream)) if stop_t is not None else len(stream)
    idx = (s - 1) if (s and s - 1 >= 0 and s - 1 < len(stream)) else 0
    pp = percept_from_dict(stream[idx], grid)
    fk = pp.bits.ravel()
    n = int(fk.sum())                       # D1': sum_k k f_k == cumulative detections
    n_cells = int(np.count_nonzero(fk))     # D1 secondary: distinct scanned cells
    return {"run": seed_idx, "env_seed": env_seed, "aaars_t": stop_t,
            "n_prime": n, "n_cells": n_cells,
            "cover_at_stop": float(pp.fleet_coverage),
            "wall_s": time.perf_counter() - t0}


def run_chunk(chunk_ids, chunk_out):
    """Run a list of seed indices serially; write compact results to chunk_out."""
    from src.experiments.runner import DEFAULT_CFG
    cohort = json.load(open(os.path.join(RAW, "power_revision.json")))
    by_run = {r["run"]: r for r in cohort if r.get("alloc") == "minerich"}
    cfg = {**DEFAULT_CFG, "max_steps": 6000, "coverage_only_threshold": 0.95,
           "aaars": AAARS_PARAMS}
    grid = cfg["grid_size"]
    outs = []
    for seed_idx in chunk_ids:
        rec = by_run.get(seed_idx)
        if not rec:
            continue
        o = extract_one(cfg, seed_idx, rec["env_seed"], rec.get("aaars__t"), grid)
        o["aaars_fc"] = int(rec.get("aaars__recall", 100.0) < 95.0)
        outs.append(o)
        print(f"chunk {os.path.basename(chunk_out)} seed={seed_idx} "
              f"n_prime={o['n_prime']} fc={o['aaars_fc']} [~{time.perf_counter()/60:.0f}m prog]",
              flush=True)
    with open(chunk_out, "w") as f:
        json.dump(outs, f, indent=2)
    return outs


def main():
    from src.experiments.runner import DEFAULT_CFG

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-idx", type=int, default=None,
                    help="worker mode: run only this chunk index")
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--out", type=str, default="d1_prime_minerich.json")
    args = ap.parse_args()

    cohort = json.load(open(os.path.join(RAW, "power_revision.json")))
    minerich_runs = sorted({r["run"] for r in cohort if r.get("alloc") == "minerich"})
    print(f"{len(minerich_runs)} minerich runs", flush=True)

    # -- worker mode ------------------------------------------------
    if args.chunk_idx is not None:
        nchunks = args.n_chunks or args.workers
        mine_runs = minerich_runs
        chunks = [mine_runs[i::nchunks] for i in range(nchunks)]
        chunk_out = os.path.join(RAW, f"{args.out}.{args.chunk_idx}.tmp.json")
        run_chunk(chunks[args.chunk_idx], chunk_out)
        print(f"worker {args.chunk_idx} done -> {chunk_out}", flush=True)
        sys.exit(0)

    # -- launcher mode: spawn chunk workers via subprocess ----------
    n = args.workers
    procs = []
    for i in range(n):
        out_path = os.path.join(RAW, f"{args.out}.{i}.tmp.json")
        for suf in ("",):
            p = out_path + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, THIS, "--chunk-idx", str(i),
               "--n-chunks", str(n), "--out", args.out]
        with open(out_path + ".log", "w") as lf, \
             open(out_path + ".err.log", "w") as ef:
            p = subprocess.Popen(cmd, cwd=_ROOT, stdout=lf, stderr=ef)
        procs.append(p)
        print(f"launched chunk {i} pid={p.pid}", flush=True)

    t_start = time.perf_counter()
    for p in procs:
        p.wait()
    print(f"all workers done in {(time.perf_counter()-t_start)/60:.1f} min",
          flush=True)

    merged = []
    for i in range(n):
        cp = os.path.join(RAW, f"{args.out}.{i}.tmp.json")
        if os.path.exists(cp):
            merged += json.load(open(cp))
    final = os.path.join(RAW, args.out)
    json.dump(merged, open(final, "w"), indent=2)
    print(f"MERGED {len(merged)} episodes -> {final}", flush=True)


if __name__ == "__main__":
    main()
