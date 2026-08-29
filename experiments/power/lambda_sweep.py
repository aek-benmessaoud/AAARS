#!/usr/bin/env python
"""
lambda_sweep.py — sweep AAARS risk_lambda on one (lambda, alloc) cell over a
seed range. Outcome-only episodes. Writes a chunk JSON.

alpha_adj = base_alpha / (1 + lam * risk); lam=0 => plain fixed-threshold
confidence stopping (no risk adaptivity); larger lam => tighter threshold
under risk (fewer false certs, later stops).
"""
import os
import sys
import time
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import run_episode, DEFAULT_CFG

SEED_STRIDE = 1000

BASE_AAARS = {
    "w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30, "num_bins": 5,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--alloc", type=str, required=True,
                    choices=["boustro", "minerich"])
    ap.add_argument("--offset", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=80)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    aaars_kw = {**BASE_AAARS, "risk_lambda": args.lam}
    cfg = {**DEFAULT_CFG, "max_steps": 6000, "aaars": aaars_kw}

    results = []
    t_start = time.perf_counter()
    for i in range(args.seeds):
        seed_idx = args.offset + i
        env_seed = seed_idx * SEED_STRIDE
        t0 = time.perf_counter()
        r = run_episode(args.alloc, seed_idx, env_seed, cfg=cfg,
                        collect_trace=False)
        r["lam"] = args.lam
        results.append(r)
        el = time.perf_counter() - t0
        print(f"  lamb={args.lam} {args.alloc:8s} seed={seed_idx:<3d} "
              f"aaars_t={r.get('aaars__t')} chao1_t={r.get('chao1_ci__t')} "
              f"[{el:.0f}s]", flush=True)

    out_abs = os.path.join(_PROJECT_ROOT, "results", "raw", args.out)
    with open(out_abs, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved {len(results)} records -> {out_abs} "
          f"({(time.perf_counter()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
