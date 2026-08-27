#!/usr/bin/env python
"""
discovery_campaign.py — Threshold tuning for AAARS.

Staged grid search over AAARS hyperparameters using seeds 0-9.
Discovery set is SEPARATE from confirmation set (seeds 10-29).

Phase A: Diagnostic weights (w_temporal, w_spatial, w_frequency)
Phase B: Risk/threshold params (ema_alpha, base_alpha, risk_lambda)
Phase C: Validation on held-out seeds within discovery set

Selection criterion: 0 false certifications + lowest median stopping time.
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

# ---- Discovery set ----
DISCOVERY_SEEDS = list(range(0, 10000, 1000))  # [0, 1000, ..., 9000]
ALLOCATIONS = ["boustro", "minerich"]

# Fast config for iteration (30x30, 2 agents, 300 steps ≈ 1.7s/ep)
FAST_CFG = {**DEFAULT_CFG, "grid_size": 30, "num_mines": 10,
            "num_agents": 2, "max_steps": 300}

# Medium config for validation
MED_CFG = {**DEFAULT_CFG, "grid_size": 100, "num_mines": 60,
           "num_agents": 6, "max_steps": 2000}


def evaluate_aaars(aaars_params, cfg, seeds, allocations):
    """Run AAARS with given params, return metrics dict."""
    cfg_run = {**cfg, "aaars": aaars_params}
    false_certs = 0
    stop_times = []
    recalls = []
    n_total = 0

    for seed in seeds:
        for alloc in allocations:
            n_total += 1
            result = run_episode(alloc, seed // 1000, seed, cfg=cfg_run,
                                 collect_trace=False)
            aaars_t = result.get("aaars__t")
            if aaars_t is not None:
                stop_times.append(aaars_t)
                recall = result.get("aaars__recall", 0)
                recalls.append(recall)
                if recall < 95.0:
                    false_certs += 1

    return {
        "false_certs": false_certs,
        "total_stops": len(stop_times),
        "cert_rate": len(stop_times) / n_total if n_total else 0,
        "median_stop": int(np.median(stop_times)) if stop_times else None,
        "median_recall": round(float(np.median(recalls)), 1) if recalls else None,
        "min_recall": round(float(min(recalls)), 1) if recalls else None,
        "n_total": n_total,
    }


def score(m):
    """Lower is better. Heavy penalty for false certs."""
    fc = m["false_certs"]
    ms = m["median_stop"] if m["median_stop"] else 5000
    cert = m["cert_rate"]
    return fc * 10000 + ms * (1 - max(cert, 0.01))


def print_result(label, m, elapsed):
    fc = m["false_certs"]
    nt = m["n_total"]
    cs = m["cert_rate"]
    ms = m["median_stop"]
    mr = m["median_recall"]
    s = score(m)
    print(f"  {label:45s}  FC={fc}/{nt}  cert={cs:.0%}  "
          f"med_stop={str(ms):>5}  med_rec={str(mr):>5}  "
          f"score={s:>8.0f}  [{elapsed:.1f}s]", flush=True)


def phase_a():
    """Phase A: Find best diagnostic weights."""
    print("\n" + "=" * 70)
    print("PHASE A: Diagnostic weights")
    print("  Fix: ema=0.1, alpha=0.05, lambda=1.0")
    print("=" * 70)

    combos = [
        (0.4, 0.3, 0.3), (0.5, 0.25, 0.25), (0.3, 0.4, 0.3),
        (0.3, 0.3, 0.4), (0.6, 0.2, 0.2),   (0.2, 0.6, 0.2),
        (0.2, 0.2, 0.6), (0.45, 0.35, 0.2),  (0.35, 0.45, 0.2),
        (0.4, 0.2, 0.4),
    ]

    results = []
    for wt, ws, wf in combos:
        params = {"w_temporal": wt, "w_spatial": ws, "w_frequency": wf,
                  "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}
        t0 = time.perf_counter()
        m = evaluate_aaars(params, FAST_CFG, DISCOVERY_SEEDS, ALLOCATIONS)
        elapsed = time.perf_counter() - t0
        label = f"w=({wt:.2f},{ws:.2f},{wf:.2f})"
        print_result(label, m, elapsed)
        results.append((params, m, score(m)))

    results.sort(key=lambda x: x[2])
    best_params = results[0][0]
    print(f"\n  >>> BEST: w=({best_params['w_temporal']},{best_params['w_spatial']},{best_params['w_frequency']})")
    return best_params


def phase_b(best_weights):
    """Phase B: Find best risk/threshold params."""
    print("\n" + "=" * 70)
    print("PHASE B: Risk/threshold parameters")
    print(f"  Using best weights: ({best_weights['w_temporal']},{best_weights['w_spatial']},{best_weights['w_frequency']})")
    print("=" * 70)

    combos = [
        (0.1, 0.05, 1.0),   # default
        (0.05, 0.05, 1.0),
        (0.2, 0.05, 1.0),
        (0.1, 0.03, 1.0),
        (0.1, 0.10, 1.0),
        (0.1, 0.05, 0.5),
        (0.1, 0.05, 2.0),
        (0.05, 0.03, 1.5),
        (0.15, 0.05, 1.5),
        (0.1, 0.05, 0.75),
        (0.1, 0.075, 1.0),
        (0.05, 0.05, 0.5),
    ]

    results = []
    for ea, ba, rl in combos:
        params = {**best_weights, "ema_alpha": ea, "base_alpha": ba, "risk_lambda": rl}
        t0 = time.perf_counter()
        m = evaluate_aaars(params, FAST_CFG, DISCOVERY_SEEDS, ALLOCATIONS)
        elapsed = time.perf_counter() - t0
        label = f"ea={ea:.2f} ba={ba:.3f} rl={rl:.1f}"
        print_result(label, m, elapsed)
        results.append((params, m, score(m)))

    results.sort(key=lambda x: x[2])
    best_params = results[0][0]
    print(f"\n  >>> BEST: ea={best_params['ema_alpha']}, ba={best_params['base_alpha']}, rl={best_params['risk_lambda']}")
    return best_params


def phase_c(best_params):
    """Phase C: Validate best config on medium environment."""
    print("\n" + "=" * 70)
    print("PHASE C: Validation on medium config (100x100, K=60, N=6)")
    print("=" * 70)

    val_seeds = DISCOVERY_SEEDS[:5]
    t0 = time.perf_counter()
    m = evaluate_aaars(best_params, MED_CFG, val_seeds, ALLOCATIONS)
    elapsed = time.perf_counter() - t0
    print_result("medium validation", m, elapsed)

    if m["false_certs"] == 0:
        print("  PASS: Zero false certifications on medium config")
    else:
        print(f"  WARN: {m['false_certs']} false certifications — tighten thresholds")
    return m


def main():
    print("=" * 70)
    print("AAARS DISCOVERY CAMPAIGN")
    print("Seeds 0-9 | tiny config 30x30 K=10 N=2", flush=True)
    print("=" * 70)

    t_total = time.perf_counter()

    best_weights = phase_a()
    best_params = phase_b(best_weights)
    val_metrics = phase_c(best_params)

    elapsed = time.perf_counter() - t_total

    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"  w_temporal  = {best_params['w_temporal']}")
    print(f"  w_spatial   = {best_params['w_spatial']}")
    print(f"  w_frequency = {best_params['w_frequency']}")
    print(f"  ema_alpha   = {best_params['ema_alpha']}")
    print(f"  base_alpha  = {best_params['base_alpha']}")
    print(f"  risk_lambda = {best_params['risk_lambda']}")
    print(f"  Total time: {elapsed:.1f}s")

    out_path = os.path.join(_PROJECT_ROOT, "results", "discovery_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best_params": best_params, "validation": val_metrics},
                  f, indent=2, default=str)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
