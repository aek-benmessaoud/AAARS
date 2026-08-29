#!/usr/bin/env python
"""
analyze_policies.py — FC / recall / CI summary for the realistic biased
allocation policies (frontier, greedy, hotspot) vs the ground-truth boustro
baseline, reusing the same conventions as power_confirm.py.

A method "false-certifies" (FC) on an episode if it stops (has a stop time)
but returns recall < 95% (coverage threshold = 0.95).
"""
import os
import json
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RAW = os.path.join(_PROJECT_ROOT, "results", "raw")

POLICIES = ["frontier", "greedy", "hotspot"]
RECALL_THR = 95.0
METHODS = ["chao1_ci", "aaars", "discrete_aaars", "threshold_aaars",
           "chao92_ci", "oracle_95", "fixed_2", "diminishing"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def fisher_exact(a, b, c, d):
    from scipy.stats import fisher_exact as fe
    _, p = fe([[a, b], [c, d]])
    return p


def mean_recall(stops, method):
    vals = [r[f"{method}__recall"] for r in stops
            if r[f"{method}__recall"] is not None]
    return np.mean(vals) if vals else float("nan")


def main():
    path = os.path.join(RAW, "policies_results.json")
    with open(path) as f:
        results = json.load(f)
    print(f"loaded {len(results)} records from {path}")

    print("=" * 78)
    print("REALISTIC BIASED POLICIES — FC TABLE (recall<95% => false-cert)")
    print("=" * 78)

    for pol in POLICIES:
        subset = [r for r in results if r.get("alloc") == pol]
        n = len(subset)
        print(f"\n--- {pol.upper()} ({n} episodes) ---")
        for method in METHODS:
            stops = [r for r in subset if r.get(f"{method}__t") is not None]
            ns = len(stops)
            if ns == 0:
                print(f"  {method:16s}: 0/{n} stops")
                continue
            fc = sum(1 for r in stops if r[f"{method}__recall"] < RECALL_THR)
            lo, hi = wilson_ci(fc, n)
            mr = mean_recall(stops, method)
            times = [r[f"{method}__t"] for r in stops]
            med_t = np.median(times)
            over = sum(1 for r in stops if r.get("aaars__t") is not None and
                       r.get("chao1_ci__t") is not None)
            print(f"  {method:16s}: {ns:>3d}/{n} stops, FC={fc} "
                  f"({100.0*fc/n:5.1f}% [{lo:5.1f},{hi:5.1f}]) "
                  f"med_t={med_t:6.0f} meanRecall={mr:5.1f}")

        # Fisher exact AAARS vs Chao1 on FC counts (episodes both stopped)
        both = [r for r in subset if r.get("aaars__t") is not None
                and r.get("chao1_ci__t") is not None]
        if both:
            ma = sum(1 for r in both if r["aaars__recall"] < RECALL_THR)
            mch = sum(1 for r in both if r["chao1_ci__recall"] < RECALL_THR)
            na = len(both)
            p = fisher_exact(ma, na - ma, mch, na - mch)
            print(f"  PAIRED FISHER (both-stop subset n={na}): "
                  f"AAARS FA {ma}/{na} vs Chao1 FA {mch}/{na}  p={p:.4f}")

        # mean time benefit on episodes where both stop
        bt = [r for r in subset if r.get("aaars__t") is not None
              and r.get("chao1_ci__t") is not None]
        if bt:
            dt = np.mean([r["chao1_ci__t"] - r["aaars__t"] for r in bt])
            print(f"  mean t(Cao1)-t(AAARS) on jointly-stopped: {dt:+.1f} steps")

        # coverage concentration / bias indicator: how far final coverage from
        # full at the time each method stops
        for method in ("chao1_ci", "aaars"):
            sc = [r.get("coverage_series", []) for r in subset]
            break


if __name__ == "__main__":
    main()
