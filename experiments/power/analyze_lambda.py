#!/usr/bin/env python
"""analyze_lambda.py — FC rate + median stop time vs risk_lambda."""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "results", "raw")

THR = 95.0
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0]
ALLOCS = ["boustro", "minerich"]


def wilson_ci(k, n):
    from numpy import sqrt
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def main():
    with open(os.path.join(RAW, "lambda_sweep.json")) as f:
        recs = json.load(f)
    print(f"loaded {len(recs)} records\n")

    for alloc in ALLOCS:
        sub = [r for r in recs if r["alloc"] == alloc]
        print(f"=== {alloc.upper()} ===  (AAARS; Chao1 reference FC "
              f"= {sum(1 for r in sub if r.get('chao1_ci__recall',100)<THR)}/80)")
        print(f"{'lam':>5s} {'AAARS FC':>8s} {'FC%':>7s} {'95% CI':>16s} "
              f"{'med_t':>6s} {'meanR':>6s} {'stops':>6s}")
        for lam in LAMS:
            xs = [r for r in sub if r.get("lam") == lam]
            n = len(xs)
            stops = [r for r in xs if r.get("aaars__t") is not None]
            ns = len(stops)
            fc = sum(1 for r in stops if r["aaars__recall"] < THR)
            lo, hi = wilson_ci(fc, n)
            times = [r["aaars__t"] for r in stops]
            med_t = np.median(times) if times else float("nan")
            mr = np.mean([r["aaars__recall"] for r in xs])
            print(f"{lam:>5.1f} {fc:>8d} {100*fc/n:>6.1f}% "
                  f"[{lo:>5.1f},{hi:>5.1f}] {med_t:>6.0f} {mr:>6.1f} "
                  f"{ns:>4d}/{n:<2d}")
        # Chao1 reference per lam should be identical across lam (same trace)
        print()


if __name__ == "__main__":
    main()
