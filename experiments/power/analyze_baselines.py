#!/usr/bin/env python
"""
analyze_baselines.py — FC / recall / stop-time summary for the P8 baselines
(rate_cs coverage-aware silent-run rule, gap_sprt Wald SPRT on discovery gap)
vs the existing stopping rules, from the merged power_baselines.json run.

Same conventions as analyze_policies.py / power_confirm.py: a method
"false-certifies" (FC) on an episode if it stops but returns recall < 95%.
"""
import os
import json
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RAW = os.path.join(_PROJECT_ROOT, "results", "raw")

RECALL_THR = 95.0
ALLOCS = ["boustro", "minerich"]
METHODS = ["chao1_ci", "coverage_only", "rate_cs", "gap_sprt",
           "aaars", "oracle_95", "fixed_2", "diminishing"]


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


def report(subset, label):
    n = len(subset)
    print(f"\n--- {label} ({n} episodes) ---")
    for method in METHODS:
        stops = [r for r in subset if r.get(f"{method}__t") is not None]
        ns = len(stops)
        if ns == 0:
            print(f"  {method:15s}: 0/{n} stops")
            continue
        fc = sum(1 for r in stops if r[f"{method}__recall"] < RECALL_THR)
        lo, hi = wilson_ci(fc, n)
        mr = mean_recall(stops, method)
        med_t = float(np.median([r[f"{method}__t"] for r in stops]))
        print(f"  {method:15s}: {ns:>3d}/{n} stops, FC={fc} "
              f"({100.0*fc/n:5.1f}% [{lo:5.1f},{hi:5.1f}]) "
              f"med_t={med_t:6.0f} meanRecall={mr:6.1f}")


def paired(subset, m1, m2, label):
    both = [r for r in subset if r.get(f"{m1}__t") is not None
            and r.get(f"{m2}__t") is not None]
    if not both:
        print(f"  paired {m1} vs {m2}: no jointly-stopped episodes")
        return
    na = len(both)
    a = sum(1 for r in both if r[f"{m1}__recall"] < RECALL_THR)
    c = sum(1 for r in both if r[f"{m2}__recall"] < RECALL_THR)
    indel_both = sum(1 for r in both
                     if r[f"{m1}__recall"] >= RECALL_THR
                     and r[f"{m2}__recall"] >= RECALL_THR)
    only1 = sum(1 for r in both
                if r[f"{m1}__recall"] < RECALL_THR
                and r[f"{m2}__recall"] >= RECALL_THR)
    only2 = sum(1 for r in both
                if r[f"{m1}__recall"] >= RECALL_THR
                and r[f"{m2}__recall"] < RECALL_THR)
    p = fisher_exact(only1, only2, na - only1 - only2, 0) if (only1 + only2) else 1.0
    print(f"  [{label}] {m1} FC {a}/{na} vs {m2} FC {c}/{na}; "
          f"discordant {m1}-only={only1}, {m2}-only={only2}; "
          f"Fisher p={p:.4f}")
    dt = np.mean([r[f"{m2}__t"] - r[f"{m1}__t"] for r in both])
    print(f"  mean t({m2})-t({m1}) on jointly-stopped: {dt:+.1f} steps")


def main():
    path = os.path.join(RAW, "power_baselines.json")
    with open(path) as f:
        results = json.load(f)
    print(f"loaded {len(results)} records from {path}")

    print("=" * 80)
    print("P8 BASELINES — FC TABLE (recall < 95% => false-cert), n=80 per alloc")
    print("=" * 80)

    for alloc in ALLOCS:
        subset = [r for r in results if r.get("alloc") == alloc]
        report(subset, f"{alloc.upper()}")
        paired(subset, "chao1_ci", "rate_cs", "Chao1 vs rate_cs")
        paired(subset, "chao1_ci", "gap_sprt", "Chao1 vs gap_sprt")
        paired(subset, "aaars", "rate_cs", "AAARS vs rate_cs")
        paired(subset, "aaars", "gap_sprt", "AAARS vs gap_sprt")

    print("\n" + "=" * 80)
    print("POOLED (160 episodes) — headline")
    print("=" * 80)
    report(results, "ALL")


if __name__ == "__main__":
    main()
