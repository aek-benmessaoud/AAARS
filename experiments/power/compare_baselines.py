#!/usr/bin/env python
"""Good-vs-baseline comparison from power_baselines.json (P8)."""
import os
import json
import numpy as np
from scipy.stats import fisher_exact

ROOT = r"F:\Project11-AAARS"
RAW = os.path.join(ROOT, "results", "raw")
THR = 95.0


def wilson(k, n):
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def stat(sub, m):
    key_t, key_r = m + "__t", m + "__recall"
    st = [x for x in sub if x.get(key_t) is not None]
    fc = sum(1 for x in st if x[key_r] < THR)
    lo, hi = wilson(fc, len(sub))
    med = float(np.median([x[key_t] for x in st])) if st else float("nan")
    mr = float(np.mean([x[key_r] for x in st])) if st else float("nan")
    return len(st), fc, 100.0 * fc / len(sub), lo, hi, med, mr


METHODS = ["aaars", "chao1_ci", "rate_cs", "gap_sprt",
           "coverage_only", "diminishing", "oracle_95", "fixed_2"]


def compare(m1, m2, sub, label):
    k1t, k1r = m1 + "__t", m1 + "__recall"
    k2t, k2r = m2 + "__t", m2 + "__recall"
    both = [x for x in sub if x.get(k1t) is not None and x.get(k2t) is not None]
    if not both:
        print(f"  [{label}] no jointly-stopped")
        return
    only1 = sum(1 for x in both if x[k1r] < THR and x[k2r] >= THR)
    only2 = sum(1 for x in both if x[k1r] >= THR and x[k2r] < THR)
    bothF = sum(1 for x in both if x[k1r] < THR and x[k2r] < THR)
    bothS = sum(1 for x in both if x[k1r] >= THR and x[k2r] >= THR)
    _, p = fisher_exact([[only1, only2], [bothF, bothS]])
    dt = float(np.mean([x[k2t] - x[k1t] for x in both]))
    print(f"  [{label}] {m1} vs {m2} (n={len(both)}): {m1}-FC {only1+bothF}, "
          f"{m2}-FC {only2+bothF}; discordant {m1}-better={only1}, {m2}-better={only2}; "
          f"exact p={p:.4f}; mean t({m2})-t({m1})={dt:+.1f}")


def main():
    r = json.load(open(os.path.join(RAW, "power_baselines.json")))
    print(f"loaded {len(r)} records")
    groups = [("BOUSTRO", [x for x in r if x["alloc"] == "boustro"]),
              ("MINERICH", [x for x in r if x["alloc"] == "minerich"]),
              ("ALL (160)", r)]
    for label, sub in groups:
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        for m in METHODS:
            s, fc, pc, lo, hi, med, mr = stat(sub, m)
            print(f"  {m:14s} stops={s:3d} FC={fc:3d} ({pc:5.1f}% "
                  f"[{lo:5.1f},{hi:5.1f}]) med_t={med:6.0f} meanRec={mr:5.1f}")
        compare("aaars", "chao1_ci", sub, label)
        compare("aaars", "rate_cs", sub, label)
        compare("aaars", "gap_sprt", sub, label)
        compare("aaars", "coverage_only", sub, label)


if __name__ == "__main__":
    main()
