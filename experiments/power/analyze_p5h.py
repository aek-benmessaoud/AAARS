#!/usr/bin/env python
"""P5-H analysis: mirror tab:confirm + paired McNemar per (regime x alloc).

Regimes:
  H0 = homogeneous (power_revision.json, 160 recs) [context baseline]
  H1 = bands_hetero (p5h_H1.json, 160 recs)
  H2 = bands_rich    (p5h_H2.json, 160 recs)   [revealed correlation]

Outputs, per regime x alloc:
  mirror table row: Stop, FC, FC% (among stoppers), Wilson 95% CI, Med t,
                    t bootstrap 95% CI, MedR, MinR   [for Chao1-CI, AAARS]
  paired McNemar: b (Chao-false/AAARS-safe), c (Chao-safe/AAARS-false),
                  n=b+c, exact two-sided p
  paired bootstrap 95% CI on diff rate (Chao1-AAARS), pp

Pre-registered checks:
  H2 advantage: AAARS FC < Chao1 FC under bands_rich, McNemar p, CI excl 0.
  H1 link: same test under bands_hetero (mechanism, no revealed correlation).
"""
import json
import os
import numpy as np
from scipy.stats import binom

ROOT = r"F:\Project11-AAARS"
RAW = os.path.join(ROOT, "results", "raw")
THR = 95.0
Z = 1.9599639845
RNG = np.random.default_rng(0)


def load(name):
    return json.load(open(os.path.join(RAW, name)))


def is_fc(r, m):
    return r.get(f"{m}__t") is not None and r[f"{m}__recall"] < THR


def wilson(k, n):
    if n == 0:
        return None, None
    zh = Z * Z / 2.0
    den = n + Z * Z
    c = (k + zh) / den
    h = Z / den * np.sqrt(max(0.0, k * (n - k) / n + Z * Z / 4.0))
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def med_ci(rs, m, keys=None):
    times = [r[f"{m}__t"] for r in rs if r.get(f"{m}__t") is not None]
    if not times:
        return None, (None, None)
    t = np.array(times, dtype=float)
    med = float(np.median(t))
    M = 20000
    boots = np.empty(M)
    for i in range(M):
        idx = RNG.integers(0, len(t), size=len(t))
        boots[i] = np.median(t[idx])
    return med, tuple(round(float(x), 0) for x in np.percentile(boots, [2.5, 97.5]))


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def paired_boot_ci(recs, M=20000):
    pairs = np.array([[is_fc(r, "chao1_ci"), is_fc(r, "aaars")]
                      for r in recs], dtype=int)
    N = len(pairs)
    diffs = np.empty(M)
    for i in range(M):
        idx = RNG.integers(0, N, size=N)
        ch = pairs[idx, 0].sum()
        aa = pairs[idx, 1].sum()
        diffs[i] = (ch - aa) / N * 100.0
    return np.percentile(diffs, [2.5, 97.5]), diffs.mean()


def block(regime, recs, alloc, label, methods=("chao1_ci", "aaars")):
    sub = [r for r in recs if r.get("alloc") == alloc]
    print(f"\n=== {label} [{regime}] alloc={alloc} N={len(sub)} ===")
    rows = {}
    for m in methods:
        stop = sum(1 for r in sub if r.get(f"{m}__t") is not None)
        fc = sum(is_fc(r, m) for r in sub)
        medR = float(np.median([r.get(f"{m}__recall") or 0 for r in sub]))
        minR = min((r.get(f"{m}__recall") for r in sub), default=0)
        medt, (lo, hi) = med_ci(sub, m)
        lo_w, hi_w = wilson(fc, max(stop, 1))
        ci_s = f"$[{lo_w:.1f},{hi_w:.1f}]$" if lo_w is not None else "--"
        tci_s = f"$[{lo:.0f},{hi:.0f}]$" if lo is not None else "--"
        print(f"  {m:11s} stop={stop}/{len(sub)}  FC={fc}  "
              f"FC%={100*fc/max(stop,1):.1f}  Wilson95={ci_s}  "
              f"Medt={medt}  tCI={tci_s}  MedR={medR:.1f}  MinR={minR:.1f}")
        rows[m] = dict(stop=stop, fc=fc, medt=medt)
    a = b = c = d = 0
    for r in sub:
        fc_c = is_fc(r, "chao1_ci")
        fc_a = is_fc(r, "aaars")
        if not fc_c and not fc_a:
            a += 1
        elif fc_c and not fc_a:
            b += 1
        elif not fc_c and fc_a:
            c += 1
        else:
            d += 1
    pv = mcnemar_p(b, c)
    (clo, chi), md = paired_boot_ci(sub)
    print(f"  McNemar 2x2: a={a} b={b} c={c} d={d}  b+c={b+c}  p={pv:.4f}")
    print(f"  Paired bootstrap 95% CI [Chao1-AAARS diff rate, pp]: "
          f"[{clo:.1f},{chi:.1f}]  mean={md:.2f}")
    return dict(regime=regime, alloc=alloc, N=len(sub), rows=rows,
                mcnemar=dict(b=b, c=c, p=pv), ci=(clo, chi))


def main():
    data = {
        "H0": load("power_revision.json"),
        "H1": load("p5h_H1.json"),
        "H2": load("p5h_H2.json"),
    }
    summary = {}
    for regime, recs in data.items():
        loc = "homogeneous" if regime == "H0" else \
              ("bands_hetero" if regime == "H1" else "bands_rich")
        dets = {r.get("detectability") for r in recs}
        print(f"regime {regime}: detectability={sorted(dets)}  n={len(recs)}")
        summary[regime] = {}
        for alloc in ("boustro", "minerich"):
            res = block(regime, recs, alloc, f"{regime}-{alloc}")
            summary[regime][alloc] = res

    print("\n\n===== PRE-REGISTERED HYPOTHESIS CHECKS =====")
    for alloc in ("boustro", "minerich"):
        h2 = summary["H2"][alloc]
        h1 = summary["H1"][alloc]
        ch2 = h2["rows"]["chao1_ci"]["fc"] / max(h2["rows"]["chao1_ci"]["stop"], 1)
        ah2 = h2["rows"]["aaars"]["fc"] / max(h2["rows"]["aaars"]["stop"], 1)
        ch1 = h1["rows"]["chao1_ci"]["fc"] / max(h1["rows"]["chao1_ci"]["stop"], 1)
        ah1 = h1["rows"]["aaars"]["fc"] / max(h1["rows"]["aaars"]["stop"], 1)
        print(f"\nH2 advantage [{alloc}]: Chao1 {100*ch2:.1f}% -> AAARS {100*ah2:.1f}%  "
              f"McNemar p={h2['mcnemar']['p']:.4f}  "
              f"boot95CI={tuple(round(x,1) for x in h2['ci'])}")
        print(f"H1 mechanism [{alloc}]: Chao1 {100*ch1:.1f}% -> AAARS {100*ah1:.1f}%  "
              f"McNemar p={h1['mcnemar']['p']:.4f}  "
              f"boot95CI={tuple(round(x,1) for x in h1['ci'])}")


if __name__ == "__main__":
    main()