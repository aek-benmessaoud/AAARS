#!/usr/bin/env python
"""Item-6 robustness analysis (obstacles + comm-delay).

Pre-registered rules (advisor 4b, mirroring H2):
  - power threshold: if a cell has b < 5 discordant (informative) pairs, report
    the McNemar p as "loss of power, not vanished/reversed effect", never as
    "effect disappeared".
  - primary test: AAARS < Chao1 (paired McNemar) per cell, asserting c == 0
    (never backfires).
Outputs per (lever-level x alloc) mirroring tab:gen.
"""
import json, os
import numpy as np
from scipy.stats import binom

ROOT = r"F:\Project11-AAARS"
RAW = os.path.join(ROOT, "results", "raw")
THR = 95.0
Z = 1.9599639845
RNG = np.random.default_rng(0)


def load(f): return json.load(open(os.path.join(RAW, f)))


def is_fc(r, m):
    return r.get(f"{m}__t") is not None and r[f"{m}__recall"] < THR


def wilson(k, n):
    if n == 0: return (None, None)
    zh = Z * Z / 2; den = n + Z * Z; c = (k + zh) / den
    h = Z / den * np.sqrt(max(0.0, k * (n - k) / n + Z * Z / 4))
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def med_ci(rs, m):
    t = [r[f"{m}__t"] for r in rs if r.get(f"{m}__t") is not None]
    if not t: return None
    t = np.array(t, dtype=float); med = float(np.median(t)); M = 20000
    b = np.empty(M)
    for i in range(M):
        idx = RNG.integers(0, len(t), size=len(t)); b[i] = np.median(t[idx])
    return med, tuple(round(float(v), 0) for v in np.percentile(b, [2.5, 97.5]))


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def pbc(rs, M=20000):
    pairs = np.array([[is_fc(r, "chao1_ci"), is_fc(r, "aaars")] for r in rs],
                     dtype=int); N = len(pairs); d = np.empty(M)
    for i in range(M):
        idx = RNG.integers(0, N, size=N)
        d[i] = (pairs[idx, 0].sum() - pairs[idx, 1].sum()) / N * 100
    return np.percentile(d, [2.5, 97.5]), d.mean()


def block(label, rs):
    print(f"\n=== {label}  N={len(rs)} ===")
    out = {}
    for m in ("chao1_ci", "aaars"):
        stop = sum(1 for r in rs if r.get(f"{m}__t") is not None)
        fc = sum(is_fc(r, m) for r in rs)
        lo, hi = wilson(fc, max(stop, 1))
        medr = float(np.median([r.get(f"{m}__recall") or 0 for r in rs]))
        minr = min((r.get(f"{m}__recall") for r in rs), default=0)
        mt = med_ci(rs, m)
        md, tci = (mt[0], mt[1]) if mt else (None, ("--", "--"))
        ci = f"[{lo:.1f},{hi:.1f}]" if lo is not None else "--"
        print(f"  {m:10s} stop={stop}/{len(rs)} FC={fc} FC%={100*fc/max(stop,1):.1f} "
              f"Wilson={ci} Medt={md} tCI={tci} MedR={medr:.1f} MinR={minr:.1f}")
        out[m] = dict(stop=stop, fc=fc, medt=md, medr=round(medr, 1),
                      minr=round(minr, 1))
    a = b = c = d = 0
    for r in rs:
        fc_c, fc_a = is_fc(r, "chao1_ci"), is_fc(r, "aaars")
        if not fc_c and not fc_a: a += 1
        elif fc_c and not fc_a: b += 1
        elif not fc_c and fc_a: c += 1
        else: d += 1
    p = mcnemar_p(b, c); (clo, chi), mdp = pbc(rs)
    how = ("SIGNIFICANT" if p < 0.05 else
           "POWER-LOSS (<5 discordant pairs: report as loss of power, "
           "not vanished effect)" if b < 5 else
           "not significant (b>=5)")
    print(f"  2x2: a={a} b={b} c={c} d={d}  McNemar p={p:.4f} "
          f"[{how}]  CI(pp)=[{clo:.1f},{chi:.1f}] mean={mdp:.2f}")
    print(f"  c==0 (never backfires): {c == 0}")
    out["mcnemar"] = dict(a=a, b=b, c=c, d=d, p=p, ci=(clo, chi))
    return out


def main():
    obs = load("p5h_obstacles.json"); cdl = load("p5h_commdelay.json")
    res = {}
    print("========= OBSTACLES (base = ratio 0.05 H0 MineRich 35.0->21.2) =========")
    for ratio in (0.10, 0.20):
        for alloc in ("boustro", "minerich"):
            cell = [r for r in obs if r["obstacle_ratio"] == ratio
                    and r["alloc"] == alloc]
            res[f"obs_{ratio}_{alloc}"] = block(f"obstacles ratio={ratio} alloc={alloc}", cell)
    print("\n========= COMM-DELAY (base = delay 0 MineRich 35.0->21.2) =========")
    for dly in (2, 4):
        for alloc in ("boustro", "minerich"):
            cell = [r for r in cdl if r["comm_delay"] == dly
                    and r["alloc"] == alloc]
            res[f"dly_{dly}_{alloc}"] = block(f"comm-delay={dly} alloc={alloc}", cell)

    print("\n===== SUMMARY vs BASE (MineRich FC%: Chao1->AAARS) =====")
    base = (35.0, 21.2)
    for key, alloc in [(k, k.rsplit("_", 1)[1]) for k in res if k.endswith("_minerich")]:
        m = res[key]
        fc1 = 100 * m["chao1_ci"]["fc"] / max(m["chao1_ci"]["stop"], 1)
        fa = 100 * m["aaars"]["fc"] / max(m["aaars"]["stop"], 1)
        p = m["mcnemar"]["p"]; b = m["mcnemar"]["b"]; c = m["mcnemar"]["c"]
        ci = m["mcnemar"]["ci"]
        print(f"  {key:22s} Chao1 {fc1:5.1f}% -> AAARS {fa:5.1f}%  "
              f"McNemar(b={b},c={c}) p={p:.3f}  CI(pp)=[{ci[0]:.1f},{ci[1]:.1f}]")


if __name__ == "__main__":
    main()