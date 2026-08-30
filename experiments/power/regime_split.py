#!/usr/bin/env python
"""
regime_split.py — advisor Step-0/Step-2 re-run, split by regime (no new sims).

The advisor observed two regimes:
  * clean  : MineRichness, Frontier  -> FC rises 0%->40-60% with mean deficit
            (delta_bar dominates cleanly) -> candidate for a conditional bound.
  * saturated: Greedy, Hotspot      -> FC already 70-90% even at low delta_bar
            (deficit explains ~nothing; risk saturated a priori).

This re-derives the same deltas (A3: delta = 1 - coverage) and bins the empirical
AAARS false-certify (FC = recall<95%, H1) rate per allocation, then *groups* the
curves so the clean signal is not diluted by the saturated one. Also reports the
per-allocation "total-D-fraction stalls" (Step-0 premise) and, where available,
a revisit-concentration proxy n_prime/n_cells from d1_prime_minerich.json.
"""
import os
import json

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0
WINDOW_FRACS = [0.05, 0.10, 0.20]
NBINS = 8
CLEAN = ["minerich", "frontier"]
SATURATED = ["greedy", "hotspot"]
COHORTS = {"minerich": "power_revision.json", "boustro": "power_revision.json",
           "frontier": "policies_results.json", "greedy": "policies_results.json",
           "hotspot": "policies_results.json"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100.0 * (c - h), 100.0 * (c + h))


def load():
    recs = {}
    for alloc, fn in COHORTS.items():
        for r in json.load(open(os.path.join(RAW, fn))):
            if r["alloc"] == alloc:
                recs.setdefault(alloc, []).append(r)
    return recs


def deficit_up_to_stop(r, method="aaars"):
    cov = np.asarray(r.get("coverage_series"), dtype=float)
    if cov.size == 0:
        return np.array([]), 0
    stop = r.get(f"{method}__t") or r.get("T_total", cov.size)
    stop = max(1, min(int(stop), cov.size))
    return 1.0 - cov[:stop], stop


def d_star(delta, stop, frac):
    w = max(1, int(round(frac * stop)))
    if delta.size < w:
        w = delta.size
    if w == 0:
        return float("nan")
    cs = np.cumsum(np.concatenate(([0.0], delta)))
    wm = (cs[w:] - cs[:-w]) / w
    return float(np.max(wm)) if wm.size else float("nan")


def is_fc(r, method="aaars"):
    return r.get(f"{method}__t") is not None and r[f"{method}__recall"] < RECALL_THR


def step0(rows, alloc):
    stalls_high = []
    trends = []
    for r in rows:
        delta, stop = deficit_up_to_stop(r)
        if delta.size < 20:
            continue
        w = max(1, delta.size // 10)
        trends.append(float(np.mean(delta[-w:]) - np.mean(delta[:w])))
        tail = delta[-max(1, delta.size // 5):]
        stalls_high.append(float(np.max(tail)) > 0.5)
    n = len(trends) or 1
    print(f"[Step0] {alloc:9s} n={len(rows):3d}  "
          f"stall>0.5 last-5th={100.0*np.mean(stalls_high):.3f}%  "
          f"mean delta(last-first)={np.mean(trends):+.3f}")


def bins_for(alloc, scalar, features):
    vals = [f[scalar] for f in features]
    pairs = [(v, f["fc"]) for v, f in zip(vals, features) if v == v]
    if len(pairs) < NBINS * 2:
        return None
    vv = np.array([p[0] for p in pairs]); fc = np.array([p[1] for p in pairs])
    edges = np.quantile(vv, np.linspace(0, 1, NBINS + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    rowsx = []
    for i in range(NBINS):
        m = (vv >= edges[i]) & (vv < edges[i + 1])
        if not m.any():
            continue
        k = int(fc[m].sum()); nn = int(m.sum())
        lo, hi = wilson_ci(k, nn)
        rowsx.append((float(vv[m].mean()), 100.0 * k / nn, nn, lo, hi))
    return rowsx


def spearman(x, y):
    import scipy.stats as st
    if len(x) < 3:
        return float("nan"), float("nan")
    return st.spearmanr(x, y)


def step2_regime(features, alloc, label, group):
    print(f"\n[{group}] {label} (n={len(features)})")
    for scalar in ["d_bar", "d_star_0.10"]:
        rows = bins_for(alloc, scalar, features)
        if not rows:
            print(f"   {scalar:12s}: skipped (too few)")
            continue
        rho, pv = spearman([r[0] for r in rows], [r[1] for r in rows])
        fcrange = (rows[0][1], rows[-1][1])
        print(f"   {scalar:12s}: FC low-bin={fcrange[0]:.1f}%  "
              f"high-bin={fcrange[1]:.1f}%  rho={rho:+.2f} (p={pv:.3f})")
        for r in rows:
            print(f"      d={r[0]:.3f} n={r[2]:3d} FC={r[1]:5.1f}% "
                  f"[{r[3]:.1f},{r[4]:.1f}]")


def main():
    recs = load()
    # optional revisit-concentration for minerich from d1_prime_minerich.json
    revisit = {}
    dp = os.path.join(RAW, "d1_prime_minerich.json")
    if os.path.exists(dp):
        for r in json.load(open(dp)):
            revisit[r["run"]] = (r["n_prime"] / r["n_cells"]) if r["n_cells"] else None
    d1runs = set(revisit)

    print("=" * 72)
    print("STEP 0 — premise falsification per allocation")
    print("=" * 72)
    for a in list(COHORTS):
        step0(recs[a], a)

    print("\n" + "=" * 72)
    print("STEP 2 — AAARS FC rate vs deficit scalar, split by regime")
    print("=" * 72)
    features = {}
    for a in list(COHORTS):
        feats = []
        for r in recs[a]:
            delta, stop = deficit_up_to_stop(r)
            if delta.size == 0:
                continue
            feats.append({
                "d_bar": float(np.mean(delta)),
                "d_star_0.10": d_star(delta, stop, 0.10),
                "fc": int(is_fc(r)),
                "revisit": revisit.get(r["run"]),
            })
        features[a] = feats

    for a in CLEAN:
        step2_regime(features[a], a, {a: a.upper()}[a], "CLEAN group (delta_bar-dominant)")
    for a in SATURATED:
        step2_regime(features[a], a, a.upper(), "SATURATED group (deficit-explains-little)")
    step2_regime(features["boustro"], "boustro", "BOUSTRO (FC~0 reference)",
                 "REFERENCE")

    print("\n" + "=" * 72)
    print("ADVISOR CHECK — does the split confirm the saturated vs clean gap?")
    print("=" * 72)
    for a in CLEAN + SATURATED:
        fb = bins_for(a, "d_bar", features[a])
        if not fb:
            continue
        lo = fb[0][1]; hi = fb[-1][1]
        tone = "clean (delta_bar discriminates)" if (hi - lo) >= 25 else \
               ("saturated (delta_bar ~flat)" if (hi - lo) <= 15 else "intermediate")
        print(f"   {a:9s}: low-bin FC={lo:.1f}% high-bin FC={hi:.1f}%  spread={hi-lo:+.1f}pp  -> {tone}")

    # revisit proxy report (minerich only for now)
    if d1runs:
        prov = [revisit[r] for r in d1runs if revisit[r]]
        fcs = {r: features['minerich'][i]['fc']
               for i, r in enumerate([x['run'] for x in recs['minerich']])}
        print("\n[revisit proxy, minerich only]")
        print(f"   n_prime/n_cells (avg revisits per scanned cell) "
              f"med={np.median(prov):.2f} min={min(prov):.2f} max={max(prov):.2f}")


if __name__ == "__main__":
    main()
