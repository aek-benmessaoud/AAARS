#!/usr/bin/env python
"""
concentration_test.py — advisor follow-up: do *spatial-concentration of visits*
statistics (normalized visit-entropy, top-10% share of hits) separate safe from
unsafe episodes for the saturated policies (greedy, hotspot), where delta_bar
and gamma2 (capture heterogeneity) both failed?

These are the advisor's "concentration spatiale / revisite" family. Derived from
the leak-free per-cell hit-count map at the AAARS stop (bits grid) — the same
map gamma2 used, but analysed for concentration (entropy / top-k share) rather
than heterogeneity.
"""
import os
import json

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0
NBINS = 8


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * (c - h), 100.0 * (c + h)


def spearman(x, y):
    import scipy.stats as st
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    return st.spearmanr(x, y)


def load():
    v2 = json.load(open(os.path.join(RAW, "gamma_policies_v2.json")))
    by = {(r["alloc"], r["run"]): r for r in v2}
    pol = json.load(open(os.path.join(RAW, "policies_results.json")))
    rows = {a: [] for a in ["frontier", "greedy", "hotspot"]}
    for a in rows:
        for rec in [r for r in pol if r["alloc"] == a]:
            cov = np.asarray(rec.get("coverage_series"), dtype=float)
            if cov.size == 0:
                continue
            stop = rec.get("aaars__t") or rec.get("T_total", cov.size)
            stop = max(1, min(int(stop), cov.size))
            delta = 1.0 - cov[:stop]
            g = by.get((a, rec["run"]), {})
            fc = int(rec["aaars__t"] is not None and rec["aaars__recall"] < RECALL_THR)
            rows[a].append({"run": rec["run"], "d_bar": float(delta.mean()),
                            "entropy": g.get("entropy"), "topk": g.get("topk"),
                            "gamma2": g.get("gamma2"), "fc": fc})
    return rows


def bins_for(features, scalar):
    vals = [f[scalar] for f in features]
    pairs = [(v, f["fc"]) for v, f in zip(vals, features) if v is not None and v == v]
    if len(pairs) < NBINS * 2:
        return None
    vv = np.array([p[0] for p in pairs]); fc = np.array([p[1] for p in pairs])
    edges = np.quantile(vv, np.linspace(0, 1, NBINS + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out = []
    for i in range(NBINS):
        m = (vv >= edges[i]) & (vv < edges[i + 1])
        if not m.any():
            continue
        k = int(fc[m].sum()); nn = int(m.sum())
        lo, hi = wilson_ci(k, nn)
        out.append((float(vv[m].mean()), 100.0 * k / nn, nn, lo, hi))
    return out


def report(features, a, scalars):
    fc_all = 100.0 * sum(f["fc"] for f in features) / len(features)
    print(f"\n== {a.upper()} (n_ep={len(features)}, overall FC={fc_all:.1f}%) ==")
    for s in scalars:
        b = bins_for(features, s)
        if b is None:
            print(f"   {s:9s}: skipped")
            continue
        rho, pv = spearman([x[0] for x in b], [x[1] for x in b])
        spread = b[-1][1] - b[0][1]
        rng = (min(x[0] for x in b), max(x[0] for x in b))
        flag = "DISCRIMINATES" if spread >= 40 else ("some" if spread >= 20 else "weak/~none")
        print(f"   {s:9s}: FC low={b[0][1]:5.1f}% high={b[-1][1]:5.1f}%  "
              f"spread={spread:+5.1f}pp  rho={rho:+.2f} (p={pv:.3f})  "
              f"{s} range=[{rng[0]:.4f},{rng[1]:.4f}]  -> {flag}")


def main():
    rows = load()
    scalars = ["entropy", "topk"]
    print("ADVISOR concentration test — spatial concentration of visits vs FC")
    print("(gamma2 = capture heterogeneity shown for reference)")
    for a in ["frontier", "greedy", "hotspot"]:
        report(rows[a], a, scalars + ["gamma2"])

    # Entropy is near-maximal overall -> how much variation actually exists?
    print("\n== dynamic-range diagnostic: is there ANY concentration signal? ==")
    for a in ["frontier", "greedy", "hotspot"]:
        e = [r["entropy"] for r in rows[a] if r["entropy"] is not None]
        t = [r["topk"] for r in rows[a] if r["topk"] is not None]
        print(f"   {a:9s}: entropy span={max(e)-min(e):.4f}  "
              f"topk span={max(t)-min(t):.4f}")


if __name__ == "__main__":
    main()
