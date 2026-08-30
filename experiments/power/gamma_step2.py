#!/usr/bin/env python
"""
gamma_step2.py — advisor follow-up: is spatial concentration (gamma2 / revisit)
the real driver of FC for the "saturated" policies (greedy, hotspot) where mean
deficit delta_bar fails to discriminate?

For each of the 3 realistic policies (from gamma_policies.json + the committed
policies_results.json) we bin the empirical AAARS FC rate against:
  * delta_bar (time-mean deficit, from committed coverage_series) -> should fail
    for greedy/hotspot (saturated)
  * gamma2  (squared CV of per-cell hit counts, leak-free spatial concentration)
  * revisit (n_prime / n_cells, mean revisits per scanned cell)
and report monotonicity (Spearman rho) + the low-bin vs high-bin FC spread, to
see which scalar cleanly predicts FC for each policy.
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
    return (100.0 * (c - h), 100.0 * (c + h))


def spearman(x, y):
    import scipy.stats as st
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    return st.spearmanr(x, y)


def load():
    gamma = json.load(open(os.path.join(RAW, "gamma_policies.json")))
    pol = json.load(open(os.path.join(RAW, "policies_results.json")))
    g_by = {(r["alloc"], r["run"]): r for r in gamma}
    rows = {a: [] for a in ["frontier", "greedy", "hotspot"]}
    for a in rows:
        for rec in [r for r in pol if r["alloc"] == a]:
            cov = np.asarray(rec.get("coverage_series"), dtype=float)
            if cov.size == 0:
                continue
            stop = rec.get("aaars__t") or rec.get("T_total", cov.size)
            stop = max(1, min(int(stop), cov.size))
            delta = 1.0 - cov[:stop]
            g = g_by.get((a, rec["run"]), {})
            fc = int(rec["aaars__t"] is not None and rec["aaars__recall"] < RECALL_THR)
            rows[a].append({
                "run": rec["run"],
                "d_bar": float(delta.mean()),
                "gamma2": g.get("gamma2"),
                "revisit": g.get("revisit"),
                "n_cells": g.get("n_cells"),
                "fc": fc,
            })
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


def report(features, a, scalars=("d_bar", "gamma2", "revisit")):
    print(f"\n== {a.upper()} (n_ep={len(features)}) ==")
    fc_all = 100.0 * sum(f["fc"] for f in features) / len(features)
    print(f"   overall FC = {fc_all:.1f}%")
    for s in scalars:
        b = bins_for(features, s)
        if b is None:
            print(f"   {s:9s}: skipped")
            continue
        rho, pv = spearman([x[0] for x in b], [x[1] for x in b])
        spread = b[-1][1] - b[0][1]
        rng = (min(x[0] for x in b), max(x[0] for x in b))
        dflag = "DOMINATES" if spread >= 40 else ("some" if spread >= 20 else "weak")
        print(f"   {s:9s}: FC low={b[0][1]:5.1f}% high={b[-1][1]:5.1f}%  "
              f"spread={spread:+5.1f}pp  rho={rho:+.2f} (p={pv:.3f})  "
              f"{s} range=[{rng[0]:.3f},{rng[1]:.3f}]  -> {dflag}")
        # show first 2 + last 2 bins compactly
        show = b[:2] + ["..."] + b[-2:]
        for x in show:
            if x == "...":
                print("          ...")
            else:
                print(f"          {s}={x[0]:.4f} n={x[2]:3d} FC={x[1]:5.1f}% "
                      f"[{x[3]:.1f},{x[4]:.1f}]")


def main():
    rows = load()
    print("ADVISOR STEP-2 follow-up — which scalar drives FC per policy?")
    for a in ["frontier", "greedy", "hotspot"]:
        report(rows[a], a)

    # residual: does delta_bar still fail for saturated policies even at low delta?
    print("\n== delta_bar low-bin FC by policy (is greedy/hotspot saturated?) ==")
    for a in ["frontier", "greedy", "hotspot"]:
        b = bins_for(rows[a], "d_bar")
        if b:
            print(f"   {a:9s}: d_bar low-bin FC = {b[0][1]:.1f}% "
                  f"(d_bar low {b[0][0]:.3f}); high-bin {b[-1][1]:.1f}%")


if __name__ == "__main__":
    main()
