#!/usr/bin/env python
"""
adversarial_stress.py — Phase-1 empirical stress test for a conditional
coverage-deficit bound (advisor's Steps 0-2).

Goal (before any Proposition): determine cheaply, on existing data (0 new
sims), whether a bound of the form
    P[AAARS false-certifies (FC) | adversary cannot sustain deficit > d_max] <= g(...)
is even directionally true. If the premise that adversarial policies get
"squeezed toward lower deficit" is false, the conditional framing collapses.

Locked decisions (advisor-approved):
  A3  delta(t) = 1 - fleet_coverage(t); mine re-sims only if ambiguous.
  C1  AAARS is the bounded rule; Chao1 is kept as reference curve only.
  E1  probability space = environment seeds, config fixed (lam fixed).
  G1  d_star windows expressed RELATIVE to episode stop time (not absolute).
  H1  FC = stop with recall < 95%.

Phase-1 output:
  Step 0  falsify the "eventually squeezed" premise: trajectory of delta(t),
          per-allocation share of episodes that keep high deficit to the end.
  Step 1  per-episode scalars: d_bar (time-mean), d_star(w) (max sustained
          deficit over a trailing window of length w = frac * stop time).
  Step 2  bin episodes by each scalar; empirical AAARS FC rate per bin with
          Wilson CI; check monotonicity (with Chao1 as reference curve).

No new simulation is required: everything derives from the committed
per-episode 'coverage_series' and the per-rule stop/recall fields.
"""
import os
import json
import csv

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")

RECALL_THR = 95.0
COHORT = os.path.join(RAW, "power_revision.json")
POLICIES = os.path.join(RAW, "policies_results.json")
# Relative trailing-window fractions for d_star (G1: express w in units of the
# episode stop time so window scales are comparable across policies).
WINDOW_FRACS = [0.05, 0.10, 0.20]
NBINS = 8


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def deficit_up_to_stop(r, method="aaars"):
    """Return delta(t) = 1 - coverage(t) for t = 1..min(stop, series_len)."""
    cov = np.asarray(r.get("coverage_series"), dtype=float)
    if cov.size == 0:
        return np.array([]), r.get("T_total", 0)
    stop = r.get(f"{method}__t")
    if stop is None:
        stop = r.get("T_total", cov.size)
    stop = max(1, min(int(stop), cov.size))
    return 1.0 - cov[:stop], stop


def d_bar(cov, stop):
    return float(np.mean(cov)) if cov.size else float("nan")


def d_star(delta, stop, frac):
    """Max sustained deficit over any trailing window of length w=frac*stop."""
    w = max(1, int(round(frac * stop)))
    if delta.size < w:
        w = delta.size
    if w == 0:
        return float("nan")
    cs = np.cumsum(np.concatenate(([0.0], delta)))
    win_mean = (cs[w:] - cs[:-w]) / w
    return float(np.max(win_mean)) if win_mean.size else float("nan")


def is_fc(r, method="aaars"):
    return (r.get(f"{method}__t") is not None
            and r[f"{method}__recall"] < RECALL_THR)


def step0(records, alloc):
    """Falsify premise: can deficit stay high to the horizon (never squeezed)?"""
    rows = [r for r in records if r["alloc"] == alloc and r.get("coverage_series")]
    n = len(rows)
    # First vs last decile of delta -> trend; and final-5th max deficit.
    trends, tail_high, tail_very_high = [], [], []
    for r in rows:
        delta, stop = deficit_up_to_stop(r)
        if delta.size < 20:
            continue
        w = max(1, delta.size // 10)
        d_first = float(np.mean(delta[:w]))
        d_last = float(np.mean(delta[-w:]))
        trends.append(d_last - d_first)          # <~0 -> squeezed down
        tail = delta[-max(1, delta.size // 5):]
        tail_high.append(float(np.max(tail)) > 0.5)
        tail_very_high.append(float(np.max(tail)) > 0.7)
    trends = np.array(trends)
    print(f"\n[Step 0] {alloc.upper()}  (n={n}) — premise: deficit gets squeezed down")
    print(f"   mean delta(last-first)  = {trends.mean():+.3f}  "
          f"(<0 -> downward trend)")
    print(f"   share episodes with final-5th max deficit > 0.5 : "
          f"{100.0*np.mean(tail_high):.1f}%")
    print(f"   share episodes with final-5th max deficit > 0.7 : "
          f"{100.0*np.mean(tail_very_high):.1f}%")
    print(f"   => premise {'PLAUSIBLE' if np.mean(tail_very_high) < 0.05 else 'QUESTIONABLE'} "
          f"(episodes stuck near-high deficit to end)")
    return rows


def build_scalars(rows, method="aaars"):
    """Return list of (alloc, d_bar, {frac: d_star}, fc) per episode."""
    out = []
    for r in rows:
        delta, stop = deficit_up_to_stop(r, method)
        if delta.size == 0:
            continue
        feat = {
            "alloc": r["alloc"],
            "d_bar": d_bar(delta, stop),
            "d_star": {f: d_star(delta, stop, f) for f in WINDOW_FRACS},
            "fc": int(is_fc(r, method)),
            "recall": r[f"{method}__recall"],
        }
        out.append(feat)
    return out


def monotone_check(k_vals, stats):
    """stats: list of (bins, fc_rate[], n[]) ; report ascending monotonicity."""
    rates = [s[1] for s in stats]
    import scipy.stats as st
    if len(rates) >= 3:
        rho, p = st.spearmanr(np.arange(len(rates)), rates)
        return rho, p
    return float("nan"), float("nan")


def step2(features, label):
    print(f"\n[Step 2] {label} — empirical AAARS FC rate vs deficit scalar "
          f"(Wilson CI; increasing => FC worsens with deficit)")
    scalars = [("d_bar", None)] + [("d_star", f) for f in WINDOW_FRACS]
    for key, frac in scalars:
        label = key if key == "d_bar" else f"d_star({frac})"
        if key == "d_bar":
            vals = [x["d_bar"] for x in features]
        else:
            vals = [x["d_star"][frac] for x in features]
        # drop nans
        pairs = [(v, x["fc"]) for v, x in zip(vals, features) if v == v]
        if len(pairs) < NBINS * 2:
            print(f"   {label:12s}: too few valid episodes "
                  f"({len(pairs)}) -- skipped")
            continue
        vv = np.array([p[0] for p in pairs])
        fc = np.array([p[1] for p in pairs])
        edges = np.quantile(vv, np.linspace(0, 1, NBINS + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_lo, bin_hi, bin_rate, bin_n = [], [], [], []
        print(f"   {label:12s}: bins by quantile (low -> high deficit)")
        for i in range(NBINS):
            m = (vv >= edges[i]) & (vv < edges[i + 1])
            if not m.any():
                continue
            k = int(fc[m].sum())
            nn = int(m.sum())
            lo, hi = wilson_ci(k, nn)
            bin_lo.append(vv[m].mean())
            bin_rate.append(100.0 * k / nn)
            bin_n.append(nn)
            print(f"      bin[{i}] d_avg={vv[m].mean():.3f}  "
                  f"n={nn:3d}  FC={100.0*k/nn:5.1f}% [{lo:5.1f},{hi:5.1f}]")
        if len(bin_rate) >= 3:
            rho, p = monotone_check(None, [(0, r) for r in bin_rate])
            print(f"      -> Spearman(rho) of FC on this scalar = {rho:+.2f} "
                  f"(p={p:.3f})")
    return


def main():
    # (file, [allocs]) cohorts; all carry coverage_series + aaars stop/recall.
    cohorts = [
        (COHORT, ["minerich", "boustro"]),
        (POLICIES, ["frontier", "greedy", "hotspot"]),
    ]

    print("=" * 78)
    print("STEP 0 — Falsify the 'eventually squeezed' premise")
    print("=" * 78)
    expanded = {}
    for path, allocs in cohorts:
        with open(path) as f:
            records = json.load(f)
        for a in allocs:
            rows = step0(records, a)
            expanded[a] = rows

    print("\n" + "=" * 78)
    print("STEP 1+2 — AAARS FC rate vs deficit scalar (C1: AAARS bounded)")
    print("=" * 78)
    labels = {
        "minerich": "MineRichness (AAARS)",
        "boustro": "Boustro (AAARS, FC~0 reference)",
        "frontier": "Frontier (AAARS)",
        "greedy": "Greedy (AAARS)",
        "hotspot": "Hotspot (AAARS)",
    }
    for a in ["minerich", "boustro", "frontier", "greedy", "hotspot"]:
        if a in expanded:
            step2(build_scalars(expanded[a]), labels[a])

    # C1 suggestion: keep Chao1 as reference curve on the same binning.
    print("\n" + "=" * 78)
    print("STEP 2 (reference) — Chao1 FC rate vs deficit (C1: reference only)")
    print("=" * 78)
    if "minerich" in expanded:
        step2(build_scalars(expanded["minerich"], method="chao1_ci"),
              "MineRichness (Chao1 reference)")


if __name__ == "__main__":
    main()
