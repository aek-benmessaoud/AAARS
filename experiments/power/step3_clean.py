#!/usr/bin/env python
"""
step3_clean.py — advisor Step 3 (form check) on the *clean* group only:
MineRichness + Frontier, where mean deficit delta_bar dominates FC.

Tests whether a conditional bound of the form  P[FC] <= g(delta_bar, n)
is well-behaved, where n = D1' = cumulative detections (denominator for the
missing-mass / Good-Turing concentration bound). n comes from
  * minerich: d1_prime_minerich.json
  * frontier: gamma_policies.json
and delta_bar (time-mean of 1 - coverage) from the committed coverage_series.

Shape check:
  (a) does FC rise monotonically in delta_bar?        (Spearman on bin means)
  (b) does FC fall with growing n (more total draws)? (lower missing mass)
  (c) a logistic fit FC ~ b0 + b1*delta_bar + b2*(1/n) with correct signs
      (b1>0, b2>0) and decent AUC => the exponential form sits on a real,
      monotone surface rather than on the saturated plateau.
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


def load_clean():
    # minerich: d1_prime gives n_prime; power_revision gives delta_bar + fc
    d1m = {r["run"]: r for r in json.load(open(os.path.join(RAW, "d1_prime_minerich.json")))}
    rev = json.load(open(os.path.join(RAW, "power_revision.json")))
    # frontier: gamma_policies gives n_prime; policies_results gives delta_bar + fc
    gam = {(r["alloc"], r["run"]): r
           for r in json.load(open(os.path.join(RAW, "gamma_policies.json")))}
    pol = json.load(open(os.path.join(RAW, "policies_results.json")))

    def rows_for(alloc, nsrc, polrecs):
        by = {r["run"]: r for r in polrecs if r["alloc"] == alloc}
        out = []
        for run, rec in by.items():
            cov = np.asarray(rec.get("coverage_series"), dtype=float)
            if cov.size == 0:
                continue
            stop = rec.get("aaars__t") or rec.get("T_total", cov.size)
            stop = max(1, min(int(stop), cov.size))
            dbar = float((1.0 - cov[:stop]).mean())
            n = (nsrc.get(run) or {}).get("n_prime")
            fc = int(rec["aaars__t"] is not None and rec["aaars__recall"] < RECALL_THR)
            if n is None:
                continue
            out.append({"run": run, "d_bar": dbar, "n": n, "fc": fc})
        return out

    mine = rows_for("minerich", d1m, rev)
    front_n = {run: r for (a, run), r in gam.items() if a == "frontier"}
    front = rows_for("frontier", front_n, pol)
    return {"MineRichness": mine, "Frontier": front}


def bins_by(features, scalar, nbins=NBINS):
    vals = [f[scalar] for f in features]
    pairs = [(v, f["fc"]) for v, f in zip(vals, features) if v == v]
    if len(pairs) < nbins * 2:
        return None
    vv = np.array([p[0] for p in pairs]); fc = np.array([p[1] for p in pairs])
    edges = np.quantile(vv, np.linspace(0, 1, nbins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out = []
    for i in range(nbins):
        m = (vv >= edges[i]) & (vv < edges[i + 1])
        if not m.any():
            continue
        k = int(fc[m].sum()); nn = int(m.sum())
        out.append((float(vv[m].mean()), 100.0 * k / nn, nn,
                    wilson_ci(k, nn)))
    return out


def logistic(features):
    """Joint least-squares fit FC ~ b0 + b1*d_bar + b2*(1/n); AUC per model."""
    nfeat = len(features)
    if nfeat == 0:
        return None
    x = np.array([[1.0, f["d_bar"], 1.0 / f["n"]] for f in features])
    y = np.array([f["fc"] for f in features], float)
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    def auc_of(pred):
        pred = np.asarray(pred, float)
        pos = pred[y == 1]; neg = pred[y == 0]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        return float((pos[:, None] > neg[None, :]).mean())

    z = x @ coef
    pred = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    joint_auc = auc_of(pred)

    # univariate AUCs: single-covariate least-squares logistic
    uni = {}
    for i, nm in [(1, "d_bar"), (2, "n")]:
        xi = np.column_stack([np.ones(nfeat), x[:, i]])
        ci, *_ = np.linalg.lstsq(xi, y, rcond=None)
        zi = xi @ ci
        pi = 1.0 / (1.0 + np.exp(-np.clip(zi, -30, 30)))
        uni[nm] = auc_of(pi)
    return coef, joint_auc, uni


def report(label, features):
    if not features:
        print(f"\n==== {label.upper()}  (EMPTY / no data) ====")
        return
    print(f"\n==== {label.upper()}  (n_ep={len(features)}, "
          f"overall FC={100.0*sum(f['fc'] for f in features)/len(features):.1f}%) ====")
    print(f"   n (D1')      : min={min(f['n'] for f in features)} "
          f"med={np.median([f['n'] for f in features]):.0f} "
          f"max={max(f['n'] for f in features)}")
    print(f"   delta_bar     : min={min(f['d_bar'] for f in features):.3f} "
          f"max={max(f['d_bar'] for f in features):.3f}")

    for scalar, desc in [("d_bar", "FC rises with deficit?"),
                         ("n", "FC falls with more draws (n)?")]:
        b = bins_by(features, scalar)
        if not b:
            print(f"   {scalar:9s}: skipped")
            continue
        rho, pv = spearman([x[0] for x in b], [x[1] for x in b])
        spread = b[-1][1] - b[0][1]
        direction = "rising" if spread >= 0 else "falling"
        print(f"   {scalar:9s} ({desc}) : FC low={b[0][1]:5.1f}% -> "
              f"high={b[-1][1]:5.1f}%  (spread {spread:+5.1f}pp, {direction})  "
              f"rho={rho:+.2f} (p={pv:.3f})")

    fit = logistic(features)
    if fit:
        coef, auc, uni = fit
        b1, b2 = coef[1], coef[2]
        print(f"   logistic FC ~ b0 + b1*delta_bar + b2*(1/n): "
              f"b1(dbar)={b1:+.3f}, b2(1/n)={b2:+.3f}, AUC={auc:.3f}")
        print(f"   univariate AUC : delta_bar={uni['d_bar']:.3f}  "
              f"n(D1')={uni['n']:.3f}")
        # collinearity: sign reversal of b1 in joint fit while univariate > 0
        joint_div = b1 > 0
        uni_div = uni["d_bar"] >= 0.6
        print(f"   delta_bar discriminates? univariate-AUC>=0.6: {uni_div} ; "
              f"joint-sign(+ve): {joint_div} "
              f"({'OK' if uni_div else 'weak'})")
        verdict = ("well-behaved" if (uni["n"] >= 0.7 and uni_div)
                   else "partial (n-driven)" if uni["n"] >= 0.7
                   else "not-well-behaved")
        print(f"   -> form {verdict}")
    else:
        print("   logistic fit failed")


def main():
    sets = load_clean()
    print("ADVISOR STEP 3 — exponential-form check on the *clean* group only")
    for label, feats in sets.items():
        report(label, feats)


if __name__ == "__main__":
    main()
