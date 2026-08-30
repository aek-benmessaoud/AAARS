#!/usr/bin/env python
"""checkpoint_analysis.py — advisor circularity test for the D1' covariate.

Compare the predictive AUC of FC on cumulative detections n
  * at the AAARS stop time  (n_stop -> suspect: tautological with FC)
  * at fixed external checkpoints t0 (n_at[t0]) decoupled from the stop decision.

If AUC persists at the decoupled checkpoints -> real missing-mass signal.
If it collapses toward ~0.5 -> stop-time tautology.
"""
import os
import json

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
CHECKPOINTS = [100, 200, 300, 400]


def auc_roc(labels, scores):
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # directional: higher score -> positive label
    return float((pos[:, None] > neg[None, :]).mean())


def bins(vals, labels, nbins=6):
    vals = np.asarray(vals, float); labels = np.asarray(labels, int)
    edges = np.quantile(vals, np.linspace(0, 1, nbins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out = []
    for i in range(nbins):
        m = (vals >= edges[i]) & (vals < edges[i + 1])
        if not m.any():
            continue
        out.append((float(vals[m].mean()), 100.0 * labels[m].mean(), int(m.sum())))
    return out


def report(a, data):
    items = data
    print(f"\n== {a.upper()} (n_ep={len(items)}, "
          f"FC={100.0*np.mean([d['fc'] for d in items]):.1f}%) ==")
    labs = [d["fc"] for d in items]
    rng_stop = (min(d["n_stop"] for d in items), max(d["n_stop"] for d in items))
    print(f"   n_stop range: {rng_stop}")
    # stop-time
    a_stop = auc_roc(labs, [-d["n_stop"] for d in items])
    print(f"   n @ STOP     : FC-vs-AUC={a_stop:.3f}   <- stop-time (suspect)")
    for t0 in CHECKPOINTS:
        outside = [d for d in items if d["aaars_t"] is not None and d["aaars_t"] > t0]
        if len(outside) < 40:
            print(f"   n @ {t0:4d}  : skipped (only {len(outside)} episodes stop after t0)")
            continue
        # low n at checkpoint -> higher FC, so scores negated for AUC
        sc = [-d["n_at"].get(str(t0), 0) for d in outside]
        a = auc_roc([d["fc"] for d in outside], sc)
        # geometric mean of checkpoint n split by fc outcome
        nf = np.median([d["n_at"].get(str(t0), 0) for d in outside if d["fc"] == 0])
        np_ = np.median([d["n_at"].get(str(t0), 0) for d in outside if d["fc"] == 1])
        print(f"   n @ {t0:4d}  : FC-vs-AUC={a:.3f}   "
              f"(med n safe={nf:.0f} vs fc={np_:.0f})  [n_ep after t0={len(outside)}]")
    # checkpoint trend bins at t0=200
    b = bins([d["n_at"].get("200", 0) for d in items],
             [d["fc"] for d in items])
    print(f"   n@200 bins (low->high): " +
          ", ".join(f"{x[1]:.0f}%" for x in b))


def main():
    data = json.load(open(os.path.join(RAW, "checkpoint_n_clean.json")))
    by = {"minerich": [], "frontier": []}
    for d in data:
        if d["alloc"] in by:
            by[d["alloc"]].append(d)
    print("ADVISOR CIRCULARITY TEST — FC vs n at fixed checkpoints (decoupled)")
    for a in ["minerich", "frontier"]:
        report(a, by[a])


if __name__ == "__main__":
    main()
