#!/usr/bin/env python
"""dbar_checkpoint.py — checkpoint-decoupling test for the *fallback* covariate.

delta_bar is currently mean(deficit up to the AAARS stop window), which shares
the same circularity family as n(stop) (a stop-time statistic). To decide
whether delta_bar is an honest exogenous covariate, re-test FC vs mean deficit
computed at FIXED external checkpoints t0 (no dependence on when the rule
stopped). Uses only the committed coverage_series (no new simulation).

If FC vs delta_bar(t0) keeps a real (modest) AUC at decoupled checkpoints ->
delta_bar is an honest covariate and a conditional bound P[FC] <= g(delta_bar)
is salvageable. If it collapses to ~0.5 -> delta_bar is also stop-confounded.
"""
import os
import json

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0
CHECKPOINTS = [100, 200, 300, 400, 600]


def auc_roc_direct(labels, scores):
    """higher score -> label 1"""
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean())


def load():
    rev = json.load(open(os.path.join(RAW, "power_revision.json")))
    pol = json.load(open(os.path.join(RAW, "policies_results.json")))
    out = {"minerich": [], "frontier": []}
    for a, cohort in [("minerich", rev), ("frontier", pol)]:
        for rec in [r for r in cohort if r["alloc"] == a]:
            cov = np.asarray(rec.get("coverage_series"), dtype=float)
            if cov.size == 0:
                continue
            delta = 1.0 - cov
            stop = rec.get("aaars__t")
            # honesty: exclude episodes whose stop < t0 == 0 (must stop after)
            bas = {"run": rec["run"],
                   "stop": stop,
                   "fc": int(rec["aaars__t"] is not None
                             and rec["aaars__recall"] < RECALL_THR),
                   "db_stop": float(delta[:min(stop or cov.size, cov.size)].mean())}
            for t0 in CHECKPOINTS:
                lim = min(t0, cov.size)
                bas[f"db_{t0}"] = float(delta[:lim].mean()) if lim else float("nan")
            out[a].append(bas)
    return out


def report(a, items):
    print(f"\n== {a.upper()} (n_ep={len(items)}, "
          f"FC={100.0*np.mean([d['fc'] for d in items]):.1f}%) ==")
    labs = [d["fc"] for d in items]
    # stop-window delta_bar
    db_stop = [d["db_stop"] for d in items]
    a_stop = auc_roc_direct(labs, db_stop)
    print(f"   delta_bar @ stop-window : AUC={a_stop:.3f}  (family-circular reference)")
    for t0 in CHECKPOINTS:
        # keep episodes that actually stop after t0 (so they have a real 'later' outcome)
        keep = [d for d in items if d["stop"] is not None and d["stop"] > t0]
        if len(keep) < 40:
            print(f"   delta_bar @ {t0:4d} : skipped (only {len(keep)} stop after t0)")
            continue
        sc = [d[f"db_{t0}"] for d in keep]
        lk = [d["fc"] for d in keep]
        a = auc_roc_direct(lk, sc)
        nf = np.median([d[f"db_{t0}"] for d in keep if d["fc"] == 0])
        npp = np.median([d[f"db_{t0}"] for d in keep if d["fc"] == 1])
        print(f"   delta_bar @ {t0:4d} : AUC={a:.3f}  "
              f"(med safe={nf:.3f} vs fc={npp:.3f})  [n_stop_after={len(keep)}]")


def main():
    data = load()
    print("ADVISOR FALLBACK CHECK — is delta_bar an honest covariate?")
    for a in ["minerich", "frontier"]:
        report(a, data[a])


if __name__ == "__main__":
    main()
