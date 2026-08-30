#!/usr/bin/env python
"""
trajectory_test.py — advisor's ONE final cost-free test: trajectory dynamics.

Point scalars (n, delta_bar) both failed the checkpoint-decoupling test, so the
discriminating signal may live in the *shape* of the deficit trajectory, not a
frozen instant. Two families (all from committed coverage_series, no new sim):

  A) DECOUPLED slope (prospective-honest): rate of deficit closure
     d'(t0) over a trailing window ending at fixed external checkpoints t0.
     Still decoupled from the stop decision.

  B) WITHIN-WINDOW shape (retrospective diagnostic): the "slow-then-abrupt"
     signature the advisor described — how fast delta closes in the window just
     before the AAARS stop, relative to the preceding period.
       late_acc  = (delta closure rate in final [T-stop,T]) / (in the window
                    before it).  Large => deficit closes abruptly just before stop.

AUC reported at the same directional convention (higher feature -> FC).
If both sit ~0.5 -> stop the search entirely (exhaustive: static, spatial,
trajectory all null).
"""
import os
import json

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0
CHECKPOINTS = [200, 400, 600]


def auc_roc_direct(labels, scores):
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
            if cov.size < 60:
                continue
            delta = 1.0 - cov
            stop = rec.get("aaars__t")
            if stop is None or stop < 60:
                continue
            stop = min(int(stop), cov.size)
            fc = int(rec["aaars__t"] is not None and rec["aaars__recall"] < RECALL_THR)
            row = {"run": rec["run"], "stop": stop, "fc": fc, "delta": delta}
            for t0 in CHECKPOINTS:
                if t0 + 50 <= cov.size:
                    # closure speed: -(delta[t0]-delta[t0-50])/50  (downward = closing)
                    row[f"slope_{t0}"] = -(delta[t0] - delta[t0 - 50]) / 50.0
                else:
                    row[f"slope_{t0}"] = float("nan")
            # within-window "slow-then-abrupt": late vs prior closure rate
            W = 100
            if stop - W >= 1 and stop - 2 * W >= 0:
                late_drop = delta[stop - W] - delta[stop]          # >0 = closing
                prior_drop = delta[stop - 2 * W] - delta[stop - W]
                # late_acc large => late closure is abrupt relative to prior
                row["late_acc"] = float(late_drop / max(prior_drop, 1e-6))
                row["late_drop"] = float(late_drop)
            else:
                row["late_acc"] = float("nan"); row["late_drop"] = float("nan")
            out[a].append(row)
    return out


def report(a, items):
    labs = [d["fc"] for d in items]
    print(f"\n== {a.upper()} (n_ep={len(items)}, "
          f"FC={100.0*np.mean(labs):.1f}%) ==")
    # A) decoupled slope at fixed checkpoints
    print("   A) decoupled deficit-closure slope d'(t0):")
    for t0 in CHECKPOINTS:
        keep = [d for d in items if d["stop"] > t0 and d[f"slope_{t0}"] == d[f"slope_{t0}"]]
        if len(keep) < 40:
            print(f"      slope@{t0:3d}: skipped (n={len(keep)})")
            continue
        a_ = auc_roc_direct([d["fc"] for d in keep], [d[f"slope_{t0}"] for d in keep])
        print(f"      slope@{t0:3d}: AUC={a_:.3f}  (n_stop_after={len(keep)})")
    # B) within-window late-acceleration shape
    print("   B) within-window 'slow-then-abrupt' shape (retrospective):")
    keep = [d for d in items if d["late_acc"] == d["late_acc"]]
    if len(keep) >= 40:
        a_ = auc_roc_direct([d["fc"] for d in keep], [d["late_acc"] for d in keep])
        la = np.median([d["late_acc"] for d in keep if d["fc"] == 0])
        la_ = np.median([d["late_acc"] for d in keep if d["fc"] == 1])
        ldrop = np.median([d["late_drop"] for d in keep if d["fc"] == 0])
        ldrop_ = np.median([d["late_drop"] for d in keep if d["fc"] == 1])
        print(f"      late_acc: AUC={a_:.3f}  (med safe={la:.2f} vs fc={la_:.2f})")
        print(f"      late_drop(closure in final 100 steps): "
              f"med safe={ldrop:.4f} vs fc={ldrop_:.4f}")
    else:
        print("      skipped")


def main():
    data = load()
    print("ADVISOR TRAJECTORY TEST (one shot, cost zero)")
    for a in ["minerich", "frontier"]:
        report(a, data[a])


if __name__ == "__main__":
    main()
