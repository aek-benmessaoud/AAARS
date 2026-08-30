#!/usr/bin/env python
"""
regression_guard.py — deliverable D4: assert the manuscript's reported numbers
against the committed campaign data on disk.

The paper (`paper/aaars.tex`) is generated from five campaign runs. This guard
recomputes the aggregate statistics that each manuscript table reports, straight
from the committed raw JSON, and fails if any figure no longer matches what is
written in the text. This makes the "reproducible / reproducible evaluation"
claim checkable by a reviewer-style artifact check, and prevents silent
regeneration of a number that contradicts the submitted manuscript.

Campaign files <-> manuscript tables:
  confirmation confirm      : results/raw/power_revision.json    -> tab:confirm
  paired 2x2                : results/raw/power_revision.json    -> tab:paired
  mechanism isolation       : results/raw/power_revision.json +
                              results/raw/power_baselines.json   -> tab:iso
  realistic policies        : results/raw/policies_results.json  -> tab:realistic
  lambda sensitivity        : results/raw/lambda_sweep.json      -> sec:lambda
  K x N sweep               : results/raw/config_sweep.json      -> tab:sweep
  P5 robustness             : results/raw/p5h_obstacles.json +
                              results/raw/p5h_commdelay.json     -> tab:robust

Every FC count / FC% / median stop time / mean recall below was transcribed
verbatim from `paper/aaars.tex`. Update the fixtures here ONLY when the paper
is intentionally updated.
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RAW = os.path.join(_ROOT, "results", "raw")
RECALL_THR = 95.0


def load(name):
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("results", d)


def by_alloc(recs, alloc):
    return [r for r in recs if r.get("alloc") == alloc]


def fc_count(recs, method):
    return sum(1 for r in recs
               if r.get(f"{method}__t") is not None
               and r[f"{method}__recall"] < RECALL_THR)


def median_stop(recs, method):
    ts = [r[f"{method}__t"] for r in recs if r.get(f"{method}__t") is not None]
    return float(np.median(ts)) if ts else None


def mean_recall(recs, method):
    v = [r[f"{method}__recall"] for r in recs
         if r[f"{method}__recall"] is not None]
    return float(np.mean(v)) if v else None


def exact_mcnemar(b, c):
    from math import exp, lgamma
    n = b + c
    if n == 0:
        return 1.0
    p0 = exp(lgamma(n + 1) - lgamma(min(b, c) + 1)
             - lgamma(n - min(b, c) + 1) - n * np.log(2.0))
    tot = 0.0
    for k in range(n + 1):
        pk = exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
                 - n * np.log(2.0))
        if pk <= p0 * (1 + 1e-12):
            tot += pk
    return tot


def paired_2x2(recs, m1, m2):
    a = b = c = dd = 0
    for r in recs:
        fc1 = r.get(f"{m1}__t") is not None and r[f"{m1}__recall"] < RECALL_THR
        fc2 = r.get(f"{m2}__t") is not None and r[f"{m2}__recall"] < RECALL_THR
        if not fc1 and not fc2:
            a += 1
        elif fc1 and not fc2:
            b += 1
        elif not fc1 and fc2:
            c += 1
        else:
            dd += 1
    return a, b, c, dd


def paired_boot_ci(recs, m1, m2, n_iter=20000, seed=0, alpha=0.05):
    pairs = [(r.get(f"{m1}__t") is not None and r[f"{m1}__recall"] < RECALL_THR,
              r.get(f"{m2}__t") is not None and r[f"{m2}__recall"] < RECALL_THR)
             for r in recs]
    N = len(pairs)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, N, size=N)
        ch = sum(1 for j in idx if pairs[j][0])
        aa = sum(1 for j in idx if pairs[j][1])
        diffs[i] = (ch - aa)
    rate = diffs / N * 100.0
    return tuple(round(x, 1) for x in
                 np.percentile(rate, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def approx(a, b, tol=0.06):
    return a is not None and b is not None and abs(a - b) <= tol


class Guard:
    def __init__(self):
        self.failures = []

    def check(self, what, cond, detail=""):
        print(f"  [{'OK ' if cond else 'FAIL'}] {what}"
              + (f"  ({detail})" if detail else ""))
        if not cond:
            self.failures.append(what)

    def finalize(self):
        print("=" * 70)
        if not self.failures:
            print("REGRESSION GUARD: ALL MANUSCRIPT FIGURES REPRODUCED")
            return 0
        print(f"REGRESSION GUARD: {len(self.failures)} FIGURES DRIFTED:")
        for f in self.failures:
            print("   -", f)
        return 1


def main():
    rev = load("power_revision.json")
    bases = load("power_baselines.json")
    pols = load("policies_results.json")
    lam = load("lambda_sweep.json")
    sweep = load("config_sweep.json")
    h1 = load("p5h_H1.json")
    h2 = load("p5h_H2.json")
    front_h2 = load("p5h_frontier_H2.json")
    g = Guard()

    # ---- tab:confirm ---------------------------------------------------
    print("tab:confirm (power_revision.json)")
    mr = by_alloc(rev, "minerich")
    bo = by_alloc(rev, "boustro")
    g.check("Boustro Chao1/AAARS FC=0/0",
            fc_count(bo, "chao1_ci") == 0 and fc_count(bo, "aaars") == 0)
    g.check("MineRich Chao1/AAARS FC=28/17",
            fc_count(mr, "chao1_ci") == 28 and fc_count(mr, "aaars") == 17)
    g.check("MineRich Chao1 med_t=732",
            round(median_stop(mr, "chao1_ci")) == 732,
            f"got {median_stop(mr,'chao1_ci')}")
    g.check("MineRich AAARS med_t=945",
            round(median_stop(mr, "aaars")) == 945,
            f"got {median_stop(mr,'aaars')}")
    g.check("Discrete-AAARS MineRich FC=28 (==Chao1)",
            fc_count(mr, "discrete_aaars") == 28)
    g.check("Chao92 MineRich FC=2", fc_count(mr, "chao92_ci") == 2)
    g.check("Diminishing MineRich FC=37", fc_count(mr, "diminishing") == 37)

    # ---- tab:paired ----------------------------------------------------
    print("tab:paired (power_revision.json)")
    a, b, c, dd = paired_2x2(mr, "chao1_ci", "aaars")
    g.check("2x2 52,0 / 11,17", (a, b, c, dd) == (52, 11, 0, 17),
            f"got {a},{b},{c},{dd}")
    p = exact_mcnemar(b, c)
    g.check("Exact McNemar p<0.001", p < 0.001, f"p={p:.4f}")
    lo, hi = paired_boot_ci(mr, "chao1_ci", "aaars")
    # paired_boot_ci returns the positive diff rate (Chao-AAARS). The paper
    # prints the reduction-in-AAARS-FC framing [-21.2,-6.2] = -(hi, lo).
    g.check("paired bootstrap CI (paper [-21.2,-6.2])",
            lo == 6.2 and hi == 21.2, f"raw [{lo},{hi}] => paper [{-hi},{-lo}]")

    # ---- tab:iso -------------------------------------------------------
    print("tab:iso (power_revision.json + power_baselines.json)")
    g.check("Threshold-only MineRich FC=18/22.5%",
            fc_count(mr, "threshold_aaars") == 18,
            f"med_t={median_stop(mr,'threshold_aaars')}")
    g.check("Threshold-only MineRich med_t=923",
            round(median_stop(mr, "threshold_aaars")) == 923,
            f"got {median_stop(mr,'threshold_aaars')}")
    g.check("Coverage-only Boustro FC=36/45.0%",
            fc_count(bo, "coverage_only") == 36)
    g.check("Coverage-only MineRich FC=33/41.2%",
            fc_count(mr, "coverage_only") == 33)
    mb = by_alloc(bases, "minerich")
    bb = by_alloc(bases, "boustro")
    g.check("rateCS Boustro/MineRich FC=72/70",
            fc_count(bb, "rate_cs") == 72 and fc_count(mb, "rate_cs") == 70)
    g.check("gapSPRT Boustro/MineRich FC=77/80",
            fc_count(bb, "gap_sprt") == 77 and fc_count(mb, "gap_sprt") == 80)
    g.check("rateCS MineRich med_t=468",
            round(median_stop(mb, "rate_cs")) == 468,
            f"got {median_stop(mb,'rate_cs')}")
    g.check("gapSPRT MineRich med_t=78",
            round(median_stop(mb, "gap_sprt")) == 78,
            f"got {median_stop(mb,'gap_sprt')}")
    a, b, c, dd = paired_2x2(mb, "aaars", "gap_sprt")
    # paired_2x2(m1=aaars, m2=gap_sprt): c = AAARS-safe & gapSPRT-FC
    # (= 63 episodes gapSPRT false-certs that AAARS keeps safe), b = 0.
    g.check("AAARS vs gapSPRT discordant 63/0",
            c == 63 and b == 0, f"got gapSPRT-fc-only={c},aaars-fc-only={b}")
    a, b, c, dd = paired_2x2(mb, "aaars", "rate_cs")
    g.check("AAARS vs rateCS discordant 54/1",
            c == 54 and b == 1, f"got rateCS-fc-only={c},aaars-fc-only={b}")

    # ---- tab:realistic -------------------------------------------------
    print("tab:realistic (policies_results.json)")
    expect = {
        "frontier": (24, 19, 95.1, 96.6),
        "greedy":   (75, 65, 81.1, 88.5),
        "hotspot":  (80, 71, 57.2, 85.1),
    }
    for pol, (fc1, faa, r1, raa) in expect.items():
        p = by_alloc(pols, pol)
        g.check(f"{pol} Chao1/AAARS FC={fc1}/{faa}",
                fc_count(p, "chao1_ci") == fc1 and fc_count(p, "aaars") == faa,
                f"got {fc_count(p,'chao1_ci')}/{fc_count(p,'aaars')}")
        g.check(f"{pol} mean recall {r1}/{raa}",
                approx(mean_recall(p, "chao1_ci"), r1)
                and approx(mean_recall(p, "aaars"), raa),
                f"got {mean_recall(p,'chao1_ci')}/{mean_recall(p,'aaars')}")

    # ---- sec:lambda ----------------------------------------------------
    print("sec:lambda (lambda_sweep.json)")
    lam_spec = {0.0: (27, 766), 0.5: (22, 886), 1.0: (17, 945),
                2.0: (13, 980), 4.0: (10, 1180)}
    for la, (fc_exp, t_exp) in lam_spec.items():
        cell = [r for r in lam if r.get("alloc") == "minerich"
                and r.get("lam") == la]
        g.check(f"MineRich lam={la} AAARS FC={fc_exp}",
                fc_count(cell, "aaars") == fc_exp,
                f"got {fc_count(cell,'aaars')}")
        g.check(f"MineRich lam={la} med_t={t_exp}",
                round(median_stop(cell, "aaars")) == t_exp,
                f"got {median_stop(cell,'aaars')}")
    bo_lam = [r for r in lam if r.get("alloc") == "boustro"]
    per_lam_ok = all(fc_count([r for r in bo_lam if r.get("lam") == la_],
                              "aaars") == 0
                     for la_ in set(r.get("lam") for r in bo_lam))
    g.check("Boustro AAARS FC=0 at every lambda", per_lam_ok)

    # ---- tab:sweep -----------------------------------------------------
    print("tab:sweep (config_sweep.json)")
    sweep_spec = {
        (30, 3): (2, 2), (30, 6): (1, 0), (30, 12): (2, 2),
        (60, 3): (0, 0), (60, 6): (2, 1), (60, 12): (1, 0),
        (120, 3): (3, 2), (120, 6): (2, 2), (120, 12): (2, 0),
    }
    for (k, n), (fc1, faa) in sweep_spec.items():
        cell = [r for r in sweep if r.get("num_mines") == k
                and r.get("num_agents") == n and r.get("alloc") == "minerich"]
        g.check(f"sweep K={k} N={n} FC {fc1}/{faa}",
                fc_count(cell, "chao1_ci") == fc1
                and fc_count(cell, "aaars") == faa,
                f"got {fc_count(cell,'chao1_ci')}/{fc_count(cell,'aaars')}")

    # ---- tab:gen (P5 heterogeneous detectability) --------------------
    print("tab:gen (p5h_H1.json + p5h_H2.json + p5h_frontier_H2.json)")
    mr_h1 = by_alloc(h1, "minerich")
    mr_h2 = by_alloc(h2, "minerich")
    ft_h2 = by_alloc(front_h2, "frontier")
    # H1
    g.check("H1 MineRich Chao1/AAARS FC=27/21",
            fc_count(mr_h1, "chao1_ci") == 27 and fc_count(mr_h1, "aaars") == 21,
            f"got {fc_count(mr_h1,'chao1_ci')}/{fc_count(mr_h1,'aaars')}")
    g.check("H1 MineRich AAARS med_t=828",
            round(median_stop(mr_h1, "aaars")) == 828,
            f"got {median_stop(mr_h1,'aaars')}")
    a, b, c, dd = paired_2x2(mr_h1, "chao1_ci", "aaars")
    g.check("H1 McNemar (6,0) p=0.031 c=0",
            b == 6 and c == 0 and abs(exact_mcnemar(b, c) - 0.03125) < 1e-6,
            f"got b={b},c={c},p={exact_mcnemar(b,c):.4f}")
    g.check("H1 MinR AAARS=78.3",
            round(min(r["aaars__recall"] for r in mr_h1), 1) == 78.3)
    # H2 MineRichness
    g.check("H2 MineRich Chao1/AAARS FC=29/25",
            fc_count(mr_h2, "chao1_ci") == 29 and fc_count(mr_h2, "aaars") == 25,
            f"got {fc_count(mr_h2,'chao1_ci')}/{fc_count(mr_h2,'aaars')}")
    g.check("H2 MineRich med_t AAARS=761",
            round(median_stop(mr_h2, "aaars")) == 761,
            f"got {median_stop(mr_h2,'aaars')}")
    a, b, c, dd = paired_2x2(mr_h2, "chao1_ci", "aaars")
    g.check("H2 MineRich McNemar (4,0) p=0.125 c=0",
            b == 4 and c == 0 and abs(exact_mcnemar(b, c) - 0.125) < 1e-6,
            f"got b={b},c={c},p={exact_mcnemar(b,c):.4f}")
    g.check("H2 MineRich MinR AAARS=61.7",
            round(min(r["aaars__recall"] for r in mr_h2), 1) == 61.7)
    # H2 Frontier
    g.check("H2 Frontier Chao1/AAARS FC=15/12",
            fc_count(ft_h2, "chao1_ci") == 15 and fc_count(ft_h2, "aaars") == 12,
            f"got {fc_count(ft_h2,'chao1_ci')}/{fc_count(ft_h2,'aaars')}")
    g.check("H2 Frontier med_t AAARS=822",
            round(median_stop(ft_h2, "aaars")) == 822,
            f"got {median_stop(ft_h2,'aaars')}")
    a, b, c, dd = paired_2x2(ft_h2, "chao1_ci", "aaars")
    g.check("H2 Frontier McNemar (3,0) p=0.250 c=0",
            b == 3 and c == 0 and abs(exact_mcnemar(b, c) - 0.25) < 1e-6,
            f"got b={b},c={c},p={exact_mcnemar(b,c):.4f}")
    g.check("H2 Frontier MinR AAARS=85.0",
            min(r["aaars__recall"] for r in ft_h2) == 85.0)
    # c=0 invariant across ALL five tab:gen cells
    all_five = [by_alloc(rev, "minerich"), mr_h1, mr_h2, ft_h2,
                by_alloc(rev, "boustro")]
    c_all = all(paired_2x2(s, "chao1_ci", "aaars")[2] == 0 for s in all_five)
    g.check("tab:gen c=0 in every cell (never backfires)", c_all)

    # ---- tab:robust (P5 obstacles + comm-delay) -------------------
    print("tab:robust (p5h_obstacles.json + p5h_commdelay.json)")
    rob_obs = load("p5h_obstacles.json")
    rob_dly = load("p5h_commdelay.json")

    def rob_cell(recs, levkey, lev, alloc="minerich", n=80):
        cell = [r for r in recs if r.get(levkey) == lev and r.get("alloc") == alloc]
        assert len(cell) == n, f"{levkey}={lev} alloc={alloc}: got {len(cell)} != {n}"
        return cell

    # expected (lever, level) -> (chao1_fc, aaars_fc, b, c, p)
    robust_expected = {
        ("obs", 0.10): (27, 25, 2, 0, 0.5),
        ("obs", 0.20): (29, 23, 6, 0, 0.03125),
        ("dly", 2):    (29, 20, 9, 0, 0.00390625),
        ("dly", 4):    (23, 16, 7, 0, 0.015625),
    }
    rob_all_cells = []
    for (kind, lev) in robust_expected:
        if kind == "obs":
            cell = rob_cell(rob_obs, "obstacle_ratio", lev)
            cname = f"obs {lev}"
        else:
            cell = rob_cell(rob_dly, "comm_delay", lev)
            cname = f"dly {lev}"
        rob_all_cells.append(cell)
        e_c1, e_aa, e_b, e_c, e_p = robust_expected[(kind, lev)]
        fc1, faa = fc_count(cell, "chao1_ci"), fc_count(cell, "aaars")
        g.check(f"{cname}: FC {e_c1}/{e_aa}",
                fc1 == e_c1 and faa == e_aa, f"got {fc1}/{faa}")
        a, b, c, dd = paired_2x2(cell, "chao1_ci", "aaars")
        p = exact_mcnemar(b, c)
        g.check(f"{cname}: McNemar ({e_b},{e_c}) p={e_p}",
                b == e_b and c == e_c and abs(p - e_p) < 1e-6,
                f"got b={b},c={c},p={p:.6f}")
    c_all_rob = all(paired_2x2(s, "chao1_ci", "aaars")[2] == 0
                    for s in rob_all_cells)
    g.check("tab:robust c=0 in every cell (never backfires)", c_all_rob)

    sys.exit(g.finalize())


if __name__ == "__main__":
    main()
