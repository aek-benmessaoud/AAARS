"""Compute all statistics for the AAARS paper tables (v2: with CIs and tests).

Usage: python results/paper_stats_v2.py <confirm_json> <sweep_json> [label]

Adds:
  - Wilson 95% CI on every proportion (FC%, stopping rate)
  - Bootstrap CI on median stop time
  - SD on mean recall
  - Fisher exact (AAARS vs Chao1) on FC counts
"""
import json
import sys
import numpy as np

CONFIRM = sys.argv[1] if len(sys.argv) > 1 else r"F:\Project11-AAARS\results\raw\confirmation_base.json"
SWEEP = sys.argv[2] if len(sys.argv) > 2 else r"F:\Project11-AAARS\results\raw\config_sweep.json"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "confirmation"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * (centre - half), 100.0 * (centre + half))


def med_ci(vals, n_boot=2000, seed=0, ci=0.95):
    """Bootstrap percentile CI on the median."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(vals, size=len(vals), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi


def fisher_exact(a, b, c, d):
    from math import lgamma, exp
    def log_hypergeom(x, a, b, c, d):
        row0 = a + b; col0 = a + c
        return (lgamma(row0 + 1) + lgamma(c + d + 1)
                + lgamma(col0 + 1) + lgamma(b + d + 1)
                - lgamma(a + b + c + d + 1)
                - lgamma(x + 1) - lgamma(row0 - x + 1)
                - lgamma(col0 - x + 1) - lgamma(b + d - (row0 - x) + 1))
    n = a + b + c + d
    row0 = a + b
    lo = max(0, a + c - (c + d))
    hi = min(row0, a + c)
    p_obs_p = exp(log_hypergeom(a, a, b, c, d))
    total = 0.0
    for x in range(lo, hi + 1):
        p = exp(log_hypergeom(x, a, b, c, d))
        if p <= p_obs_p * (1 + 1e-9):
            total += p
    return total


with open(CONFIRM) as f:
    D = json.load(f)
with open(SWEEP) as f:
    SW = json.load(f)

METHODS = ["chao1_ci", "chao92_ci", "aaars", "discrete_aaars", "oracle_95",
           "fixed_2", "diminishing"]
ALLOCS = ["boustro", "minerich"]

n_ep = len(D) // 2
print(f"=== TABLE 1: Confirmation ({LABEL}, {n_ep} seeds/alloc, K=60, N=6) ===")
print("   [Wilson 95% CI on FC%; bootstrap 95% CI on med stop time]\n")
hdr = (f"{'Method':14s}{'Alloc':10s}{'Stop':>7s}{'FC':>4s}{'FC% [CI]':>22s}"
       f"{'Med_t[CI]':>20s}{'MedR':>7s}{'sdR':>7s}{'MinR':>7s}")
print(hdr); print("-" * len(hdr))
for m in METHODS:
    for a in ALLOCS:
        sub = [r for r in D if r["alloc"] == a]
        stops = [r for r in sub if r.get(f"{m}__t") is not None]
        n = len(sub)
        if not stops:
            print(f"{m:14s}{a:10s}{0:>4d}/{n:<3d}{'-':>4s}{'-':>22s}{'-':>20s}{'-':>7s}{'-':>7s}{'-':>7s}")
            continue
        fc = sum(1 for r in stops if r[f"{m}__recall"] < 95.0)
        fc_pct = 100.0 * fc / len(stops)
        lo, hi = wilson_ci(fc, len(stops))
        times = [r[f"{m}__t"] for r in stops]
        med_t = np.median(times)
        tlo, thi = med_ci(times)
        recalls = [r[f"{m}__recall"] for r in stops]
        med_r = np.median(recalls)
        sd_r = np.std(recalls)
        mn_r = min(recalls)
        print(f"{m:14s}{a:10s}{len(stops):>4d}/{n:<3d}{fc:>4d}"
              f"{fc_pct:>7.1f}% [{lo:>4.1f},{hi:>4.1f}]"
              f"{med_t:>7.0f} [{tlo:>5.0f},{thi:>5.0f}]"
              f"{med_r:>7.1f}{sd_r:>7.1f}{mn_r:>7.1f}")

print("\n=== FISHER EXACT: AAARS vs Chao1 FC (mine richness) ===")
for a in ALLOCS:
    aars = [r for r in D if r["alloc"] == a and r.get("aaars__t")]
    chao = [r for r in D if r["alloc"] == a and r.get("chao1_ci__t")]
    ma = sum(1 for r in aars if r["aaars__recall"] < 95.0)
    mc = sum(1 for r in chao if r["chao1_ci__recall"] < 95.0)
    na, nc = len(aars), len(chao)
    p = fisher_exact(ma, na - ma, mc, nc - mc)
    print(f"  {a:10s}: AAARS {ma}/{na} vs Chao1 {mc}/{nc}  "
          f"FC% {100*ma/na:.0f} vs {100*mc/nc:.0f}  p={p:.4f}" +
          ("  (*)" if p < 0.05 else ""))

print("\n=== Stop timing: AAARS vs Chao1 under minerich ===")
mm = [r for r in D if r["alloc"] == "minerich"]
later = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"] > r["chao1_ci__t"])
same = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"] == r["chao1_ci__t"])
earlier = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"] < r["chao1_ci__t"])
print(f"  later: {later}, same: {same}, earlier: {earlier}  (of {len(mm)})")

print("\n=== Risk / switches ===")
for a in ALLOCS:
    sub = [r for r in D if r["alloc"] == a]
    rv = [r.get("aaars__final_risk", 0) for r in sub]
    sw = [r.get("aaars__switches", 0) for r in sub]
    print(f"  {a:10s}: risk mean={np.mean(rv):.4f} sd={np.std(rv):.4f}; "
          f"switches mean={np.mean(sw):.1f}")

KHS = [30, 60, 120]; NHS = [3, 6, 12]
print(f"\n=== TABLE 2: KxN SWEEP under minerich (EXPLORATORY, {len(SW)//(9*2)} seeds/cell) ===")
print("   Wilson 95% CI on each FC%; N/A significance framing")
print(f"{'K':>4s}{'N':>4s}  {'Chao1 FC%':>22s}{'AAARS FC%':>22s}")
for k in KHS:
    for n in NHS:
        sub = [r for r in SW if r["num_mines"] == k and r["num_agents"] == n
               and r["alloc"] == "minerich"]
        nsub = len(sub)
        c = [r for r in sub if r.get("chao1_ci__t")]
        a = [r for r in sub if r.get("aaars__t")]
        cf = sum(1 for r in c if r["chao1_ci__recall"] < 95)
        af = sum(1 for r in a if r["aaars__recall"] < 95)
        cl, ch = wilson_ci(cf, nsub)
        al, ah = wilson_ci(af, nsub)
        print(f"{k:>4d}{n:>4d}  {cf:>2d}/{nsub:<2d} "
              f"({100.0*cf/nsub:>3.0f}% [{cl:>3.0f},{ch:>3.0f}]) "
              f"{af:>2d}/{nsub:<2d} "
              f"({100.0*af/nsub:>3.0f}% [{al:>3.0f},{ah:>3.0f}])")
