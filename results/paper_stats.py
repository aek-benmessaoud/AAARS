"""Compute all statistics for the AAARS paper tables."""
import json, numpy as np

with open(r"F:\Project11-AAARS\results\raw\confirmation_base.json") as f:
    D = json.load(f)

METHODS = ["chao1_ci", "chao92_ci", "aaars", "discrete_aaars", "oracle_95",
           "fixed_2", "diminishing"]
ALLOCS = ["boustro", "minerich"]

print("=== TABLE 1: Confirmation (20 seeds, K=60, N=6, 100x100) ===\n")
hdr = f"{'Method':14s}{'Alloc':10s}{'Stop':>7s}{'FC':>5s}{'FC%':>8s}{'Med_t':>7s}{'Med_R%':>8s}{'Min_R%':>8s}"
print(hdr); print("-"*len(hdr))
for m in METHODS:
    for a in ALLOCS:
        sub = [r for r in D if r["alloc"] == a]
        stops = [r for r in sub if r.get(f"{m}__t") is not None]
        n = len(sub)
        if not stops:
            print(f"{m:14s}{a:10s}{0:>4d}/{n:<3d}{'-':>5s}{'-':>8s}{'-':>7s}{'-':>8s}{'-':>8s}")
            continue
        fc = sum(1 for r in stops if r[f"{m}__recall"] < 95.0)
        fc_pct = 100.0*fc/len(stops)
        med_t = int(np.median([r[f"{m}__t"] for r in stops]))
        med_r = float(np.median([r[f"{m}__recall"] for r in stops]))
        mn_r = float(min([r[f"{m}__recall"] for r in stops]))
        print(f"{m:14s}{a:10s}{len(stops):>4d}/{n:<3d}{fc:>5d}{fc_pct:>7.1f}%{med_t:>7d}{med_r:>8.1f}{mn_r:>8.1f}")

print("\n=== AAARS switch counts (continuous blend) ===")
for a in ALLOCS:
    val = [r.get("aaars__switches",0) for r in D if r["alloc"]==a]
    print(f"  {a:10s}: mean={np.mean(val):.1f} median={int(np.median(val))} range={min(val)}-{max(val)}")

print("\n=== Risk scores ===")
for a in ALLOCS:
    val = [r.get("aaars__final_risk",0) for r in D if r["alloc"]==a]
    print(f"  {a:10s}: mean={np.mean(val):.4f} sd={np.std(val):.4f}")

# Distinctee: how many episodes where AAARS stops later than chao1 under minerich
mm = [r for r in D if r["alloc"]=="minerich"]
later = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"]>r["chao1_ci__t"])
same = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"]==r["chao1_ci__t"])
earlier = sum(1 for r in mm if r.get("aaars__t") and r.get("chao1_ci__t") and r["aaars__t"]<r["chao1_ci__t"])
print(f"\n=== minerich: AAARS vs Chao1 stop timing (of 20) ===")
print(f"  AAARS later: {later}, same: {same}, earlier: {earlier}")

print("\n=== SWEEP (K x N, 5 seeds) ===\n")
with open(r"F:\Project11-AAARS\results\raw\config_sweep.json") as f:
    SW = json.load(f)
KHS=[30,60,120]; NHS=[3,6,12]
print(f"{'K':>4s}{'N':>4s}  {'Chao1 FC%':>11s}{'AAARS FC%':>11s}{'Delta pp':>9s}")
for k in KHS:
    for n in NHS:
        sub=[r for r in SW if r["num_mines"]==k and r["num_agents"]==n and r["alloc"]=="minerich"]
        c=[r for r in sub if r.get("chao1_ci__t")]
        aa=[r for r in sub if r.get("aaars__t")]
        cf=100.0*sum(1 for r in c if r["chao1_ci__recall"]<95)/len(c) if c else 0
        af=100.0*sum(1 for r in aa if r["aaars__recall"]<95)/len(aa) if aa else 0
        print(f"{k:>4d}{n:>4d}  {cf:>10.0f}%{af:>10.0f}%{af-cf:>+8.0f}")
