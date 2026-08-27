import json, numpy as np, os

PROJECT_ROOT = r"F:\Project11-AAARS"
with open(os.path.join(PROJECT_ROOT, "results", "raw", "config_sweep.json")) as f:
    all_results = json.load(f)

K_VALUES = [30, 60, 120]
N_VALUES = [3, 6, 12]
ALLOCATIONS = ["boustro", "minerich"]

print("=" * 80)
print("SWEEP SUMMARY: AAARS vs Chao1 by K and N")
print("=" * 80)
header = f"{'K':>4s} {'N':>3s} {'Alloc':8s} {'AAARS':>10s} {'FC':>4s} {'FC%':>6s} {'Med_t':>7s} | {'Chao1':>10s} {'FC':>4s} {'FC%':>6s} {'Med_t':>7s}"
print(f"\n{header}")
print("-" * len(header))

for k in K_VALUES:
    for n in N_VALUES:
        for alloc in ALLOCATIONS:
            subset = [r for r in all_results
                      if r["alloc"] == alloc
                      and r["num_mines"] == k
                      and r["num_agents"] == n]

            # AAARS
            aaars_stops = [r for r in subset if r.get("aaars__t") is not None]
            aaars_fc = sum(1 for r in aaars_stops if r.get("aaars__recall", 100) < 95.0)
            aaars_n = len(subset)
            aaars_fc_pct = 100.0 * aaars_fc / len(aaars_stops) if aaars_stops else 0
            aaars_med_t = int(np.median([r["aaars__t"] for r in aaars_stops])) if aaars_stops else None

            # Chao1
            chao1_stops = [r for r in subset if r.get("chao1_ci__t") is not None]
            chao1_fc = sum(1 for r in chao1_stops if r.get("chao1_ci__recall", 100) < 95.0)
            chao1_fc_pct = 100.0 * chao1_fc / len(chao1_stops) if chao1_stops else 0
            chao1_med_t = int(np.median([r["chao1_ci__t"] for r in chao1_stops])) if chao1_stops else None

            aaars_s = f"{len(aaars_stops):>2d}/{aaars_n:<2d}"
            chao1_s = f"{len(chao1_stops):>2d}/{aaars_n:<2d}"
            aaars_mt = str(aaars_med_t) if aaars_med_t else "---"
            chao1_mt = str(chao1_med_t) if chao1_med_t else "---"

            print(f"{k:>4d} {n:>3d} {alloc:8s} "
                  f"{aaars_s:>10s} {aaars_fc:>4d} {aaars_fc_pct:>5.1f}% {aaars_mt:>7s} | "
                  f"{chao1_s:>10s} {chao1_fc:>4d} {chao1_fc_pct:>5.1f}% {chao1_mt:>7s}")

print()

# Key insight: AAARS vs Chao1 FC reduction
print("=" * 80)
print("FC REDUCTION: AAARS vs Chao1 under minerich")
print("=" * 80)
for k in K_VALUES:
    for n in N_VALUES:
        subset = [r for r in all_results if r["alloc"] == "minerich"
                  and r["num_mines"] == k and r["num_agents"] == n]
        aaars_stops = [r for r in subset if r.get("aaars__t") is not None]
        chao1_stops = [r for r in subset if r.get("chao1_ci__t") is not None]
        if aaars_stops and chao1_stops:
            aaars_fc = sum(1 for r in aaars_stops if r.get("aaars__recall", 100) < 95.0)
            chao1_fc = sum(1 for r in chao1_stops if r.get("chao1_ci__recall", 100) < 95.0)
            aaars_pct = 100.0 * aaars_fc / len(aaars_stops)
            chao1_pct = 100.0 * chao1_fc / len(chao1_stops)
            delta = chao1_pct - aaars_pct
            print(f"  K={k:<3d} N={n:<2d}: Chao1={chao1_fc}/{len(chao1_stops)} ({chao1_pct:.0f}%) "
                  f"-> AAARS={aaars_fc}/{len(aaars_stops)} ({aaars_pct:.0f}%) "
                  f"  delta={delta:+.0f}pp")
