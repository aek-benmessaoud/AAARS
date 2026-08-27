"""Compare 3-component vs 4-component risk configs on same seeds."""
import sys
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

W3 = {"w_temporal": 0.0, "w_spatial": 0.45, "w_frequency": 0.55,
      "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
      "blend_threshold": 0.30, "num_bins": 5}
W4 = {"w_temporal": 0.0, "w_spatial": 0.15, "w_revisit": 0.40,
      "w_frequency": 0.45, "ema_alpha": 0.1, "base_alpha": 0.05,
      "risk_lambda": 1.0, "blend_threshold": 0.30, "num_bins": 5}

def summarize(name, cfg, cache):
    print(f"\n=== {name} ===")
    for alloc in ["boustro", "minerich"]:
        stops = [r for r in cache if r[0]==alloc and r[1]["aaars__t"] is not None]
        n_stop = len(stops)
        fc = sum(1 for _,r in stops if r.get("aaars__recall",100) < 95.0)
        risks = [r.get("aaars__final_risk",0) for a_,r in cache if a_==alloc]
        print(f"  {alloc:10s}: stops={n_stop}/5 FC={fc} risk_mean={sum(risks)/len(risks):.3f}")

# Run both
for name, aaars in [("W3", W3), ("W4", W4)]:
    cache = []
    for alloc in ["boustro", "minerich"]:
        for seed in [10,11,12,13,14]:
            r = run_episode(alloc, seed, seed*1000, cfg={**DEFAULT_CFG, "aaars": aaars}, collect_trace=False)
            cache.append((alloc, r))
    summarize(name, aaars, cache)
