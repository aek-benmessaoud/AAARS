import sys, os, time
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

# Tiny config for discovery: 30x30 grid, 10 mines, 2 agents, 300 steps
cfg = {**DEFAULT_CFG, "grid_size": 30, "num_mines": 10, "num_agents": 2, "max_steps": 300,
       "aaars": {"w_temporal": 0.4, "w_spatial": 0.3, "w_frequency": 0.3, "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}}

t0 = time.perf_counter()
r = run_episode("boustro", 0, 0, cfg=cfg, collect_trace=False)
elapsed = time.perf_counter() - t0
print("300-step tiny episode: %.1f s" % elapsed)
print("aaars_t=%s risk=%s" % (r.get("aaars__t"), r.get("aaars__final_risk")))

# 20 episodes
t0 = time.perf_counter()
count = 0
for seed in range(0, 5000, 1000):
    for alloc in ["boustro", "minerich"]:
        r = run_episode(alloc, seed // 1000, seed, cfg=cfg, collect_trace=False)
        count += 1
elapsed = time.perf_counter() - t0
print("%d episodes: %.1f s (%.1f s each)" % (count, elapsed, elapsed/count))
print("Estimated Phase A (10 configs x 10 seeds x 2 allocs = 200 eps): %.0f min" % (elapsed/count * 200 / 60))
