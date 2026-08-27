import sys, os, time
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "grid_size": 50, "num_mines": 20, "num_agents": 3, "max_steps": 800, "aaars": {"w_temporal": 0.4, "w_spatial": 0.3, "w_frequency": 0.3, "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}}

# Test 1 episode with 800 steps
t0 = time.perf_counter()
r = run_episode("boustro", 0, 0, cfg=cfg, collect_trace=False)
elapsed = time.perf_counter() - t0
print("800-step episode: %.1f s" % elapsed)
print("aaars_t=%s risk=%s" % (r.get("aaars__t"), r.get("aaars__final_risk")))

# Test 2 episodes
t0 = time.perf_counter()
for seed in [0, 1000]:
    for alloc in ["boustro", "minerich"]:
        r = run_episode(alloc, seed // 1000, seed, cfg=cfg, collect_trace=False)
        print("  %s seed=%d aaars_t=%s" % (alloc, seed, r.get("aaars__t")))
elapsed = time.perf_counter() - t0
print("4 episodes: %.1f s" % elapsed)
print("Estimated full Phase A (10 configs x 20 eps): %.0f min" % (elapsed / 4 * 10 * 20 / 60))
