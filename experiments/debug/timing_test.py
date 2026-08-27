import sys, os, time
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "grid_size": 50, "num_mines": 20, "num_agents": 3, "max_steps": 200}
t0 = time.perf_counter()
r = run_episode("boustro", 0, 0, cfg=cfg, collect_trace=False)
elapsed = time.perf_counter() - t0
print("One episode: %.1f s" % elapsed)
print("aaars_t=%s risk=%s" % (r.get("aaars__t"), r.get("aaars__final_risk")))
print("chao1_t=%s oracle_t=%s" % (r.get("chao1_ci__t"), r.get("oracle_95__t")))

# Estimate total time for discovery
n_evals = 10 + 12 + 1  # phase_a + phase_b + phase_c
n_episodes_per_eval = 10 * 2  # seeds x allocs
total = elapsed * n_evals * n_episodes_per_eval
print("Estimated discovery time: %.0f min" % (total / 60))
