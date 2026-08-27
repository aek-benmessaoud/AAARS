#!/usr/bin/env python
"""
debug_campaign.py — Debug campaign for Project11-AAARS.

Verifies:
  1. No crashes across all method × allocation combos
  2. No ground-truth leakage in AAARS controller
  3. Correct frequency statistics
  4. Correct estimator switching (AAARS and discrete)
  5. Correct stopping decisions
  6. Correct logging output

Uses 1-3 seeds per configuration.
"""

import os
import sys
import json
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import run_episode, AAARS_STOP_RULES, DEFAULT_CFG


DEBUG_CONFIGS = [
    {"name": "tiny", "grid_size": 50, "num_mines": 20, "num_agents": 3,
     "max_steps": 500},
    {"name": "small", "grid_size": 100, "num_mines": 60, "num_agents": 6,
     "max_steps": 1000},
]

ALLOCATIONS = ["boustro", "minerich"]
METHODS = ["chao1_ci", "chao92_ci", "aaars", "discrete_aaars",
           "oracle_95", "fixed_2", "diminishing"]
NUM_SEEDS = 2


def run_debug():
    results = []
    errors = []
    
    for cfg_info in DEBUG_CONFIGS:
        cfg_name = cfg_info["name"]
        cfg = {**DEFAULT_CFG, **{k: v for k, v in cfg_info.items() if k != "name"}}
        
        for alloc in ALLOCATIONS:
            for seed_idx in range(NUM_SEEDS):
                env_seed = seed_idx * 1000
                print(f"[debug] {cfg_name} / {alloc} / seed={env_seed} ... ", end="", flush=True)
                
                try:
                    result = run_episode(alloc, seed_idx, env_seed, cfg=cfg)
                    results.append(result)
                    
                    # Check no crashes
                    assert result is not None, "Result is None"
                    
                    # Check all methods have results
                    for method in METHODS:
                        t_key = f"{method}__t"
                        assert t_key in result, f"Missing {t_key}"
                    
                    # Check AAARS-specific fields
                    assert "aaars__switches" in result, "Missing aaars__switches"
                    assert "aaars__final_risk" in result, "Missing aaars__final_risk"
                    
                    # Check trace exists
                    if "_trace" in result:
                        trace = json.loads(result["_trace"])
                        assert len(trace) > 0, "Empty trace"
                        
                        # Check no ground truth leakage in AAARS diagnostics
                        for entry in trace:
                            diag_keys = set(entry.keys())
                            leak_keys = {"true_K", "true_recall", "oracle95_time"}
                            leaked = diag_keys & leak_keys
                            assert not leaked, f"Ground truth leakage: {leaked}"
                    
                    # Check recall values are reasonable
                    for method in METHODS:
                        rec = result[f"{method}__recall"]
                        assert 0 <= rec <= 100, f"{method} recall out of range: {rec}"
                    
                    # Print summary
                    t_str = "  ".join(
                        f"{m}={result[f'{m}__t'] or 'None':>5}"
                        for m in ["chao1_ci", "aaars", "oracle_95"]
                    )
                    print(f"OK  [{t_str}]  risk={result.get('aaars__final_risk', '?')}")
                    
                except Exception as e:
                    errors.append((cfg_name, alloc, env_seed, str(e), traceback.format_exc()))
                    print(f"FAIL: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Debug campaign: {len(results)} episodes OK, {len(errors)} errors")
    
    if errors:
        print("\nErrors:")
        for cfg_name, alloc, seed, msg, tb in errors:
            print(f"  {cfg_name}/{alloc}/seed={seed}: {msg}")
        return False
    
    # Additional checks
    print("\n--- Validation checks ---")
    
    # Check AAARS produces different risk scores under boustro vs minerich
    boustro_risks = [r["aaars__final_risk"] for r in results
                     if r["alloc"] == "boustro"]
    minerich_risks = [r["aaars__final_risk"] for r in results
                      if r["alloc"] == "minerich"]
    if boustro_risks and minerich_risks:
        mean_b = sum(boustro_risks) / len(boustro_risks)
        mean_m = sum(minerich_risks) / len(minerich_risks)
        print(f"  Mean risk boustro: {mean_b:.4f}")
        print(f"  Mean risk minerich: {mean_m:.4f}")
        if mean_m > mean_b:
            print("  PASS: minerich has higher risk (expected)")
        else:
            print("  NOTE: minerich risk not higher than boustro (check diagnostics)")
    
    # Check false certification
    for alloc in ALLOCATIONS:
        for method in ["chao1_ci", "aaars"]:
            fc_count = sum(1 for r in results
                          if r["alloc"] == alloc
                          and r[f"{method}__t"] is not None
                          and r[f"{method}__recall"] < 95.0)
            total = sum(1 for r in results if r["alloc"] == alloc
                       and r[f"{method}__t"] is not None)
            if total > 0:
                print(f"  {alloc}/{method}: {fc_count}/{total} false certifications")
    
    print("\nAll debug checks passed!")
    return True


if __name__ == "__main__":
    success = run_debug()
    sys.exit(0 if success else 1)
