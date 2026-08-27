"""Canonical path builders for result files."""
import os

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")


def raw_csv_path(campaign, tier, alloc, cfg_tag):
    d = os.path.join(RESULTS_ROOT, campaign, "raw")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cfg_tag}_{tier}_{alloc}.csv")
