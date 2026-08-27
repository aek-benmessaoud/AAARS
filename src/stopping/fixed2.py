"""Fixed-2 stopping rule: stop when every cell in the mission area has been swept >= 2 times."""

def fixed_2_stop(env):
    """Check if the entire mission domain has been scanned at least 2 times.

    Args:
        env: MineGridEnv instance

    Returns:
        bool: True if all domain cells have scan_count >= 2
    """
    sc = env.fleet_scan_max()
    dom = env.fleet_area_domain()
    if not dom.any():
        return True
    return bool(sc[dom].min() >= 2)
