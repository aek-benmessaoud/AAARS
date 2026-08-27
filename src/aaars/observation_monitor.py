"""
observation_monitor.py — Online observation statistics for AAARS.

Maintains sliding-window detection statistics used by all diagnostics.
Does NOT use ground truth — only fleet-observable beliefs.

Performance: POPCOUNT_LUT computed once; detection history capped at 500;
revisit history capped at 10 per cell.
"""

import numpy as np
from collections import deque

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)],
                          dtype=np.uint8)

MAX_DETECTION_HISTORY = 500
MAX_REVISIT_PER_CELL = 10


class ObservationMonitor:
    """Track online detection statistics from fleet block bits."""

    def __init__(self, window_size=8):
        self.window_size = window_size
        self._reset()

    def _reset(self):
        self.n_det = 0
        self.f1 = 0.0
        self.f2 = 0.0
        self.f3 = 0.0
        self.fk = {}
        self.detection_times = deque(maxlen=MAX_DETECTION_HISTORY)
        self._detection_steps = deque(maxlen=MAX_DETECTION_HISTORY)
        self.revisit_times = {}
        self._step_count = 0

    def update(self, bits_union, step):
        self._step_count = step
        pc = _POPCOUNT_LUT[bits_union]

        self.n_det = int(np.count_nonzero(pc))

        # Frequency counts via bincount (much faster than loop)
        bc = np.bincount(pc.ravel(), minlength=17)
        self.fk = {int(k): int(bc[k]) for k in range(1, 17) if bc[k] > 0}
        self.f1 = float(bc[1])
        self.f2 = float(bc[2])
        self.f3 = float(bc[3])

        # Track detection steps (one entry per step that had detections)
        nz = np.flatnonzero(pc)
        if nz.size > 0 and (not self._detection_steps or
                            self._detection_steps[-1] != step):
            self._detection_steps.append(step)

        # Track cell-level events (capped)
        if nz.size > 0:
            if nz.size > 20:
                rng = np.random.default_rng(step)
                nz = rng.choice(nz, size=20, replace=False)
            gs = bits_union.shape[0]
            for idx in nz:
                r, c = int(idx // gs), int(idx % gs)
                key = (r, c)
                self.detection_times.append((step, key))
                if key not in self.revisit_times:
                    self.revisit_times[key] = deque(maxlen=MAX_REVISIT_PER_CELL)
                self.revisit_times[key].append(step)

    def get_inter_detection_intervals(self):
        if len(self.detection_times) < 2:
            return []
        times = [t for t, _ in self.detection_times]
        return [times[i + 1] - times[i] for i in range(len(times) - 1)]

    def get_revisit_intervals(self):
        intervals = []
        for times_deque in self.revisit_times.values():
            if len(times_deque) >= 2:
                tl = list(times_deque)
                for i in range(len(tl) - 1):
                    intervals.append(tl[i + 1] - tl[i])
        return intervals

    def get_short_revisit_fraction(self, threshold_fraction=0.1):
        intervals = self.get_revisit_intervals()
        if not intervals:
            return 0.0
        arr = np.array(intervals, dtype=float)
        med = np.median(arr)
        if med <= 0:
            return 0.0
        return float(np.mean(arr < threshold_fraction * med))

    def get_frequency_array(self, max_agents=16):
        fk = np.zeros(max_agents + 1, dtype=float)
        for k, v in self.fk.items():
            if k <= max_agents:
                fk[k] = v
        return fk
