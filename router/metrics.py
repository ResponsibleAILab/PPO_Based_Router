# =============================
# File: router/metrics.py
# =============================
from __future__ import annotations
import collections
import math

class SlidingWindow:
    def __init__(self, k: int = 50):
        self.k = k
        self.q = collections.deque(maxlen=k)

    def append(self, x: float):
        self.q.append(float(x))

    def mean(self):
        return sum(self.q)/len(self.q) if self.q else None

    def percentile(self, p: float):
        if not self.q: return None
        arr = sorted(self.q)
        idx = int(max(0, min(len(arr) - 1, round((p/100.0)*(len(arr)-1)))))
        return arr[idx]

    def queue_estimate(self):
        if not self.q: return 0.0
        m = self.mean()
        return sum(1 for x in self.q if x > m) / len(self.q)