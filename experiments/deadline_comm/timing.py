"""CPU evaluation timing only; network latency is virtual, never slept."""
import os
from time import perf_counter_ns
from contextlib import contextmanager


def configure_cpu():
    # CLI calls this before importing policy/torch.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

class Timer:
    def __init__(self, enabled=True):
        self.enabled, self.parts = enabled, {}
    def now(self):
        return perf_counter_ns() if self.enabled else 0
    @contextmanager
    def section(self, key):
        start = self.now()
        yield
        self.parts[key] = self.parts.get(key, 0) + self.now() - start
