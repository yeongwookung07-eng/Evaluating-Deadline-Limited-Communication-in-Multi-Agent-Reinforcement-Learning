"""Whole-episode replay preserving exogenous communication delivery events."""
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
import numpy as np
from framework.buffers.episode import Episode

@dataclass
class CommEpisode:
    episode: Episode
    attempt_mask: np.ndarray       # [T+1,N], includes bootstrap observation
    delivery_mask: np.ndarray      # [T+1,N]
    source_step: np.ndarray        # [T+1,N], after each delivery
    initial_cache: np.ndarray      # [N,M], reset cache (zero)
    initial_source: np.ndarray     # [N], -1 at episode start
    def __post_init__(self):
        length, n = self.episode.observations.shape[:2]
        if any(x.shape != (length,n) for x in (self.attempt_mask,self.delivery_mask,self.source_step)):
            raise ValueError("Communication events must cover T+1 observations")
        if np.any(self.delivery_mask & ~self.attempt_mask): raise ValueError("Delivery without attempt")
        if self.initial_cache.shape[0] != n or self.initial_source.shape != (n,): raise ValueError("Invalid initial cache")
        # Full episodes only: arbitrary hidden/cache snapshots would require burn-in.
        if np.any(self.initial_source != -1) or np.any(self.initial_cache != 0): raise ValueError("Require reset cache")
        source = self.initial_source.copy()
        for t in range(length):
            source[self.delivery_mask[t]] = t
            if not np.array_equal(source,self.source_step[t]): raise ValueError("Inconsistent source history")

class CommReplay:
    def __init__(self, capacity=100, seed=0):
        if capacity < 1: raise ValueError("Invalid capacity")
        self.items = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.items)
    def insert(self, episode): self.items.append(deepcopy(episode))
    def sample(self, batch_size):
        ids = self.rng.choice(len(self.items),batch_size,replace=False)
        return [deepcopy(self.items[int(i)]) for i in ids]

class StratifiedReplay:
    """Separate immutable-demo and online pools; exactly 1/4 demo per minibatch.

    Early online samples use replacement until 12 distinct records are available.
    All returned records are copies. Online successes remain in the online stratum.
    """
    def __init__(self, capacity=2000, seed=0, oracle_capacity=50):
        if capacity<1 or oracle_capacity<1: raise ValueError('Invalid capacity')
        self.online=deque(maxlen=capacity)
        self.oracle=deque(maxlen=oracle_capacity)
        self.rng=np.random.default_rng(seed)
    @property
    def items(self): return list(self.oracle)+list(self.online)
    def __len__(self): return len(self.oracle)+len(self.online)
    def insert(self, record):
        (self.oracle if getattr(record,'oracle_source',False) else self.online).append(deepcopy(record))
    def sample(self, batch_size):
        if batch_size<4 or batch_size%4: raise ValueError('Batch must be divisible by four')
        if not self.oracle or not self.online: raise ValueError('Both strata must be populated')
        counts=(batch_size//4,3*batch_size//4);batch=[]
        for pool,count in zip((self.oracle,self.online),counts):
            ids=self.rng.choice(len(pool),count,replace=len(pool)<count)
            batch.extend(deepcopy(pool[int(i)]) for i in ids)
        self.rng.shuffle(batch)
        return batch
