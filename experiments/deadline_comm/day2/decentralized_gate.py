"""Local decisions within public rotating slots, with no score exchange."""
import math
import torch
from .dac import DACGate, q_gap


class DecentralizedDACGate(DACGate):
    def __init__(self, n_agents, seed=0, deterministic=False, temperature=4., **kwargs):
        super().__init__(n_agents, **kwargs)
        self.rng = torch.Generator().manual_seed(seed)
        self.deterministic = deterministic
        self.temperature = temperature
        self.cached = None

    def start_step(self):
        # Public cumulative schedule does not depend on other agents' attempts.
        self.steps += 1
        self.cached = None
        self.credit += self.budget_bytes
        packets = math.floor(self.steps*self.budget_bytes/self.packet_bytes)
        previous = math.floor((self.steps-1)*self.budget_bytes/self.packet_bytes)
        return min(len(self.age), packets-previous)

    def select(self, scores, k_max):
        n = len(self.age)
        phase = (torch.arange(n) + self.steps-1) % n
        slots = phase < max(0, min(n, int(k_max)))
        if self.cached is None:
            probability = torch.sigmoid(self.temperature*(scores[0]-self.threshold))
            self.cached = (probability >= .5 if self.deterministic else
                           torch.rand(n, generator=self.rng) < probability)
        return slots & self.cached
