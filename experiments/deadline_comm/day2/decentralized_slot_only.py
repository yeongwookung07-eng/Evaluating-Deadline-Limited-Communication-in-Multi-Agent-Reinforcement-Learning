"""Public rotating slots with score-independent local transmission probability."""
import torch
from .decentralized_gate import DecentralizedDACGate


class DecentralizedSlotOnlyGate(DecentralizedDACGate):
    def __init__(self, n_agents, probability=1., **kwargs):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be in [0, 1]")
        super().__init__(n_agents, **kwargs)
        self.probability = probability

    def scores(self, q, available, use_age=True):
        return torch.zeros(q.shape[:2], device=q.device)

    def select(self, scores, k_max):
        phase = (torch.arange(len(self.age)) + self.steps-1) % len(self.age)
        slots = phase < max(0, min(len(self.age), int(k_max)))
        if self.cached is None:
            self.cached = (torch.full_like(slots, self.probability >= .5) if self.deterministic
                           else torch.rand(len(self.age), generator=self.rng) < self.probability)
        return slots & self.cached
