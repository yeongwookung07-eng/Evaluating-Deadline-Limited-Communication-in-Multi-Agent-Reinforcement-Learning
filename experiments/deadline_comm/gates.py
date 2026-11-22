"""User-defined diagnostic gates; VBC/TMC names do not imply native reproduction."""
import numpy as np
import torch

METHODS = ("full", "periodic", "random", "vbc", "tmc")

class Gate:
    def __init__(self, method, n_agents, seed=0, period=2, p=.5, q_threshold=.1, message_threshold=.1):
        if method not in METHODS or period < 1 or not 0 <= p <= 1:
            raise ValueError("Invalid gate")
        self.method, self.n = method, n_agents
        self.period, self.p = period, p
        self.q_threshold, self.message_threshold = q_threshold, message_threshold
        self.rng = np.random.default_rng(seed)
        self.last = [None] * n_agents
    def decide(self, i, t, message, local_q, available, previous_message=...):
        if self.method == "full": return True
        if self.method == "periodic": return (t+i) % self.period == 0
        if self.method == "random": return bool(self.rng.random() < self.p)
        if self.method == "vbc":
            q = local_q[available.bool()]
            if len(q) < 2: return False
            top = q.topk(2).values
            return bool(top[0]-top[1] <= self.q_threshold)
        previous = self.last[i] if previous_message is ... else previous_message
        return previous is None or bool(torch.linalg.vector_norm(message-previous) >= self.message_threshold)
    def record_attempt(self, i, message):
        # Sender memory updates even when the receiver drops the packet. No ACK.
        self.last[i] = message.detach().clone()
