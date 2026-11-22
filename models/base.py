"""Shared recurrent joint-Q contract; algorithms supply value and greedy inference."""
from abc import ABC, abstractmethod
import torch
from torch import nn
from framework.agents.rnn_agent import RNNAgent


class RecurrentJointQ(nn.Module, ABC):
    agent: RNNAgent
    n_actions: int

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.agent.initial_hidden(batch_size)

    def encode(self, obs: torch.Tensor, previous_actions: torch.Tensor,
               hidden: torch.Tensor) -> torch.Tensor:
        return self.agent(obs, previous_actions, hidden)

    @abstractmethod
    def joint_q(self, hidden: torch.Tensor, actions: torch.Tensor,
                state: torch.Tensor | None = None) -> torch.Tensor: ...

    @abstractmethod
    def greedy_actions(self, hidden: torch.Tensor, available: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def coordination_stats(self) -> dict[str, float | int]: ...
