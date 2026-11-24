"""One shared GRU representation network for homogeneous agents."""
import torch
from torch import nn


class RNNAgent(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, n_actions: int,
                 hidden_dim: int = 64, rnn_hidden_dim: int = 64,
                 include_observation: bool = True, include_previous_action: bool = True,
                 include_agent_id: bool = True):
        super().__init__()
        self.n_agents, self.n_actions = n_agents, n_actions
        self.rnn_hidden_dim = rnn_hidden_dim
        self.include_observation = include_observation
        self.include_previous_action = include_previous_action
        self.include_agent_id = include_agent_id
        input_dim = (obs_dim * include_observation + n_actions * include_previous_action
                     + n_agents * include_agent_id)
        if input_dim == 0 or min(hidden_dim, rnn_hidden_dim) < 1:
            raise ValueError("Agent needs an input and positive hidden dimensions")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, rnn_hidden_dim)

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return next(self.parameters()).new_zeros(batch_size, self.n_agents, self.rnn_hidden_dim)

    def forward(self, obs: torch.Tensor, previous_actions: torch.Tensor,
                hidden: torch.Tensor) -> torch.Tensor:
        """obs [B,N,O], previous_actions [B,N,A], hidden/output [B,N,H]."""
        if obs.ndim != 3:
            raise ValueError("Observations must have shape [batch, agents, features]")
        b, n, _ = obs.shape
        if n != self.n_agents:
            raise ValueError("Observation agent count does not match model")
        if previous_actions.shape != (b, n, self.n_actions):
            raise ValueError("Previous actions must have shape [batch, agents, actions]")
        if hidden.shape != (b, n, self.rnn_hidden_dim):
            raise ValueError("Hidden state must have shape [batch, agents, hidden]")
        if previous_actions.device != obs.device or hidden.device != obs.device:
            raise ValueError("Observations, previous actions and hidden state must share a device")
        inputs = []
        if self.include_observation:
            inputs.append(obs)
        if self.include_previous_action:
            inputs.append(previous_actions)
        if self.include_agent_id:
            inputs.append(torch.eye(n, device=obs.device, dtype=obs.dtype).expand(b, -1, -1))
        encoded = self.encoder(torch.cat(inputs, dim=-1))
        return self.gru(encoded.reshape(b * n, -1), hidden.reshape(b * n, -1)).reshape(b, n, -1)
