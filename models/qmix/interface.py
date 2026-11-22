"""QMIX recurrent utilities and state-conditioned monotonic mixer."""
import torch
from torch import nn
from torch.nn import functional as F

from framework.agents.rnn_agent import RNNAgent
from models.base import RecurrentJointQ


class MonotonicMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32, hypernet_embed_dim=64):
        super().__init__()
        if min(n_agents, state_dim, mixing_embed_dim, hypernet_embed_dim) < 1:
            raise ValueError("QMIX dimensions must be positive")
        self.n_agents, self.state_dim, self.embed_dim = n_agents, state_dim, mixing_embed_dim
        self.hyper_w_1 = nn.Sequential(nn.Linear(state_dim, hypernet_embed_dim), nn.ReLU(),
                                      nn.Linear(hypernet_embed_dim, n_agents*mixing_embed_dim))
        self.hyper_w_final = nn.Sequential(nn.Linear(state_dim, hypernet_embed_dim), nn.ReLU(),
                                          nn.Linear(hypernet_embed_dim, mixing_embed_dim))
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.value = nn.Sequential(nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(),
                                   nn.Linear(mixing_embed_dim, 1))

    def forward(self, agent_q, state):
        if agent_q.ndim != 2 or agent_q.shape[1] != self.n_agents:
            raise ValueError("agent_q must have shape [batch, agents]")
        if state.shape != (agent_q.shape[0], self.state_dim):
            raise ValueError("state must have shape [batch, state_dim]")
        batch = agent_q.shape[0]
        w1 = self.hyper_w_1(state).abs().reshape(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(state).reshape(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_q.reshape(batch, 1, self.n_agents), w1) + b1)
        final = self.hyper_w_final(state).abs().reshape(batch, self.embed_dim, 1)
        return (torch.bmm(hidden, final) + self.value(state).reshape(batch, 1, 1)).reshape(batch)


class QMIX(RecurrentJointQ):
    def __init__(self, obs_dim, state_dim, n_agents, n_actions, agent_config,
                 mixing_embed_dim=32, hypernet_embed_dim=64):
        super().__init__()
        self.n_agents, self.n_actions = n_agents, n_actions
        self.agent = RNNAgent(obs_dim, n_agents, n_actions, **agent_config)
        self.utility = nn.Sequential(nn.Linear(self.agent.rnn_hidden_dim, self.agent.rnn_hidden_dim),
                                     nn.ReLU(), nn.Linear(self.agent.rnn_hidden_dim, n_actions))
        self.mixer = MonotonicMixer(n_agents, state_dim, mixing_embed_dim, hypernet_embed_dim)

    def individual_q(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] != self.n_agents:
            raise ValueError("hidden must have shape [batch, agents, features]")
        return self.utility(hidden)

    def joint_q(self, hidden, actions, state=None):
        if state is None:
            raise ValueError("QMIX joint_q requires centralized state during training")
        utilities = self.individual_q(hidden)
        chosen = utilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        return self.mixer(chosen, state)

    def greedy_actions(self, hidden, available):
        utilities = self.individual_q(hidden).masked_fill(~available.bool(), -torch.inf)
        if not available.bool().any(-1).all():
            raise ValueError("Every agent needs at least one available action")
        return utilities.argmax(-1)

    def coordination_stats(self):
        return {"active_factors": self.n_agents, "pairwise_factors": 0,
                "mean_factor_order": 1.0, "maximum_factor_order": 1,
                "max_sum_iterations": 0, "message_count_per_inference": 0,
                "estimated_payoff_entries_reduced": 0,
                "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
                "centralized_mixer": 1}


def build_qmix(**kwargs):
    return QMIX(**kwargs)
