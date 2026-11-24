"""Communication-conditioned utilities with shared existing GRU and QMIX mixer."""
import torch
from torch import nn
from models.shared.agents.rnn_agent import RNNAgent
from models.qmix.interface import MonotonicMixer

class CommPolicy(nn.Module):
    def __init__(self, obs_dim, state_dim, n_agents, n_actions, hidden_dim=32, message_dim=8, mixer="qmix"):
        super().__init__()
        if mixer not in ("qmix", "vdn"): raise ValueError("Unknown mixer")
        self.n_agents, self.n_actions, self.message_dim = n_agents, n_actions, message_dim
        self.agent = RNNAgent(obs_dim, n_agents, n_actions, hidden_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, message_dim)
        # Same head used with zero communication for VBC's local Q: trained, not a random extra head.
        self.utility = nn.Sequential(nn.Linear(hidden_dim+message_dim+2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,n_actions))
        self.mixer = MonotonicMixer(n_agents,state_dim) if mixer == "qmix" else None
    def initial_hidden(self, b=1): return self.agent.initial_hidden(b)
    def encode(self, obs, previous, hidden): return self.agent(obs,previous,hidden)
    def local_q(self, hidden):
        return self.utility(torch.cat([hidden, hidden.new_zeros(*hidden.shape[:-1], self.message_dim+2)], -1))
    def q(self, hidden, cache, source, t):
        # Broadcast cache [B,sender,M], receiver excludes itself; no free own-message transfer.
        n = self.n_agents
        valid = source >= 0
        mask = valid[:,None,:] & ~torch.eye(n,device=hidden.device,dtype=torch.bool)[None,:,:]
        weights = mask.to(hidden.dtype)
        count = weights.sum(-1,keepdim=True)
        aggregate = torch.einsum("brs,bsm->brm", weights, cache) / count.clamp_min(1)
        age = (t-source).clamp_min(0).to(hidden.dtype)
        mean_age = (weights*age[:,None,:]).sum(-1,keepdim=True)/count.clamp_min(1)
        fraction = count/max(n-1,1)
        return self.utility(torch.cat([hidden,aggregate,torch.log1p(mean_age),fraction],-1))
    def joint(self, q, actions, state):
        chosen = q.gather(-1,actions.long().unsqueeze(-1)).squeeze(-1)
        return self.mixer(chosen,state) if self.mixer is not None else chosen.sum(-1)
