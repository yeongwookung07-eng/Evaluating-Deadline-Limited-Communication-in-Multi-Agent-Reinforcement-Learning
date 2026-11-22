"""Double-Q learning on recorded delivery histories, never re-running gates."""
import torch
from torch.nn import functional as F
from framework.learners.td_learner import TDLearner
from .td_targets import td_target


def unroll(model, record):
    device = next(model.parameters()).device
    ep = record.episode
    hidden = model.initial_hidden(1)
    previous = torch.zeros(1, model.n_agents, model.n_actions, device=device)
    cache = torch.as_tensor(record.initial_cache,device=device).unsqueeze(0)
    source = torch.as_tensor(record.initial_source,device=device).unsqueeze(0)
    values = []
    for t, obs in enumerate(ep.observations):
        hidden = model.encode(torch.as_tensor(obs,device=device).unsqueeze(0),previous,hidden)
        message = model.message(hidden)
        delivered = torch.as_tensor(record.delivery_mask[t],device=device).unsqueeze(0)
        cache = torch.where(delivered[:,:,None],message,cache)
        source = torch.where(delivered,torch.full_like(source,t),source)
        values.append(model.q(hidden,cache,source,t))
        if t < len(ep.actions):
            previous = F.one_hot(torch.as_tensor(ep.actions[t],device=device),model.n_actions).float().unsqueeze(0)
    return values

class CommLearner(TDLearner):
    def train_batch(self, records):
        self.model.train()
        device = next(self.model.parameters()).device
        losses = []
        for record in records:
            ep = record.episode
            online = unroll(self.model, record)
            with torch.no_grad(): target = unroll(self.target, record)
            for t in range(len(ep.actions)):
                actions = torch.as_tensor(ep.actions[t],device=device).unsqueeze(0)
                state = torch.as_tensor(ep.states[t],device=device).unsqueeze(0)
                prediction = self.model.joint(online[t],actions,state)
                with torch.no_grad():
                    mask = torch.as_tensor(ep.available_actions[t+1],device=device).unsqueeze(0)
                    chosen = online[t+1].masked_fill(~mask,-torch.inf).argmax(-1)
                    next_state = torch.as_tensor(ep.states[t+1],device=device).unsqueeze(0)
                    expected = td_target(float(ep.rewards[t]), self.target.joint(target[t+1],chosen,next_state), ep.terminated[t], ep.truncated[t], self.config["gamma"], self.config.get("bootstrap_timeouts", True))
                losses.append((prediction-expected).square().mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config["grad_clip"],error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config["target_update_interval"] == 0:
            self.target.load_state_dict(self.model.state_dict())
        return {"td_loss":loss.item(),"grad_norm":float(grad),"updates":self.updates}
