import torch
import numpy as np
from .learner import unroll
from .td_targets import td_target

def validate(learner,records):
 pred=[];targets=[];nextqs=[];timeout=[];errors=[];terminal_errors=[]
 with torch.no_grad():
  for record in records:
   ep=record.episode;online=unroll(learner.model,record);target=unroll(learner.target,record)
   for t in range(len(ep.actions)):
    q=learner.model.joint(online[t],torch.as_tensor(ep.actions[t])[None],torch.as_tensor(ep.states[t])[None])
    act=online[t+1].masked_fill(~torch.as_tensor(ep.available_actions[t+1])[None],-torch.inf).argmax(-1)
    nq=learner.target.joint(target[t+1],act,torch.as_tensor(ep.states[t+1])[None])
    y=td_target(float(ep.rewards[t]),nq,ep.terminated[t],ep.truncated[t],learner.config['gamma'],learner.config.get('bootstrap_timeouts',True))
    assert q.shape==y.shape==(1,)
    if ep.terminated[t] or (ep.truncated[t] and not learner.config.get('bootstrap_timeouts',True)):assert y.item()==float(ep.rewards[t])
    pred.append(q.item());targets.append(y.item());nextqs.append(nq.item());errors.append((q-y).square().item())
    if ep.truncated[t]:timeout.append(nq.item())
    if ep.terminated[t]:terminal_errors.append((q-y).square().item())
 return dict(validation_loss=float(np.mean(errors)),max_abs_q=max(map(abs,pred)),max_abs_target=max(map(abs,targets)),max_abs_next_q=max(map(abs,nextqs)),timeout_max_abs_next_q=max(map(abs,timeout)) if timeout else None,terminal_mse=float(np.mean(terminal_errors)) if terminal_errors else None)
