"""Explicit horizon semantics; legacy configurations preserve timeout bootstrap."""
import torch

def td_target(reward, next_q, terminated, truncated, gamma, bootstrap_timeouts=True):
    terminated=torch.as_tensor(terminated,device=next_q.device,dtype=torch.bool)
    truncated=torch.as_tensor(truncated,device=next_q.device,dtype=torch.bool)
    done=terminated if bootstrap_timeouts else terminated | truncated
    # where also avoids terminal 0*inf; gamma=0 is pure immediate-reward regression.
    future=torch.zeros_like(next_q) if gamma==0 else gamma*torch.where(done,torch.zeros_like(next_q),next_q)
    return torch.as_tensor(reward,device=next_q.device,dtype=next_q.dtype)+future
