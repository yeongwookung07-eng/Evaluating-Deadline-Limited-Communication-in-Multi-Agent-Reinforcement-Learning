"""Batched priorities, causal sender ages and bounded traffic credit."""
import math
import torch


def q_gap(q, available):
    if q.shape != available.shape or q.ndim != 3 or q.shape[-1]<2:
        raise ValueError('Expected matching [B,N,A>=2]')
    legal=available.bool()
    if not legal.any(-1).all(): raise ValueError('No legal action')
    values=q.masked_fill(~legal,-torch.inf).topk(2,dim=-1).values
    return torch.where(legal.sum(-1)>=2,values[...,0]-values[...,1],torch.zeros_like(values[...,0]))


def decay_messages(cache, source, t, lam):
    if not 0<=lam<=1: raise ValueError('Decay lambda must be in [0,1]')
    valid=source>=0
    age=(t-source).clamp_min(0).to(cache.dtype)
    weight=torch.where(valid,torch.pow(torch.as_tensor(lam,dtype=cache.dtype,device=cache.device),age),torch.zeros_like(age))
    return cache*weight[...,None]


def link_cap(deadline_ns,r_post_ns,enc_gate_ns,bandwidth_bytes_s,packet_bytes=48,latency_ns=0):
    window=max(0.,deadline_ns-r_post_ns-enc_gate_ns)
    raw=math.floor(window*1e-9*bandwidth_bytes_s/packet_bytes)
    effective=math.floor(max(0.,window-latency_ns)*1e-9*bandwidth_bytes_s/packet_bytes)
    return max(0,raw),max(0,effective),window


class DACGate:
    def __init__(self,n_agents,alpha=.1,threshold=0.,budget_bytes=120.,packet_bytes=48,burst_packets=3):
        if alpha<0 or budget_bytes<0: raise ValueError('Invalid DAC configuration')
        self.age=torch.zeros(n_agents,dtype=torch.long)
        self.alpha,self.threshold=alpha,threshold
        self.budget_bytes,self.packet_bytes=budget_bytes,packet_bytes
        self.burst_bytes=burst_packets*packet_bytes
        self.credit=0.;self.spent=0;self.steps=0
    def start_step(self):
        self.credit=min(self.burst_bytes,self.credit+self.budget_bytes)
        self.steps+=1
        return min(len(self.age),int(self.credit//self.packet_bytes))
    def scores(self,q,available,use_age=True):
        return q_gap(q,available)+self.alpha*(self.age.to(q.device)[None] if use_age else 0)
    def select(self,scores,k_max):
        # Stable index tie-break is explicit and covered by tests.
        order=torch.argsort(scores[0],descending=True,stable=True)
        eligible=order[scores[0,order]>=self.threshold]
        selected=torch.zeros_like(self.age,dtype=torch.bool)
        selected[eligible[:max(0,int(k_max))]]=True
        return selected
    def commit(self,attempt):
        cost=int(attempt.sum())*self.packet_bytes
        if cost>self.credit+1e-8: raise AssertionError('Traffic credit exceeded')
        self.credit-=cost;self.spent+=cost
        self.age=torch.where(attempt,torch.zeros_like(self.age),self.age+1)
        assert self.spent<=self.steps*self.budget_bytes+1e-8
