"""Separate persistent-network evaluator, not a replacement for day-one timings.

Control ticks are separated by D_max; missed ticks slip to action completion.
When the radio has work, receive polling lasts until cutoff. Late packets remain
in flight and may refresh a future cache with their original generation step.
Network operations are real Python CPU work; channel service is still simulated.
"""
import numpy as np
import torch
from torch.nn import functional as F
from ..timing import Timer
from ..day2.dac import DACGate,link_cap
from .queued_link import QueuedLink


class QueuedHarness:
    def __init__(self,model,method,cfg,seed,measured=True):
        if method not in ('full','periodic','dac','no_comm'):raise ValueError(method)
        self.model=model;self.method=method;self.cfg=cfg;self.measured=measured
        self.net=QueuedLink(cfg['link']['bandwidth_bytes_s'],cfg['loss'],cfg['retries'],cfg['link']['latency_ns']/1e9,seed)
        self.gate=DACGate(model.n_agents,**cfg['dac']);self.clock=0.;self.pending=[]
        self.hidden=model.initial_hidden(1);self.previous=torch.zeros(1,model.n_agents,model.n_actions)
        self.cache=torch.zeros(1,model.n_agents,model.message_dim);self.source=torch.full((1,model.n_agents),-1,dtype=torch.long)
        model.decay_lambda=cfg['decay_lambda'] if method=='dac' else 1.
    @torch.inference_mode()
    def step(self,obs,available,t):
        timer=Timer(self.measured);start=timer.now();tick=self.clock;cfg=self.cfg
        before=self.net.snapshot();cutoff=tick+(cfg['deadline_ns']-cfg['r_post_ns'])/1e9
        with timer.section('encode_ns'):
            mask=torch.as_tensor(available,dtype=torch.bool)[None]
            self.hidden=self.model.encode(torch.as_tensor(obs,dtype=torch.float32)[None],self.previous,self.hidden)
            messages=self.model.message(self.hidden)[0]
        with timer.section('gate_ns'):
            elapsed=timer.now()-start if self.measured else cfg['training_ready_ns']
            if self.method=='dac':
                budget=self.gate.start_step();scores=self.gate.scores(self.model.local_q(self.hidden),mask)
                _,cap,_=link_cap(cfg['deadline_ns'],cfg['r_post_ns'],elapsed,self.net.bandwidth,latency_ns=cfg['link']['latency_ns'])
                attempted=self.gate.select(scores,min(budget,cap))
            else:
                attempted=torch.tensor([self.method=='full' or (self.method=='periodic' and (i+t)%2==0) for i in range(self.model.n_agents)])
        with timer.section('cache_ns'):
            payload={i:messages[i].contiguous().numpy().tobytes() for i in torch.where(attempted)[0].tolist()}
        if self.method=='dac':
            with timer.section('gate_ns'):
                elapsed=timer.now()-start if self.measured else cfg['training_ready_ns']
                _,cap,_=link_cap(cfg['deadline_ns'],cfg['r_post_ns'],elapsed,self.net.bandwidth,latency_ns=cfg['link']['latency_ns'])
                attempted &= self.gate.select(scores,min(budget,cap));payload={i:p for i,p in payload.items() if attempted[i]};self.gate.commit(attempted)
        ready=timer.now()-start if self.measured else cfg['training_ready_ns'];ready_abs=tick+ready/1e9
        with timer.section('network_cpu_ns'):
            # Do not advance backwards if CPU has missed its receive cutoff.
            self.pending.extend(self.net.advance(max(self.net.time,ready_abs)))
            for i in sorted(payload,key=lambda i:(i-t)%self.model.n_agents):self.net.submit(i,t,len(payload[i])+16,payload[i])
            busy=bool(self.net.active or self.net.queue or self.net.arrivals)
            if cutoff>=self.net.time:self.pending.extend(self.net.advance(cutoff))
            received=[p for p in self.pending if p['arrival_s']<=cutoff]
            self.pending=[p for p in self.pending if p['arrival_s']>cutoff]
        refreshed=0
        with timer.section('cache_ns'):
            for p in received:
                i=p['sender']
                if p['source_step']>self.source[0,i]:
                    self.cache[0,i]=torch.from_numpy(np.frombuffer(p['payload'],np.float32).copy())
                    self.source[0,i]=p['source_step'];refreshed+=1
            ages=(t-self.source)[self.source>=0]
        with timer.section('act_ns'):
            q=self.model.q(self.hidden,self.cache,self.source,t)
            action=q.masked_fill(~mask,-torch.inf).argmax(-1);self.previous=F.one_hot(action,self.model.n_actions).float()
        cpu=timer.now()-start;wait=max(0.,cutoff-ready_abs)*1e9 if busy else 0.;total=cpu+wait
        self.clock=max(tick+cfg['deadline_ns']/1e9,tick+total/1e9)
        after=self.net.snapshot()
        row=dict(t=t,method=self.method,K_attempt=int(attempted.sum()),offered_bytes=48*len(payload),
                 received_payload_bytes=32*len(received),fresh_cache_updates=refreshed,
                 wire_bytes_started=after['wire_bytes_started']-before['wire_bytes_started'],
                 retry_bytes=after['retransmitted_wire_bytes']-before['retransmitted_wire_bytes'],
                 failed_packets=after['failed_packets']-before['failed_packets'],backlog_packets=after['backlog_packets'],
                 mean_age=float(ages.float().mean()) if len(ages) else None,invalid_fraction=float((self.source<0).float().mean()),
                 cpu_ns=cpu,modeled_wait_ns=wait,step_total_ns=total,action_miss=total>cfg['deadline_ns'],
                 received_old_packets=sum(p['source_step']<t for p in received))
        return action[0].numpy().copy(),row
