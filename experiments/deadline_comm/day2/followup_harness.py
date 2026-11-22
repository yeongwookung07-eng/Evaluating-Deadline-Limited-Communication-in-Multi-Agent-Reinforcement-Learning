"""One day-two loop, with all methods charged for their actual computations."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from ..gates import Gate
from ..transport import SerialLink
from ..timing import Timer
from .dac import DACGate,q_gap,link_cap
from .decentralized_gate import DecentralizedDACGate

METHODS=('full','periodic','random','vbc','vbc_batched','tmc','dac','dac_decentralized')
ABLATIONS=('dac','dac_no_cap','dac_no_aoi','dac_no_decay')

class Day2Harness:
    def __init__(self,model,method,cfg,seed,measured=True):
        if method not in METHODS+ABLATIONS:raise ValueError(method)
        self.model,self.method,self.cfg,self.measured=model,method,cfg,measured
        self.is_dac=method.startswith('dac')
        self.gate=Gate('full' if self.is_dac else 'vbc' if method=='vbc_batched' else method,model.n_agents,seed=seed,**cfg['gate'])
        self.link=SerialLink(**cfg['link'])
        packet=4*model.message_dim+self.link.header_bytes
        if packet!=48:raise ValueError('Registered packet size must be48 bytes')
        self.dac=DACGate(model.n_agents,**cfg['dac'])
        if method == 'dac_decentralized':
            self.dac=DecentralizedDACGate(model.n_agents,seed=seed,**cfg['dac'])
        self.hidden=model.initial_hidden(1);self.previous=torch.zeros(1,model.n_agents,model.n_actions)
        self.cache=torch.zeros(1,model.n_agents,model.message_dim);self.source=torch.full((1,model.n_agents),-1,dtype=torch.long)
        self.model.decay_lambda=cfg['decay_lambda'] if self.is_dac and method!='dac_no_decay' else 1.
    @torch.inference_mode()
    def step(self,obs,available,t):
        timer=Timer(self.measured);start=timer.now()
        with timer.section('encode_ns'):
            inputs=torch.as_tensor(obs,dtype=torch.float32)[None];mask=torch.as_tensor(available,dtype=torch.bool)[None]
            self.hidden=self.model.encode(inputs,self.previous,self.hidden);message=self.model.message(self.hidden)[0]
        with timer.section('cache_ns'):
            tx_age_before=self.dac.age.clone() if self.is_dac else None
            refs=list(self.gate.last)
        with timer.section('gate_ns'):
            if self.is_dac:
                budget_cap=self.dac.start_step()
                scores=self.dac.scores(self.model.local_q(self.hidden),mask,self.method!='dac_no_aoi')
                elapsed=self.cfg['training_ready_ns'] if self.method=='dac_decentralized' else timer.now()-start if self.measured else self.cfg['training_ready_ns']
                raw_cap,effective_cap,window=link_cap(self.cfg['deadline_ns'],self.cfg['r_post_ns'],elapsed,self.link.bandwidth,latency_ns=self.link.latency_ns)
                kmax=budget_cap if self.method=='dac_no_cap' else min(budget_cap,effective_cap)
                attempt=self.dac.select(scores,kmax)
            elif self.method=='vbc_batched':
                gaps=q_gap(self.model.local_q(self.hidden),mask)[0]
                attempt=(gaps<=self.gate.q_threshold)&(mask[0].sum(-1)>1)
            else:
                selected=[]
                for i in range(self.model.n_agents):
                    local=self.model.local_q(self.hidden[:,i:i+1])[0,0] if self.method=='vbc' else None
                    selected.append(self.gate.decide(i,t,message[i],local,mask[0,i],previous_message=refs[i]))
                attempt=torch.tensor(selected,dtype=torch.bool)
        serialize=0;payload={}
        with timer.section('cache_ns'):
            for i in torch.where(attempt)[0].tolist():
                tick=timer.now();payload[i]=message[i].contiguous().numpy().tobytes();serialize+=timer.now()-tick
        # Recheck after packing; discarded candidate packing remains charged.
        with timer.section('gate_ns'):
            if self.is_dac:
                cap_elapsed=self.cfg['training_ready_ns'] if self.method=='dac_decentralized' else timer.now()-start if self.measured else self.cfg['training_ready_ns']
                raw_cap,effective_cap,window=link_cap(self.cfg['deadline_ns'],self.cfg['r_post_ns'],cap_elapsed,self.link.bandwidth,latency_ns=self.link.latency_ns)
                kmax=budget_cap if self.method=='dac_no_cap' else min(budget_cap,effective_cap)
                attempt &= self.dac.select(scores,kmax)
                payload={i:v for i,v in payload.items() if attempt[i]}
        with timer.section('cache_ns'):
            if self.is_dac:self.dac.commit(attempt)
            for i in payload:self.gate.record_attempt(i,message[i])
        ready=timer.now()-start if self.measured else self.cfg['training_ready_ns']
        post_start=timer.now()
        result=self.link.schedule({i:len(v) for i,v in payload.items()},self.model.n_agents,t,ready,self.cfg['deadline_ns']-self.cfg['r_post_ns'])
        delivered=torch.zeros_like(attempt)
        with timer.section('cache_ns'):
            for row in result.messages:
                i=row['sender']
                if row['accepted']:
                    self.cache[0,i].copy_(torch.from_numpy(np.frombuffer(payload[i],np.float32).copy()));self.source[0,i]=t;delivered[i]=True
            ages=(t-self.source)[self.source>=0].tolist()
        with timer.section('act_ns'):
            q=self.model.q(self.hidden,self.cache,self.source,t)
            action=q.masked_fill(~mask,-torch.inf).argmax(-1);chosen=action[0].numpy().copy()
            self.previous=F.one_hot(action,self.model.n_actions).float()
        finish=timer.now();cpu=finish-start;parts=timer.parts;total=cpu+result.wait_ns
        row=dict(method=self.method,t=t,K_attempt=int(attempt.sum()),K_delivered=int(delivered.sum()),wire_bytes=result.wire_bytes,
            attempted_bytes=result.wire_bytes,delivered_bytes=sum(r['wire_bytes'] for r in result.messages if r['accepted']),
            encode_ns=parts.get('encode_ns',0),gate_ns=parts.get('gate_ns',0),cache_ns=parts.get('cache_ns',0),act_ns=parts.get('act_ns',0),
            other_ns=cpu-sum(parts.values()),cpu_total_ns=cpu,serialize_ns=serialize,post_cpu_ns=finish-post_start,comm_model_ns=result.wait_ns,
            step_total_ns=total,deadline_ns=self.cfg['deadline_ns'],r_post_ns=self.cfg['r_post_ns'],ready_ns=ready,
            action_miss=total>self.cfg['deadline_ns'],drop_count=int(attempt.sum()-delivered.sum()),
            mean_age=float(np.mean(ages)) if ages else None,max_age=max(ages) if ages else None,
            invalid_cache_fraction=float((self.source<0).float().mean()),
            k_link_raw=raw_cap if self.is_dac else None,k_link_effective=effective_cap if self.is_dac else None,
            k_budget=budget_cap if self.is_dac else None,k_max=kmax if self.is_dac else None,
            cap_elapsed_ns=cap_elapsed if self.is_dac else None,
            tx_age_before=json.dumps(tx_age_before.tolist()) if self.is_dac else None,
            tx_age_after=json.dumps(self.dac.age.tolist()) if self.is_dac else None)
        for r in result.messages:
            i=r['sender'];last=int(self.source[0,i]);r.update(method=self.method,t=t,last_received_step=last if last>=0 else None,age=t-last if last>=0 else None)
        return chosen,row,result.messages


def rollout(env,model,method,cfg,seed,measured=True):
    model.eval();obs,_=env.reset(seed=seed);h=Day2Harness(model,method,cfg,seed+100000,measured)
    rows=[];messages=[];total=0
    for t in range(env.episode_length):
        action,row,msg=h.step(obs,env.available_actions(),t)
        obs,reward,term,trunc,info=env.step(action)
        row['return_increment']=reward;rows.append(row);messages.extend(msg);total+=reward
        if term or trunc:break
    return rows,messages,dict(method=method,seed=seed,episode_return=total,success=info['success'],length=len(rows))
