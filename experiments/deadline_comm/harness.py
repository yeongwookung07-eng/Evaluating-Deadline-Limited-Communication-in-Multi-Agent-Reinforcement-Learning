"""Causal cutoff followed by one action computation; real CPU + virtual link.

All packets attempted consume full wire bytes, even when dropped. Communication
wait ends at cutoff. The transport is a per-step model, not a socket benchmark.
"""
import json
import numpy as np
import torch
from torch.nn import functional as F
from .timing import Timer
from .transport import SerialLink
from .gates import Gate
from .replay import CommEpisode
from models.shared.buffers.episode import Episode

class DeadlineCommHarness:
    def __init__(self, policy, method, config, seed=0, measured=True):
        if config['deadline_ns'] <= 0 or not 0 <= config['r_post_ns'] <= config['deadline_ns']:
            raise ValueError('Require 0 <= R_post <= deadline')
        if next(policy.parameters()).device.type != 'cpu':
            raise ValueError('Harness uses CPU; move a policy copy to CPU')
        self.policy, self.config, self.measured = policy, config, measured
        self.gate = Gate(method,policy.n_agents,seed=seed,**config.get('gate',{}))
        self.method = method
        self.link = SerialLink(**config['link'])
        self.hidden = policy.initial_hidden(1)
        self.previous = torch.zeros(1,policy.n_agents,policy.n_actions)
        self.cache = torch.zeros(1,policy.n_agents,policy.message_dim)
        self.source = torch.full((1,policy.n_agents),-1,dtype=torch.long)

    @torch.inference_mode()
    def step(self, obs, available, t):
        timer = Timer(self.measured)
        start = timer.now()
        with timer.section('encode_ns'):
            inputs = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(available,dtype=torch.bool).unsqueeze(0)
            if not mask.any(-1).all(): raise ValueError('No legal action')
            self.hidden = self.policy.encode(inputs,self.previous,self.hidden)
            messages = self.policy.message(self.hidden)[0]
        attempts, payloads, gate_times, sender_times = [], {}, {}, {}
        serialize_ns = 0
        for i in range(self.policy.n_agents):
            # TMC lookup is cache work, its distance calculation is gate work.
            with timer.section('cache_ns'):
                old = self.gate.last[i]
            before = timer.now()
            with timer.section('gate_ns'):
                q = self.policy.local_q(self.hidden[:,i:i+1])[0,0] if self.method == 'vbc' else None
                send = self.gate.decide(i,t,messages[i],q,mask[0,i],previous_message=old)
                attempts.append(send)
            gate_times[i] = timer.now()-before
            before = timer.now()
            with timer.section('cache_ns'):
                if send:
                    serial_start = timer.now()
                    payloads[i] = messages[i].contiguous().numpy().tobytes()
                    serialize_ns += timer.now()-serial_start
                    self.gate.record_attempt(i,messages[i])
            sender_times[i] = timer.now()-before
        ready = timer.now()-start if self.measured else self.config.get('training_ready_ns',0)
        cutoff = self.config['deadline_ns']-self.config['r_post_ns']
        post_start = timer.now()
        with timer.section('transport_cpu_ns'):
            result = self.link.schedule({i:len(v) for i,v in payloads.items()},self.policy.n_agents,t,ready,cutoff)
        delivered = np.zeros(self.policy.n_agents,dtype=bool)
        receiver_times = {}
        # No candidate action is computed before the final admitted cache exists.
        for row in result.messages:
            i = row['sender']
            before = timer.now()
            with timer.section('cache_ns'):
                if row['accepted']:
                    value = torch.from_numpy(np.frombuffer(payloads[i],dtype=np.float32).copy())
                    self.cache[0,i].copy_(value)
                    self.source[0,i] = t
                    delivered[i] = True
            receiver_times[i] = timer.now()-before
        with timer.section('cache_ns'):
            valid = self.source >= 0
            ages = t-self.source
            cache = self.cache
            source = self.source
        with timer.section('act_ns'):
            q = self.policy.q(self.hidden,cache,source,t)
            actions = q.masked_fill(~mask,-torch.inf).argmax(-1)
            chosen = actions[0].numpy().copy()
            self.previous = F.one_hot(actions,self.policy.n_actions).float()
        finish = timer.now()
        cpu = finish-start
        post_cpu = finish-post_start
        parts = timer.parts
        primary = sum(parts.get(k,0) for k in ('encode_ns','gate_ns','cache_ns','act_ns'))
        total = cpu + result.wait_ns
        # Log construction/file I/O excluded from control boundary; measured separately by runner.
        valid_ages = ages[valid].tolist()
        row = dict(method=self.method,t=t,K_attempt=sum(attempts),K_delivered=int(delivered.sum()),
            attempted_bytes=sum(len(v)+self.link.header_bytes for v in payloads.values()),
            wire_bytes=result.wire_bytes,delivered_bytes=sum(m['wire_bytes'] for m in result.messages if m['accepted']),
            encode_ns=parts.get('encode_ns',0),gate_ns=parts.get('gate_ns',0),cache_ns=parts.get('cache_ns',0),
            act_ns=parts.get('act_ns',0),other_ns=cpu-primary,cpu_total_ns=cpu,
            transport_cpu_ns=parts.get('transport_cpu_ns',0),serialize_ns=serialize_ns,
            post_cpu_ns=post_cpu,gate_by_agent_ns=json.dumps(gate_times),
            sender_cache_by_agent_ns=json.dumps(sender_times),comm_model_ns=result.wait_ns,
            step_total_ns=total,deadline_ns=self.config['deadline_ns'],receive_cutoff_ns=cutoff,
            ready_ns=ready,action_miss=bool(total>self.config['deadline_ns']),drop_count=int(sum(attempts)-delivered.sum()),
            mean_age=float(np.mean(valid_ages)) if valid_ages else None,
            max_age=max(valid_ages) if valid_ages else None,invalid_cache_fraction=float((~valid).float().mean()),
            measured=self.measured)
        for m in result.messages:
            i = m['sender']
            last = int(self.source[0,i])
            m.update(method=self.method,t=t,source_step=t,last_received_step=last if last>=0 else None,
                age=t-last if last>=0 else None,gate_ns=gate_times[i],sender_cache_ns=sender_times[i],
                receiver_cache_ns=receiver_times[i])
        events = (np.asarray(attempts,dtype=bool),delivered,self.source[0].numpy().copy())
        return chosen,row,result.messages,events,q[0].numpy().copy()


def rollout(env, policy, method, config, seed, measured=True, epsilon=0.):
    policy.eval()
    obs,_ = env.reset(seed=seed)
    harness = DeadlineCommHarness(policy,method,config,seed=seed+100000,measured=measured)
    rng = np.random.default_rng(seed+200000)
    observations,states,masks = [obs],[env.get_state()],[env.available_actions()]
    actions,rewards,terms,truncs,steps,message_rows,events = [],[],[],[],[],[],[]
    for t in range(env.episode_length):
        chosen,row,messages,event,_ = harness.step(obs,masks[-1],t)
        if epsilon:
            for i in range(env.n_agents):
                if rng.random()<epsilon: chosen[i] = rng.choice(np.flatnonzero(masks[-1][i]))
            harness.previous = F.one_hot(torch.as_tensor(chosen),env.n_actions).float().unsqueeze(0)
        obs,reward,term,trunc,info = env.step(chosen)
        row['return_increment'] = reward
        actions.append(chosen); rewards.append(reward); terms.append(term); truncs.append(trunc)
        steps.append(row); message_rows.extend(messages); events.append(event)
        observations.append(obs); states.append(env.get_state()); masks.append(env.available_actions())
        if term or trunc: break
    # Explicit bootstrap event, excluded from episode latency/communication totals.
    _,_,_,event,_ = harness.step(obs,masks[-1],len(actions))
    events.append(event)
    ep = Episode(np.asarray(observations,np.float32),np.asarray(states,np.float32),np.asarray(masks,bool),
        np.asarray(actions,np.int64),np.asarray(rewards,np.float32),np.asarray(terms,bool),np.asarray(truncs,bool))
    record = CommEpisode(ep,np.stack([e[0] for e in events]),np.stack([e[1] for e in events]),
        np.stack([e[2] for e in events]),np.zeros((env.n_agents,policy.message_dim),np.float32),np.full(env.n_agents,-1,np.int64))
    summary = dict(method=method,seed=seed,**{'return':float(sum(rewards))},success=info['success'],length=len(actions),
        action_miss_rate=float(np.mean([r['action_miss'] for r in steps])),
        message_drop_rate=sum(r['drop_count'] for r in steps)/max(1,sum(r['K_attempt'] for r in steps)),
        wire_bytes=sum(r['wire_bytes'] for r in steps))
    return record,steps,message_rows,summary
