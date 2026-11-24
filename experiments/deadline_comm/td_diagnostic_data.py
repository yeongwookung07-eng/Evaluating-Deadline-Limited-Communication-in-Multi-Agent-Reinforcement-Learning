"""Data generation identical to the retained oracle TD experiment."""
import numpy as np
import torch
from torch.nn import functional as F
from models.shared.buffers.episode import Episode
from .harness import DeadlineCommHarness, rollout
from .hallway_oracle_check import oracle_actions
from .replay import CommEpisode

def generate_data(initial, env, cfg, start_seed, count):
  oracle=[];random=[]
  for ep in range(count):
   random.append(rollout(env,initial,'full',cfg['harness'],start_seed+ep,False,1.)[0])
   obs,_=env.reset(seed=start_seed+ep);h=DeadlineCommHarness(initial,'full',cfg['harness'],measured=False)
   oo=[obs];ss=[env.get_state()];mm=[env.available_actions()];aa=[];rr=[];tt=[];xx=[];events=[]
   for t in range(env.episode_length):
    _,_,_,event,_=h.step(obs,mm[-1],t);events.append(event)
    a=oracle_actions(obs,cfg['environment']['group_ids']);h.previous=F.one_hot(torch.as_tensor(a),3).float()[None]
    obs,r,term,trunc,_=env.step(a);aa.append(a);rr.append(r);tt.append(term);xx.append(trunc);oo.append(obs);ss.append(env.get_state());mm.append(env.available_actions())
    if term or trunc:break
   events.append(h.step(obs,mm[-1],len(aa))[3])
   record=CommEpisode(Episode(np.array(oo,np.float32),np.array(ss,np.float32),np.array(mm,bool),np.array(aa),np.array(rr,np.float32),np.array(tt),np.array(xx)),*[np.stack([e[i] for e in events]) for i in range(3)],np.zeros((5,8),np.float32),np.full(5,-1))
   assert sum(rr)==20;oracle.append(record)
  return random,oracle
