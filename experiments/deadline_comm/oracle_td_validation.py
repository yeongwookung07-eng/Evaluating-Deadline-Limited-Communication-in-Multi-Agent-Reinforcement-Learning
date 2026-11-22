import gc
from copy import deepcopy
import numpy as np
from .timing import configure_cpu

def main():
 configure_cpu()
 import torch
 from torch.nn import functional as F
 from framework.buffers.episode import Episode
 from .common import load_config,build_policy,output_dir,save_json
 from .adapters.hallway import Hallway
 from .harness import DeadlineCommHarness,rollout
 from .hallway_oracle_check import oracle_actions
 from .replay import CommEpisode
 from .learner import CommLearner,unroll
 from .gates import METHODS
 cfg=load_config('experiments/deadline_comm/configs/timing_rank.yaml'); spec=load_config('experiments/deadline_comm/configs/oracle_td_validation.yaml')
 out=output_dir('oracle_td_validation_001');save_json(out/'manifest.json',dict(status='running',config=cfg,spec=spec))
 results=[];timings=[]
 for seed in spec['seeds']:
  torch.manual_seed(seed);initial=build_policy(cfg);env=Hallway(**cfg['environment']);oracle=[];random=[]
  for ep in range(50):
   random.append(rollout(env,initial,'full',cfg['harness'],seed*10000+ep,False,1.)[0])
   obs,_=env.reset(seed=seed*10000+ep);h=DeadlineCommHarness(initial,'full',cfg['harness'],measured=False)
   oo=[obs];ss=[env.get_state()];mm=[env.available_actions()];aa=[];rr=[];tt=[];xx=[];events=[]
   for t in range(env.episode_length):
    _,_,_,event,_=h.step(obs,mm[-1],t);events.append(event)
    a=oracle_actions(obs,cfg['environment']['group_ids']);h.previous=F.one_hot(torch.as_tensor(a),3).float()[None]
    obs,r,term,trunc,_=env.step(a);aa.append(a);rr.append(r);tt.append(term);xx.append(trunc);oo.append(obs);ss.append(env.get_state());mm.append(env.available_actions())
    if term or trunc:break
   events.append(h.step(obs,mm[-1],len(aa))[3])
   record=CommEpisode(Episode(np.array(oo,np.float32),np.array(ss,np.float32),np.array(mm,bool),np.array(aa),np.array(rr,np.float32),np.array(tt),np.array(xx)),*[np.stack([e[i] for e in events]) for i in range(3)],np.zeros((5,8),np.float32),np.full(5,-1))
   assert sum(rr)==20;oracle.append(record)
  for arm in ['random_only','oracle_mixed']:
   model=deepcopy(initial);learner=CommLearner(model,cfg['learner']);rng=np.random.default_rng(seed)
   before=unroll(model,oracle[0])[0].detach().clone();grads={};loss=[]
   for update in range(spec['updates']):
    batch=[random[int(rng.integers(50))],(oracle if arm=='oracle_mixed' else random)[int(rng.integers(50))]]
    loss.append(learner.train_batch(batch)['td_loss'])
    for name,mod in [('gru',model.agent),('message',model.message),('mixer',model.mixer)]:
     grads[name]=max(grads.get(name,0),sum(float(p.grad.norm()) for p in mod.parameters() if p.grad is not None))
   qchange=float((unroll(model,oracle[0])[0].detach()-before).abs().max())
   learner.save_checkpoint(out/f'{arm}_seed{seed}.pt',{'policy':cfg['policy'],'environment':cfg['environment'],'seed':seed})
   for method in METHODS:
    scores=[rollout(env,model,method,cfg['harness'],910000+ep,False)[3] for ep in range(30)]
    results.append(dict(seed=seed,arm=arm,method=method,mean_return=float(np.mean([s['return'] for s in scores])),successes=sum(s['success'] for s in scores),q_change=qchange,max_gradients=grads,initial_loss=loss[0],final_loss=loss[-1]))
   if arm=='oracle_mixed':
    hc=deepcopy(cfg['harness']);hc['r_post_ns']=450000;hc['link']['bandwidth_bytes_s']=20000
    for method in METHODS:
     for ep in range(2):rollout(env,model,method,hc,920000+ep,True)
    for repeat in range(3):
     for method in METHODS[repeat:]+METHODS[:repeat]:
      rows=[]
      for ep in range(30):
       gc.collect();enabled=gc.isenabled();gc.disable()
       try: rows.extend(rollout(env,model,method,hc,930000+ep,True)[1])
       finally:
        if enabled:gc.enable()
      timings.append(dict(seed=seed,repeat=repeat,method=method,steps=len(rows),misses=sum(r['action_miss'] for r in rows),bytes=float(np.mean([r['wire_bytes'] for r in rows])),mean_ms=float(np.mean([r['step_total_ns'] for r in rows]))/1e6,max_ms=max(r['step_total_ns'] for r in rows)/1e6))
   save_json(out/'learning.json',results);save_json(out/'timing.json',timings)
 save_json(out/'manifest.json',dict(status='completed',config=cfg,spec=spec))
 print(out)
if __name__=='__main__':main()
