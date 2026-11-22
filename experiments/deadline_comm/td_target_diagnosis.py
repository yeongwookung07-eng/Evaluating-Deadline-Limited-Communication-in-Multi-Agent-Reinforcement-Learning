"""Paired gamma/target-period diagnosis, with fixed validation trajectories."""
import argparse
from copy import deepcopy
import numpy as np
from .timing import configure_cpu

def main():
 configure_cpu()
 import torch
 from .common import load_config,build_policy,output_dir,save_json,save_csv,manifest
 from .adapters.hallway import Hallway
 from .harness import rollout
 from .learner import CommLearner,unroll
 from .td_diagnostic_data import generate_data
 from .exploration_reward_audit import save_replay
 cfg=load_config('experiments/deadline_comm/configs/timing_rank.yaml');spec=load_config('experiments/deadline_comm/configs/td_target_diagnosis.yaml')
 out=output_dir(spec['output_name']);meta=manifest({'base':cfg,'diagnosis':spec},'td_target_diagnosis',0);save_json(out/'manifest.json',meta)
 curves=[];steps=[];scores=[];summaries=[]
 def validate(learner,records):
  pred=[];targets=[];nextqs=[];timeout=[];errors=[];terminal_errors=[]
  with torch.no_grad():
   for record in records:
    ep=record.episode;online=unroll(learner.model,record);target=unroll(learner.target,record)
    for t in range(len(ep.actions)):
     q=learner.model.joint(online[t],torch.as_tensor(ep.actions[t])[None],torch.as_tensor(ep.states[t])[None])
     act=online[t+1].masked_fill(~torch.as_tensor(ep.available_actions[t+1])[None],-torch.inf).argmax(-1)
     nq=learner.target.joint(target[t+1],act,torch.as_tensor(ep.states[t+1])[None])
     y=float(ep.rewards[t])+learner.config['gamma']*(not ep.terminated[t])*nq
     assert q.shape==y.shape==(1,)
     if ep.terminated[t]:assert y.item()==float(ep.rewards[t])
     pred.append(q.item());targets.append(y.item());nextqs.append(nq.item());errors.append((q-y).square().item())
     if ep.truncated[t]:timeout.append(nq.item())
     if ep.terminated[t]:terminal_errors.append((q-y).square().item())
  return dict(validation_loss=float(np.mean(errors)),max_abs_q=max(map(abs,pred)),max_abs_target=max(map(abs,targets)),max_abs_next_q=max(map(abs,nextqs)),timeout_max_abs_next_q=max(map(abs,timeout)) if timeout else None,terminal_mse=float(np.mean(terminal_errors)) if terminal_errors else None)
 try:
  for seed in spec['seeds']:
   torch.manual_seed(seed);initial=build_policy(cfg);env=Hallway(**cfg['environment'])
   torch.save(initial.state_dict(),out/f'initial_seed{seed}.pt')
   random,oracle=generate_data(initial,env,cfg,seed*10000,50)
   vr,vo=generate_data(initial,env,cfg,700000+seed*100,5);validation=vr+vo
   save_replay(out/f'train_seed{seed}.npz',random+oracle);save_replay(out/f'validation_seed{seed}.npz',validation)
   rng=np.random.default_rng(seed);schedule=[(int(rng.integers(50)),int(rng.integers(50))) for _ in range(199)]
   save_json(out/f'schedule_seed{seed}.json',schedule)
   for name,change in spec['conditions'].items():
    lc={**cfg['learner'],**change};model=deepcopy(initial);learner=CommLearner(model,lc);initial_target={k:v.clone() for k,v in learner.target.state_dict().items()}
    curves.append(dict(seed=seed,condition=name,update=0,**validate(learner,validation)))
    for u,(i,j) in enumerate(schedule,1):
     stat=learner.train_batch([random[i],oracle[j]]);steps.append(dict(seed=seed,condition=name,**stat))
     if u%lc['target_update_interval']==0:assert all(torch.equal(v,learner.target.state_dict()[k]) for k,v in model.state_dict().items())
     if name=='B_fixed_target':assert all(torch.equal(v,learner.target.state_dict()[k]) for k,v in initial_target.items())
     if u in spec['validation_updates']:curves.append(dict(seed=seed,condition=name,update=u,**validate(learner,validation)))
    learner.save_checkpoint(out/f'{name}_seed{seed}.pt',{'policy':cfg['policy'],'environment':cfg['environment'],'seed':seed})
    ss=[rollout(env,model,'full',cfg['harness'],910000+ep,False)[3] for ep in range(30)]
    for ep,s in enumerate(ss):scores.append(dict(condition=name,training_seed=seed,episode=ep,**s))
    summaries.append(dict(seed=seed,condition=name,target_copies=199//lc['target_update_interval'],mean_return=float(np.mean([s['return'] for s in ss])),successes=sum(s['success'] for s in ss),final_validation=curves[-1]))
    save_json(out/'summary.json',summaries);save_csv(out/'validation.csv',curves);save_csv(out/'training.csv',steps);save_csv(out/'evaluation.csv',scores)
    print(seed,name,summaries[-1],flush=True)
  meta['status']='completed'
 except Exception as exc:meta.update(status='failed',error=repr(exc));raise
 finally:save_json(out/'manifest.json',meta)
if __name__=='__main__':main()
