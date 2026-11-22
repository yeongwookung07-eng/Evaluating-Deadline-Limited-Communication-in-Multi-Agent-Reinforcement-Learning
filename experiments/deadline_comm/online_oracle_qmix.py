"""Bounded online QMIX with Oracle replay initialization and held-out evaluations."""
import json
from .timing import configure_cpu

def main():
 configure_cpu()
 import torch
 import numpy as np
 from .common import load_config,build_policy,output_dir,manifest,save_json,save_csv
 from .adapters.hallway import Hallway
 from .td_diagnostic_data import generate_data
 from .td_validation_metrics import validate
 from .batched_learner import BatchedCommLearner
 from .harness import rollout
 from .replay import CommReplay
 from .exploration_reward_audit import save_replay
 cfg=load_config('experiments/deadline_comm/configs/timing_rank.yaml');spec=load_config('experiments/deadline_comm/configs/online_oracle_qmix.yaml')
 for k in ['gamma','learning_rate','grad_clip','target_update_interval']:cfg['learner'][k]=spec[k]
 seed=spec['seed'];torch.manual_seed(seed);model=build_policy(cfg);env=Hallway(**cfg['environment'])
 out=output_dir(spec['output_name']);meta=manifest({'base':cfg,'online':spec},'online_oracle_qmix',seed);save_json(out/'manifest.json',meta)
 learner=BatchedCommLearner(model,cfg['learner']);replay=CommReplay(spec['replay_capacity'],seed)
 _,oracle=generate_data(model,env,cfg,600000,50)
 for r in oracle:r.oracle_source=True;replay.insert(r)
 save_replay(out/'oracle_initial.npz',oracle)
 vr,vo=generate_data(model,env,cfg,700000,5);validation=vr+vo;save_replay(out/'validation.npz',validation)
 train=[];evaluations=[];curves=[];best=-float('inf');best_update=0;transitions=0
 def evaluate(update):
  nonlocal best,best_update
  metrics=validate(learner,validation);scores=[]
  for ep in range(30):
   record,_,_,s=rollout(env,model,'full',cfg['harness'],910000+ep,False)
   evaluations.append(dict(update=update,episode=ep,**s));scores.append(s)
  ret=float(np.mean([s['return'] for s in scores]));success=sum(s['success'] for s in scores)
  curves.append(dict(update=update,mean_return=ret,successes=success,**metrics))
  learner.save_checkpoint(out/f'checkpoint_{update:04}.pt',{'policy':cfg['policy'],'environment':cfg['environment'],'seed':seed,'update':update})
  if ret>best:best=ret;best_update=update
  save_csv(out/'evaluations.csv',evaluations);save_csv(out/'validation.csv',curves)
  print(curves[-1],flush=True)
 try:
  evaluate(0)
  with (out/'online_steps.jsonl').open('w') as trace:
   for ep in range(spec['episodes']):
    epsilon=spec['epsilon_start']+(spec['epsilon_end']-spec['epsilon_start'])*ep/(spec['episodes']-1)
    record,_,_,s=rollout(env,model,'full',cfg['harness'],100000+ep,False,epsilon)
    record.oracle_source=False;replay.insert(record);batch=replay.sample(spec['batch_size']);stat=learner.train_batch(batch)
    transitions+=len(record.episode.actions)
    train.append(dict(episode=ep+1,epsilon=epsilon,episode_return=s['return'],success=s['success'],length=s['length'],
      positive_reward_steps=int((record.episode.rewards>0).sum()),negative_reward_steps=int((record.episode.rewards<0).sum()),
      terminal=bool(record.episode.terminated[-1]),truncated=bool(record.episode.truncated[-1]),
      sampled_oracles=sum(getattr(r,'oracle_source',False) for r in batch),**stat))
    for t,r in enumerate(record.episode.rewards):trace.write(json.dumps(dict(episode=ep+1,t=t,reward=float(r),actions=record.episode.actions[t].tolist(),obs=record.episode.observations[t].tolist(),terminated=bool(record.episode.terminated[t]),truncated=bool(record.episode.truncated[t])))+'\n')
    if (ep+1)%200==0 or ep+1==spec['episodes']:
     save_csv(out/'training.csv',train);evaluate(ep+1)
  save_replay(out/'final_replay.npz',list(replay.items))
  saved=torch.load(out/f'checkpoint_{best_update:04}.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model'])
  held=[rollout(env,model,'full',cfg['harness'],980000+ep,False)[3] for ep in range(30)]
  save_csv(out/'holdout.csv',[dict(episode=i,**s) for i,s in enumerate(held)])
  result=dict(best_validation_update=best_update,best_validation_return=best,holdout_return=float(np.mean([s['return'] for s in held])),holdout_successes=sum(s['success'] for s in held),online_episodes=1500,online_transitions=transitions,online_positive_reward_episodes=sum(r['positive_reward_steps']>0 for r in train),online_successes=sum(r['success'] for r in train),target_copies=learner.updates//100)
  result['useful_policy_gate']=best>=10 and result['holdout_return']>=10 and result['holdout_successes']>0
  save_json(out/'summary.json',result);meta.update(status='completed',**result)
 except Exception as exc:meta.update(status='failed',error=repr(exc));raise
 finally:save_json(out/'manifest.json',meta)
if __name__=='__main__':main()
