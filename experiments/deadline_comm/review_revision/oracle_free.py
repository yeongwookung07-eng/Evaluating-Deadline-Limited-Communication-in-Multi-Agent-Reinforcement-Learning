"""Isolated demonstration-free learning diagnostic; existing learner reused."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from ..common import load_config
from ..timing import configure_cpu
from ..day2.io import atomic_json,JSONLLog
from ..day2.policy import DecayedPolicy
from ..oracle_bc_learner import OracleBCLearner
from ..adapters.hallway import Hallway
from ..reviewer_validation.train_ten_seeds import configure_arm,trajectory,controlled

def worker(s,seed,arm,device,out):
 configure_cpu();torch.manual_seed(seed);out.mkdir(exist_ok=False)
 saved=torch.load(s['metadata_checkpoint'],map_location='cpu',weights_only=True);meta=deepcopy(saved['metadata']);n=meta['environment']['n_agents']
 model=DecayedPolicy(2,2*n,n,3,**meta['policy']);base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
 method,cfg=configure_arm(base,arm,s['training_link'],s['training_deadline_ms']);model.decay_lambda=cfg['decay_lambda'];model.to(device)
 lc=load_config('experiments/deadline_comm/configs/timing_rank.yaml')['learner'];lc.update(learning_rate=5e-4,gamma=.99,grad_clip=10.,target_update_interval=100,bootstrap_timeouts=False,bc_weight=0.)
 learner=OracleBCLearner(model,lc);cpu=deepcopy(model).cpu();env=Hallway(**meta['environment']);rng=np.random.default_rng(seed);pool=[];steps=0;start=time.monotonic();log=JSONLLog(out/'training.jsonl')
 atomic_json(out/'manifest.json',dict(seed=seed,arm=arm,device=device,oracle_count=0,bc_weight=0.,spec=s,initialization='fresh weights; source checkpoint supplies architecture metadata only'))
 try:
  for u in range(s['updates']):
   if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('worker wall limit')
   cpu.load_state_dict(model.state_dict());ep=controlled(trajectory,env,cpu,method,cfg,12000000+seed*10000+u,False,max(.05,1-.95*u/max(1,s['updates']-1)))
   assert not ep.oracle_source
   pool.append(ep);pool=pool[-2000:];steps+=len(ep.episode.actions)
   batch=[deepcopy(pool[i]) for i in rng.choice(len(pool),s['batch_size'],replace=len(pool)<s['batch_size'])]
   metrics=learner.train_batch(batch);assert metrics['bc_loss']==0.
   log.append([dict(update=u+1,online_transitions=steps,return_=float(ep.episode.rewards.sum()),**metrics)])
   if (u+1)%250==0 or u+1==s['updates']:
    meta.update(update=u+1,seed=seed,learner=lc,decay_lambda=model.decay_lambda,oracle_free=True);learner.save_checkpoint(out/f'checkpoint_{u+1:04}.pt',meta)
    print(seed,arm,u+1,steps,flush=True)
  cpu.load_state_dict(model.state_dict());returns=[]
  for e in range(s['evaluation_episodes']):
   ep=controlled(trajectory,env,cpu,method,cfg,13000000+e,False,0.);returns.append(float(ep.episode.rewards.sum()))
  atomic_json(out/'result.json',dict(seed=seed,arm=arm,returns=returns,mean_return=float(np.mean(returns)),online_transitions=steps,updates=s['updates'],timing='fixed readiness model; not measured CPU evaluation'))
 finally:log.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',nargs=3);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
 if a.smoke:s.update(output=s['output']+'_smoke',seeds=[0],updates=2,evaluation_episodes=2)
 root=Path(s['output'])
 if a.worker:
  seed,arm,device=a.worker;worker(s,int(seed),arm,device,root/f'seed{seed}_{arm}');return
 root.mkdir(exist_ok=False);atomic_json(root/'registration.json',s);start=time.monotonic();active=[];pending=[(seed,arm) for seed in s['seeds'] for arm in s['methods']];done=[]
 try:
  while pending or active:
   if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('campaign wall limit')
   for device in (['cpu'] if a.smoke else ['cuda:0','cuda:1']):
    if pending and not any(x[3]==device for x in active):
     seed,arm=pending.pop(0);f=(root/f'seed{seed}_{arm}.log').open('w');cmd=[sys.executable,'-m','experiments.deadline_comm.review_revision.oracle_free','--config',a.config,'--worker',str(seed),arm,device]+(['--smoke'] if a.smoke else [])
     proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));active.append((proc,f,(seed,arm),device))
   for item in active[:]:
    proc,f,job,device=item
    if proc.poll() is not None:
     f.close();active.remove(item)
     if proc.returncode:raise RuntimeError(f'{job} exit={proc.returncode}')
     done.append(job)
   atomic_json(root/'status.json',dict(stage='running',completed=done,running=[x[2] for x in active],pending=len(pending)));time.sleep(2)
  atomic_json(root/'status.json',dict(stage='completed',completed=done))
 except Exception as exc:
  for proc,f,job,device in active:proc.terminate();f.close()
  atomic_json(root/'status.json',dict(stage='failed',error=repr(exc),completed=done));raise
if __name__=='__main__':main()
