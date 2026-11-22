"""Frozen-checkpoint, fixed-readiness, matched decay delivery control."""
import argparse,json,time,gc
from pathlib import Path
from copy import deepcopy
import numpy as np
from ..common import load_config
from ..timing import configure_cpu
from ..day2.finetune_comm_aware import load_model
from ..adapters.hallway import Hallway
from ..selection_validation.harness import SelectionHarness
from ..day2.io import atomic_json,JSONLLog

def episode(model,env,method,cfg,reset,decay):
 obs,_=env.reset(seed=reset);h=SelectionHarness(model,method,cfg,reset,False);model.decay_lambda=decay;trace=[];ret=0.
 for t in range(env.episode_length):
  a,r,m=h.step(obs,env.available_actions(),t);obs,reward,term,trunc,info=env.step(a);ret+=reward
  trace.append(dict(t=t,senders=sorted(x['sender'] for x in m if x['accepted']),actions=a.tolist(),reward=float(reward),source=h.source.tolist(),cache=h.cache.tolist(),attempts=r['K_attempt']))
  if term or trunc:break
 return dict(reset=reset,episode_return=ret,trace=trace)
def main(s,smoke=False):
 configure_cpu();out=Path(s['output']+('_smoke' if smoke else ''));out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s)
 base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness'];start=time.monotonic();records=[]
 if smoke:s=dict(s,seeds=[0],conditions=[['slow',5.]],episodes=2)
 atomic_json(out/'status.json',dict(stage='running'))
 try:
  for seed in s['seeds']:
   model,meta=load_model(s['checkpoint'].format(seed=seed));env=Hallway(**meta['environment'])
   for link,deadline in s['conditions']:
    cfg=deepcopy(base);cfg.update(fixed_ready=True,deadline_ns=int(deadline*1e6));cfg['link']['bandwidth_bytes_s']={'slow':20000,'medium':100000}[link]
    for decay in s['decays']:
     paired={}
     for method in s['methods']:
      if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('registered wall budget')
      episodes=[];p=out/f'seed{seed}_{link}_{deadline:g}_lambda{decay}_{method}.jsonl';log=JSONLLog(p)
      try:
       for e in range(s['episodes']):
        was=gc.isenabled();gc.disable()
        try:r=episode(model,env,method,cfg,s['reset_start']+e,decay)
        finally:
         if was:gc.enable()
         gc.collect()
        episodes.append(r);log.append([r])
      finally:log.close()
      paired[method]=episodes
     ref=paired['full']
     for method,eps in paired.items():
      common=equal_sets=equal_actions=equal_cache=0
      for a,b in zip(ref,eps):
       for x,y in zip(a['trace'],b['trace']):
        common+=1;equal_sets+=x['senders']==y['senders'];equal_actions+=x['actions']==y['actions'];equal_cache+=x['cache']==y['cache'] and x['source']==y['source']
      row=dict(seed=seed,condition=f'{link}_{deadline:g}ms',decay=decay,method=method,mean_return=float(np.mean([e['episode_return'] for e in eps])),common_steps=common,same_sets=equal_sets,same_actions=equal_actions,same_cache=equal_cache)
      if smoke and method=='rr_cap':assert equal_sets==equal_cache==equal_actions==common and [e['episode_return'] for e in eps]==[e['episode_return'] for e in ref]
      records.append(row)
     atomic_json(out/'progress.json',records)
  atomic_json(out/'summary.json',dict(records=records,interpretation='fixed readiness; identical checkpoint and decay within each block; not CPU timing evidence'))
  atomic_json(out/'status.json',dict(stage='completed',cells=len(records),seconds=time.monotonic()-start))
 except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(load_config(a.config),a.smoke)
