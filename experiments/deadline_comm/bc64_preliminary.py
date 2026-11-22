"""Three requested preliminary checks on one fixed, partially successful BC64 policy."""
import gc
import json
from copy import deepcopy
from .timing import configure_cpu

def main():
 configure_cpu()
 import torch,numpy as np
 from .common import ROOT,load_config,output_dir,manifest,save_json,save_csv,build_policy
 from .adapters.hallway import Hallway
 from .harness import rollout
 from .gates import METHODS
 from .analyze import analyze
 spec=load_config(ROOT/'configs/bc64_preliminary.yaml');cfg=load_config(ROOT/'configs/timing_rank.yaml');cfg['policy']['hidden_dim']=64
 cfg['harness']['r_post_ns']=spec['r_post_ns'];out=output_dir(spec['output_name'])
 meta=manifest({'base':cfg,'preliminary':spec},'bc64_preliminary',0,spec['checkpoint']);save_json(out/'manifest.json',meta)
 saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True);model=build_policy(cfg);model.load_state_dict(saved['model']);model.eval();env=Hallway(**cfg['environment'])
 def run(method,hc,seed):
  gc.collect();was=gc.isenabled();gc.disable()
  try:return rollout(env,model,method,hc,seed,True)
  finally:
   if was:gc.enable()
 calibration=[];allsteps=[];allepisodes=[];allmessages=[]
 try:
  for method in METHODS:
   for ep in range(2):run(method,cfg['harness'],1300000+ep)
  # Bounded threshold search by bytes only, with endpoints recorded. Non-monotonic
  # response is possible; pick the actual smallest measured budget error.
  for method,key,increasing in [('vbc','q_threshold',True),('tmc','message_threshold',False)]:
   lo,hi=0.,20.;candidates=[]
   for iteration in range(spec['calibration_iterations']+2):
    threshold=lo if iteration==0 else hi if iteration==1 else (lo+hi)/2
    hc=deepcopy(cfg['harness']);hc['gate'][key]=threshold
    bytes_sum=0;step_count=0
    for ep in range(spec['calibration_episodes']):
     _,rows,_,_=run(method,hc,spec['calibration_seed']+ep)
     bytes_sum+=sum(r['wire_bytes'] for r in rows);step_count+=len(rows)
    rate=bytes_sum/step_count;candidates.append(dict(method=method,iteration=iteration,threshold=threshold,mean_wire_bytes=rate,error=abs(rate-spec['target_bytes'])))
    if iteration>=2:
     if (rate<spec['target_bytes'])==increasing:lo=threshold
     else:hi=threshold
   selected=min(candidates,key=lambda r:r['error']);cfg['harness']['gate'][key]=selected['threshold'];calibration.extend(candidates)
   save_json(out/'calibration.json',calibration);print('CALIBRATED',selected,flush=True)
  save_json(out/'frozen_harness.json',cfg['harness'])
  for rep in range(spec['repeats']):
   for condition in spec['conditions']:
    hc=deepcopy(cfg['harness']);hc['deadline_ns']=condition['deadline_ns'];hc['link']['bandwidth_bytes_s']=float(condition['bandwidth_bytes_s']);hc['link']['latency_ns']=condition.get('latency_ns',50000)
    for ep in range(spec['evaluation_episodes']):
     offset=(rep+ep)%5;order=METHODS[offset:]+METHODS[:offset]
     for method in order:
      _,steps,messages,summary=run(method,hc,spec['evaluation_seed']+ep)
      tags=dict(condition=condition['name'],repeat=rep,episode=ep,seed=spec['evaluation_seed']+ep,training_seed=0)
      allsteps.extend([{**tags,**r} for r in steps]);allmessages.extend([{**tags,**r} for r in messages]);allepisodes.append({**tags,**summary})
    print('DONE',rep,condition['name'],flush=True)
  save_csv(out/'steps.csv',allsteps);save_csv(out/'episodes.csv',allepisodes)
  with (out/'messages.jsonl').open('w') as f:
   for r in allmessages:f.write(json.dumps(r)+'\n')
  summary=analyze(out)
  repeat_rows=[]
  for c in spec['conditions']:
   for method in METHODS:
    for rep in range(spec['repeats']):
     ss=[r for r in allsteps if r['condition']==c['name'] and r['method']==method and r['repeat']==rep]
     ee=[r for r in allepisodes if r['condition']==c['name'] and r['method']==method and r['repeat']==rep]
     rate=float(np.mean([r['wire_bytes'] for r in ss]))
     repeat_rows.append(dict(condition=c['name'],method=method,repeat=rep,steps=len(ss),misses=sum(r['action_miss'] for r in ss),mean_return=float(np.mean([r['return'] for r in ee])),successes=sum(r['success'] for r in ee),mean_bytes=rate,target_budget_eligible=abs(rate-120)<=6))
  for g in summary['groups']:
   ss=[r for r in allsteps if r['condition']==g['condition'] and r['method']==g['method']]
   g['target_budget_eligible']=abs(g['mean_wire_bytes']-120)<=6
   g['invalid_cache_fraction']=float(np.mean([r['invalid_cache_fraction'] for r in ss]))
   g['miss_count']=sum(r['action_miss'] for r in ss);g['step_count']=len(ss)
   g['max_step_ms']=max(r['step_total_ns'] for r in ss)/1e6
  save_json(out/'summary.json',summary);save_csv(out/'repeat_metrics.csv',repeat_rows)
  meta.update(status='completed',episodes=len(allepisodes),steps=len(allsteps),r_post_ns=450000,thresholds=cfg['harness']['gate'])
 except Exception as exc:meta.update(status='failed',error=repr(exc));raise
 finally:save_json(out/'manifest.json',meta)
if __name__=='__main__':main()
