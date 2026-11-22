"""Identical admission and aggregation; vary only sender ranking and CPU cost."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
import numpy as np
from ..timing import configure_cpu
from ..common import load_config
from ..day2.finetune_comm_aware import load_model
from ..adapters.hallway import Hallway
from ..day2.io import atomic_json,CSVLog,JSONLLog
from ..day2.aggregate_day2 import measure
from ..training_controls.run import controlled
from ..reviewer_validation.stats import paired_bootstrap
from .harness import SelectionHarness


def rollout(env,model,method,cfg,seed):
 obs,_=env.reset(seed=seed);h=SelectionHarness(model,method,cfg,seed,True);rows=[];msgs=[];ret=0.;first=np.full(env.n_agents,-1);streak=np.zeros(env.n_agents,int);longest=streak.copy()
 for t in range(env.episode_length):
  action,row,messages=h.step(obs,env.available_actions(),t)
  got={r['sender'] for r in messages if r['accepted']}
  for i in range(env.n_agents):
   if i in got:
    if first[i]<0:first[i]=t
    streak[i]=0
   else:streak[i]+=1
  longest=np.maximum(longest,streak)
  row['source_steps']=json.dumps(h.source[0].tolist());row['actions']=json.dumps(action.tolist())
  obs,reward,term,trunc,info=env.step(action);row['reward']=reward;ret+=reward;rows.append(row);msgs.extend(messages)
  if term or trunc:break
 return rows,msgs,dict(episode_return=ret,method=method,first_receive=json.dumps(first.tolist()),longest_missing_streak=json.dumps(longest.tolist()),length=len(rows))


def main(spec,smoke=False):
 configure_cpu();out=Path(spec['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',dict(spec=spec,smoke=smoke))
 base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness'];records=[];start=time.monotonic()
 for cohort,template in spec['checkpoints'].items():
  for seed in spec['seeds']:
   model,meta=load_model(template.format(seed=seed));env=Hallway(**meta['environment'])
   for link,deadline in spec['conditions']:
    for mode in spec['modes']:
     for method in spec['methods']:
      if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Selection evaluation budget')
      cfg=deepcopy(base);cfg['fixed_ready']=mode=='fixed_ready';cfg['link']['bandwidth_bytes_s']=spec['links'][link];cfg['deadline_ns']=int(deadline*1e6)
      dest=out/f'{cohort}_seed{seed}_{link}_{deadline:g}ms_{mode}_{method}';dest.mkdir()
      for ep in range(2):controlled(rollout,env,model,method,cfg,10000000+ep)
      sl,el,ml=CSVLog(dest/'steps.csv'),CSVLog(dest/'episodes.csv'),JSONLLog(dest/'messages.jsonl');steps=[];episodes=[];messages=[]
      try:
       for ep in range(spec['episodes']):
        ss,mm,ee=controlled(rollout,env,model,method,cfg,10100000+ep)
        sl.append([dict(episode=ep,**r) for r in ss]);el.append([dict(episode=ep,**ee)]);ml.append([dict(episode=ep,**r) for r in mm])
        steps.extend(ss);episodes.append(ee);messages.extend(mm)
       row=dict(cohort=cohort,seed=seed,condition=f'{link}_{deadline:g}ms',mode=mode,method=method,**measure(episodes,steps))
       row.update(cpu_mean_us=statistics.mean(r['cpu_total_ns']/1000 for r in steps),gate_mean_us=statistics.mean(r['gate_ns']/1000 for r in steps),mean_K=statistics.mean(r['K_attempt'] for r in steps))
       row['sender_attempts']=json.dumps([sum(r['sender']==i for r in messages) for i in range(env.n_agents)])
       row['sender_deliveries']=json.dumps([sum(r['sender']==i and r['accepted'] for r in messages) for i in range(env.n_agents)])
       records.append(row);atomic_json(dest/'summary.json',row);atomic_json(out/'progress.json',records)
      finally:sl.close();el.close();ml.close()
 ci=[]
 for cohort,condition,mode in sorted({(r['cohort'],r['condition'],r['mode']) for r in records}):
  for other in [m for m in spec['methods'] if m!='dac']:
   a={r['seed']:r['mean_return'] for r in records if (r['cohort'],r['condition'],r['mode'],r['method'])==(cohort,condition,mode,'dac')}
   b={r['seed']:r['mean_return'] for r in records if (r['cohort'],r['condition'],r['mode'],r['method'])==(cohort,condition,mode,other)}
   ci.append(dict(cohort=cohort,condition=condition,mode=mode,contrast='dac_minus_'+other,**paired_bootstrap(a,b)))
 for name,rows in [('by_seed.csv',records),('paired_ci.csv',ci)]:
  log=CSVLog(out/name);log.append(rows);log.close()
 atomic_json(out/'summary.json',dict(records=records,paired_ci=ci))
 lines=['# Equal-cap selector results','', 'Primary contrast: DAC minus Round-Robin. Same checkpoint, credit, aggregation and reset seeds. Measured mode includes computation effects; fixed-readiness mode isolates selection under a common link window and is not a physical latency result. n=3 intervals are descriptive.','', '|Cohort|Condition|Mode|Contrast|Mean difference|95% paired CI|','|---|---|---|---|---|---|']
 for r in ci:lines.append(f"|{r['cohort']}|{r['condition']}|{r['mode']}|{r['contrast']}|{r['mean_difference']:.3f}|[{r['ci95_low']:.3f},{r['ci95_high']:.3f}]|")
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(out/'status.json',dict(stage='completed',cells=len(records)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
 if a.smoke:s.update(output=s['output']+'_smoke',seeds=[0],conditions=[['slow',5.]],episodes=2,checkpoints={'full2k':s['checkpoints']['full2k']})
 main(s,a.smoke)
