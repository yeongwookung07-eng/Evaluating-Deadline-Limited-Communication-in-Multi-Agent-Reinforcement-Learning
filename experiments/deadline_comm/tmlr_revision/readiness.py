"""Measured versus common-readiness primary-policy evaluation near packet boundaries."""
import argparse,gc,json,time,os,platform,subprocess
from pathlib import Path
import numpy as np
from ..common import load_config
from ..timing import configure_cpu
from ..day2.io import atomic_json,JSONLLog
from ..day2.finetune_comm_aware import load_model
from ..adapters.hallway import Hallway
from ..selection_validation.harness import SelectionHarness
from .experiments import BASE,setup

def run(s):
 configure_cpu();out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s);rows=[];start=time.monotonic()
 atomic_json(out/'hardware.json',dict(cpu_info=subprocess.check_output(['lscpu'],text=True),platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),scope='single host; no exclusive CPU reservation; measurements are not multi-machine evidence'))
 log=JSONLLog(out/'episodes.jsonl')
 try:
  for seed in s['seeds']:
   for arm in s['checkpoints']:
    model,meta=load_model(BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_medium_2ms_seed{seed}_{arm}/checkpoint_1500.pt');model.eval()
    for d in s['deadlines_ms']:
     for mode in (['measured','fixed'] if seed%2==0 else ['fixed','measured']):
      cfg=setup(['medium',d]);cfg['fixed_ready']=mode=='fixed';eps=[]
      for ep in range(-2,s['episodes']):
       if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('readiness budget')
       gc.collect();enabled=gc.isenabled();gc.disable()
       try:
        env=Hallway(**meta['environment']);obs,_=env.reset(seed=s['reset_base']+ep);h=SelectionHarness(model,arm,cfg,s['reset_base']+ep,True);trace=[];ret=0.
        for t in range(env.episode_length):
         action,r,msg=h.step(obs,env.available_actions(),t);obs,reward,term,trunc,_=env.step(action);ret+=reward
         trace.append(dict(t=t,k=r['K_delivered'],ready_ns=r['ready_ns'],cpu_ns=r['cpu_total_ns'],gate_ns=r['gate_ns'],wire=r['wire_bytes'],miss=r['action_miss'],senders=sorted(m['sender'] for m in msg if m['accepted'])))
         if term or trunc:break
       finally:
        if enabled:gc.enable()
       if ep>=0:
        row=dict(seed=seed,checkpoint_arm=arm,deadline_ms=d,mode=mode,episode=ep,return_=ret,trace=trace);log.append([row]);eps.append(row)
      tr=[x for r in eps for x in r['trace']];row=dict(seed=seed,checkpoint_arm=arm,deadline_ms=d,mode=mode,mean_return=float(np.mean([r['return_'] for r in eps])),mean_k=float(np.mean([r['k'] for r in tr])),mean_wire=float(np.mean([r['wire'] for r in tr])),ready_us=float(np.mean([r['ready_ns']/1000 for r in tr])),cpu_us=float(np.mean([r['cpu_ns']/1000 for r in tr])),gate_us=float(np.mean([r['gate_ns']/1000 for r in tr])),misses=sum(r['miss'] for r in tr),steps=len(tr))
      rows.append(row);atomic_json(out/'progress.json',rows);atomic_json(out/'status.json',dict(stage='running',cells=len(rows)))
  atomic_json(out/'summary.json',dict(records=rows,packet_boundaries_ns={str(d):{str(k):int(d*1e6-450000-50000-k*48/100000*1e9) for k in range(1,6)} for d in s['deadlines_ms']},scope='mode changes both opportunities and subsequent trajectories; fixed mode timing not deployable latency'))
  atomic_json(out/'status.json',dict(stage='completed',cells=len(rows)))
 except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
 finally:log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
 if a.smoke:s.update(output=s['output']+'_smoke_v2',seeds=[0],deadlines_ms=[2.],episodes=2)
 run(s)
