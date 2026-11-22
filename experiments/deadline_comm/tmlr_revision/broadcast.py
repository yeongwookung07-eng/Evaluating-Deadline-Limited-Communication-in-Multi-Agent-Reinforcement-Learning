"""Delivery, decoding and controller use on the completed broadcast checkpoints."""
import argparse,time,json
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from ..common import load_config
from ..timing import configure_cpu
from ..day2.policy import DecayedPolicy
from ..day2.io import atomic_json,JSONLLog
from ..reviewer_validation.hidden_source import HiddenSource
from ..reviewer_validation.train_ten_seeds import configure_arm
from ..reviewer_validation.endpoints import EndpointHarness
from .experiments import clone,step,setup,BASE

def run(s):
 out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s);rows=[];start=time.monotonic();log=JSONLLog(out/'steps.jsonl')
 try:
  for seed in s['seeds']:
   for method in s['methods']:
    model=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix');ck=BASE/f'runs/hidden_mechanism_20261119_001/broadcast_n5_seed{seed}_{method}/final.pt';model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model']);model.eval()
    for cond in s['conditions']:
     name=f'{cond[0]}_{cond[1]:g}ms';_,cfg=configure_arm(setup(cond),method,{'slow':20000,'medium':100000}[cond[0]],cond[1]);cfg['fixed_ready']=True
     actual_method='full' if method in ['ideal_full','no_comm'] else 'dac' if method=='rr_cap' else method
     for sched in ['normal','source_aware']:
      data=[]
      for ep in range(s['episodes']):
       if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('broadcast budget')
       env=HiddenSource();obs,_=env.reset(seed=s['reset_base']+ep);h=EndpointHarness(model,actual_method,cfg,ep,False)
       for t in range(env.episode_length):
        bit=int(env.bits[0]);chosen=None
        if sched=='source_aware':
         _,probe=step(deepcopy(env),clone(h),obs.copy(),t);selected=probe['senders'];k=len(selected)
         chosen=sorted(([0]+[i for i in selected if i!=0])[:k]) if k else None
        obs,tr=step(env,h,obs,t,chosen);got=0 in tr['senders'];msg=model.message(h.hidden)[0,0].tolist()
        r=dict(seed=seed,method=method,condition=name,scheduler=sched,episode=ep,t=t,bit=bit,source_delivered=got,followers_accuracy=tr['reward'],k=len(tr['senders']),reference_accuracy=1. if got else .5,source_message=msg)
        data.append(r);log.append([r])
      # Fit on disjoint reset episodes, evaluate on held-out episodes; ridge fixed before outcomes.
      split=max(1,s['episodes']//2);train=[r for r in data if r['episode']<split];test=[r for r in data if r['episode']>=split]
      x=np.array([r['source_message']+[1.] for r in train]);y=np.array([r['bit'] for r in train]);w=np.linalg.solve(x.T@x+1e-3*np.eye(x.shape[1]),x.T@y)
      pred=np.array([r['source_message']+[1.] for r in test])@w>=.5 if test else []
      def conditional(got):
       rr=[r['followers_accuracy'] for r in data if r['source_delivered']==got];return float(np.mean(rr)) if rr else None
      row=dict(seed=seed,method=method,condition=name,scheduler=sched,q=float(np.mean([r['source_delivered'] for r in data])),accuracy=float(np.mean([r['followers_accuracy'] for r in data])),reference=float(np.mean([r['reference_accuracy'] for r in data])),accuracy_received=conditional(True),accuracy_missing=conditional(False),linear_probe_accuracy=float(np.mean(pred==[r['bit'] for r in test])) if test else None,probe_train_episodes=split,probe_test_episodes=s['episodes']-split)
      assert abs(row['reference']-(.5+.5*row['q']))<1e-10;rows.append(row);atomic_json(out/'progress.json',rows);atomic_json(out/'status.json',dict(stage='running',cells=len(rows)))
  atomic_json(out/'summary.json',dict(records=rows,interpretation='Reference is a specified decoder, not an information-theoretic upper bound. Linear decoding does not guarantee controller use. Source-aware preserves realized slot count at each visited state, not trajectory-level equal loads.'))
  atomic_json(out/'status.json',dict(stage='completed',cells=len(rows)))
 except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
 finally:log.close()
if __name__=='__main__':
 configure_cpu();p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
 if a.smoke:s.update(output=s['output']+'_smoke_v2',seeds=[0],episodes=2,conditions=[['slow',5.]])
 with torch.inference_mode():run(s)
