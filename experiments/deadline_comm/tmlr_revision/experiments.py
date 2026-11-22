"""Primary-cohort feasible fresh-message and shared-prefix interventions."""
import argparse,json,time,itertools,hashlib,statistics
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from ..common import load_config
from ..timing import configure_cpu
from ..adapters.hallway import Hallway
from ..day2.finetune_comm_aware import load_model
from ..day2.io import atomic_json,JSONLLog
from ..day2.dac import q_gap
from ..selection_validation.harness import SelectionHarness

BASE=Path('experiments/deadline_comm')
def clone(h):
    # EndpointHarness.__new__ is an argument-requiring factory; bypass it when
    # cloning an already constructed inference state, retaining immutable weights.
    result=object.__new__(type(h))
    result.__dict__=deepcopy(h.__dict__,{id(h.model):h.model})
    return result
def setup(cond):
 link,d=cond;cfg=deepcopy(load_config(BASE/'configs/day2_dac_marl_main.yaml')['harness']);cfg.update(fixed_ready=True,deadline_ns=int(d*1e6));cfg['link']['bandwidth_bytes_s']={'slow':20000,'medium':100000}[link];return cfg

def step(env,h,obs,t,forced=None,hold=None):
 with ExitStack() as stack:
  if forced is not None:
   selected=torch.zeros(h.model.n_agents,dtype=torch.bool);selected[list(forced)]=True
   def select(scores,k):
    assert len(forced)<=k;return selected.clone()
   if h.is_dac:stack.enter_context(patch.object(h.dac,'select',side_effect=select))
   else:stack.enter_context(patch.object(h.gate,'decide',side_effect=lambda i,*a,**kw:i in forced))
  if hold is not None:
   original=h.link.schedule
   def schedule(*args,**kwargs):
    r=original(*args,**kwargs)
    for m in r.messages:
     if m['sender']==hold:m.update(accepted=False,drop_reason='diagnostic_withhold')
    return r
   stack.enter_context(patch.object(h.link,'schedule',side_effect=schedule))
  a,row,msg=h.step(obs,env.available_actions(),t)
 gaps=q_gap(h.model.local_q(h.hidden),torch.as_tensor(env.available_actions())[None])[0].tolist()
 nxt,r,term,trunc,_=env.step(a)
 trace=dict(t=t,actions=a.tolist(),reward=r,done=bool(term or trunc),state=env.get_state().tolist(),senders=sorted(m['sender'] for m in msg if m['accepted']),source=h.source[0].tolist(),wire=row['wire_bytes'],payload=32*row['K_delivered'],model_ready_wait_ns=h.cfg['training_ready_ns']+row['comm_model_ns'],gap=gaps)
 return nxt,trace

def suffix(env,h,obs,t,forced=None,hold=None):
 trace=[]
 for u in range(t,env.episode_length):
  obs,r=step(env,h,obs,u,forced if u==t else None,hold if u==t else None);trace.append(r)
  if r['done']:break
 return sum(r['reward'] for r in trace),trace

def switch(h,method):h.method=method;h.dac.method=method

def fresh(s,out):
 logs=JSONLLog(out/'states.jsonl');rows=[];checks=0;start=time.monotonic()
 try:
  for seed in s['seeds']:
   for cond in s['conditions']:
    name=f'{cond[0]}_{cond[1]:g}ms';cfg=setup(cond)
    for arm in s['checkpoints']:
     ck=BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_{name}_seed{seed}_{arm}/checkpoint_1500.pt';model,meta=load_model(ck);model.eval()
     for ep in range(s['episodes']):
      env=Hallway(**meta['environment']);obs,_=env.reset(seed=s['reset_base']+ep);h=SelectionHarness(model,arm,cfg,ep,False)
      for t in range(env.episode_length):
       if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('fresh budget')
       if t in s['sample_steps']:
        age=h.dac.age.tolist();base,bt=suffix(deepcopy(env),clone(h),obs.copy(),t);n0,no=suffix(deepcopy(env),clone(h),obs.copy(),t);assert bt==no;checks+=1
        chosen=bt[0]['senders'];k=len(chosen);variants=[]
        forced_ret,forced_trace=suffix(deepcopy(env),clone(h),obs.copy(),t,chosen);assert forced_trace==bt
        for subset in itertools.combinations(range(model.n_agents),k):
         value,trace=suffix(deepcopy(env),clone(h),obs.copy(),t,subset)
         variants.append(dict(senders=list(subset),return_=value,delta=value-base,action_flip=trace[0]['actions']!=bt[0]['actions']))
        holds=[]
        for i in chosen:
         value,trace=suffix(deepcopy(env),clone(h),obs.copy(),t,hold=i);holds.append(dict(sender=i,return_=value,delta=value-base,action_flip=trace[0]['actions']!=bt[0]['actions']))
        gap=np.array(bt[0]['gap']);aa=np.array(age);best=max(v['return_'] for v in variants);ranking=[]
        for alpha in [0.,.03,.1,.3,1.]:
         scores=gap+alpha*aa;sel=sorted(np.argsort(-scores,kind='stable')[:k].tolist());value=next(v['return_'] for v in variants if v['senders']==sel);ranking.append(dict(alpha=alpha,senders=sel,regret=best-value))
        norm=gap/max(float(np.mean(np.abs(gap))),1e-8)+.1*aa/model.n_agents;sel=sorted(np.argsort(-norm,kind='stable')[:k].tolist());nv=next(v['return_'] for v in variants if v['senders']==sel)
        row=dict(seed=seed,checkpoint_arm=arm,condition=name,episode=ep,t=t,k=k,baseline_return=base,normal_senders=chosen,initial_invalid=float((h.source<0).float().mean()),gap=gap.tolist(),tx_age=age,age_term=(.1*aa).tolist(),variants=variants,withhold=holds,rankings=ranking,normalized_regret=best-nv,reward_scale_rankings={str(c):sorted(np.argsort(-(c*gap+.1*aa),kind='stable')[:k].tolist()) for c in [.1,1.,10.]},range_return=best-min(v['return_'] for v in variants),source_checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
        logs.append([row]);rows.append({key:row[key] for key in ['seed','checkpoint_arm','condition','episode','t','range_return']})
       obs,tr=step(env,h,obs,t)
       if tr['done']:break
      atomic_json(out/'status.json',dict(stage='running',seed=seed,condition=name,arm=arm,episodes_completed=ep+1,states=len(rows),noop_checks=checks))
  atomic_json(out/'summary.json',dict(states=rows,noop_checks=checks,selection='all visited preselected even timesteps; outcome-independent',unit='training seed'))
 finally:logs.close()

def prefix(s,out):
 log=JSONLLog(out/'episodes.jsonl');rows=[];start=time.monotonic()
 try:
  for seed in s['seeds']:
   for cond in s['conditions']:
    name=f'{cond[0]}_{cond[1]:g}ms';cfg=setup(cond)
    for arm in s['checkpoints']:
     ck=BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_{name}_seed{seed}_{arm}/checkpoint_1500.pt';model,meta=load_model(ck);model.eval()
     for ep in range(s['episodes']):
      if time.monotonic()-start>s['wall_hours']*3600:raise TimeoutError('prefix budget')
      reset=s['reset_base']+ep;env=Hallway(**meta['environment']);obs,_=env.reset(seed=reset);h=SelectionHarness(model,'rr_cap',cfg,reset,False)
      originals={}
      for method in ['rr_cap','dac']:
       hh=clone(h);switch(hh,method);r,tr=suffix(deepcopy(env),hh,obs.copy(),0);originals[method]=dict(return_=r,trace=tr)
      pre=[];done=False
      for t in range(env.episode_length):
       obs,tr=step(env,h,obs,t);pre.append(tr);done=tr['done']
       if done or bool((h.source>=0).all()):break
      paired={};warm=bool((h.source>=0).all());survived=warm and not done
      for method in ['rr_cap','dac']:
       hh=clone(h);switch(hh,method)
       r,tr=suffix(deepcopy(env),hh,obs.copy(),len(pre)) if survived else (0.,[])
       paired[method]=dict(suffix_return=r,total_return=sum(x['reward'] for x in pre)+r,trace=tr)
      row=dict(seed=seed,checkpoint_arm=arm,condition=name,episode=ep,originals=originals,prefix=pre,prefix_complete=warm,survived=survived,paired=paired,prefix_wire=sum(x['wire'] for x in pre),prefix_payload=sum(x['payload'] for x in pre),prefix_model_ns=sum(x['model_ready_wait_ns'] for x in pre))
      log.append([row]);rows.append({k:row[k] for k in ['seed','checkpoint_arm','condition','episode','prefix_complete','survived']})
     atomic_json(out/'status.json',dict(stage='running',episodes=len(rows),seed=seed))
  atomic_json(out/'summary.json',dict(episodes=rows,scope='conditional suffix effect among shared-RR-prefix survivors; all original and prefix costs retained'))
 finally:log.close()

def main():
 configure_cpu();p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',choices=['fresh','prefix'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
 if a.smoke:s.update(output=s['output']+'_smoke',seeds=[0],episodes=1,conditions=[['slow',5.],['medium',2.]])
 out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s)
 try:
  with torch.inference_mode():globals()[a.kind](s,out)
  atomic_json(out/'status.json',dict(stage='completed'))
 except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
if __name__=='__main__':main()
