"""Paired cache-payload interventions; fixed readiness, no training or timing claims."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import numpy as np
import torch
from ..common import load_config
from ..timing import configure_cpu
from ..adapters.hallway import Hallway
from ..day2.finetune_comm_aware import load_model
from ..day2.io import atomic_json, JSONLLog
from ..selection_validation.harness import SelectionHarness
from ..day2.dac import q_gap


def suffix(env, h, obs, t, intervention=None):
    total=0.;first_action=None;diagnostic={};trace=[]
    original=h.model.q
    tx_age_before=h.dac.age.clone()
    available=torch.as_tensor(env.available_actions())[None]
    def changed(hidden,cache,source,step):
        if intervention is not None:
            sender,kind=intervention
            assert int(source[0,sender])>=0
            before=cache.clone()
            if kind=='zero':cache[0,sender].zero_()
            elif kind=='negate':cache[0,sender].neg_()
            else:raise ValueError(kind)
            keep=[i for i in range(h.model.n_agents) if i!=sender]
            assert torch.equal(cache[:,keep],before[:,keep])
            if kind=='negate':assert torch.equal(cache.norm(dim=-1),before.norm(dim=-1))
        gaps=q_gap(h.model.local_q(hidden),available)[0]
        diagnostic.update(source=source[0].tolist(),norms=cache[0].norm(dim=-1).tolist(),q_gap=gaps.tolist(),priority=(gaps+h.dac.alpha*tx_age_before).tolist())
        return original(hidden,cache,source,step)
    for step in range(t,env.episode_length):
        if step==t:
            with patch.object(h.model,'q',side_effect=changed):a,_,_=h.step(obs,env.available_actions(),step)
            first_action=a.copy()
        else:a,_,_=h.step(obs,env.available_actions(),step)
        obs,r,term,trunc,_=env.step(a);total+=r;trace.append((a.tolist(),r,term,trunc,env.get_state().tolist()))
        if term or trunc:break
    return total,first_action,trace,diagnostic


def run(spec):
    configure_cpu();out=Path(spec['output']);out.mkdir(exist_ok=False)
    atomic_json(out/'registration.json',spec);log=JSONLLog(out/'branches.jsonl');records=[];checks=0;start=time.monotonic()
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    try:
        for seed in spec['seeds']:
            model,meta=load_model(spec['checkpoint'].format(seed=seed));model.eval()
            for link,deadline in spec['conditions']:
                for method in spec['methods']:
                    cfg=deepcopy(base);cfg.update(fixed_ready=True,deadline_ns=int(deadline*1e6));cfg['link']['bandwidth_bytes_s']=spec['links'][link]
                    for episode in range(spec['episodes']):
                        env=Hallway(**meta['environment']);obs,_=env.reset(seed=spec['reset_base']+episode)
                        h=SelectionHarness(model,method,cfg,spec['reset_base']+episode,False)
                        factual=[];expected=[]
                        for t in range(env.episode_length):
                            if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Branch budget')
                            snapshot_env,snapshot_h=deepcopy(env),deepcopy(h)
                            baseline,action,trace,diag=suffix(deepcopy(snapshot_env),deepcopy(snapshot_h),obs.copy(),t)
                            noop=suffix(deepcopy(snapshot_env),deepcopy(snapshot_h),obs.copy(),t)
                            assert trace==noop[2];checks+=1;expected.append(trace)
                            for sender,source in enumerate(diag['source']):
                                if source<0:continue
                                for kind in ['zero','negate']:
                                    ret,a,_,_=suffix(deepcopy(snapshot_env),deepcopy(snapshot_h),obs.copy(),t,(sender,kind))
                                    row=dict(seed=seed,method=method,condition=f'{link}_{deadline:g}ms',episode=episode,t=t,sender=sender,source_step=source,age=t-source,intervention=kind,baseline_return=baseline,changed_return=ret,return_delta=ret-baseline,any_action_flip=bool(np.any(a!=action)),changed_agents=int(np.sum(a!=action)),message_norm=diag['norms'][sender],q_gap=diag['q_gap'][sender],priority=diag['priority'][sender],fresh_delivery=source==t)
                                    records.append(row);log.append([row])
                            a,_,_=h.step(obs,env.available_actions(),t)
                            assert np.array_equal(a,action)
                            obs,r,term,trunc,_=env.step(a);factual.append((a.tolist(),r,term,trunc,env.get_state().tolist()))
                            if term or trunc:break
                        for t,trace in enumerate(expected):assert trace==factual[t:]
                        atomic_json(out/'status.json',dict(stage='running',seed=seed,method=method,condition=f'{link}_{deadline:g}ms',episode=episode+1,branches=len(records),noop_checks=checks))
        groups=[]
        for key in sorted({(r['seed'],r['method'],r['condition'],r['intervention']) for r in records}):
            rr=[r for r in records if (r['seed'],r['method'],r['condition'],r['intervention'])==key]
            groups.append(dict(zip(['seed','method','condition','intervention'],key),n=len(rr),action_flip_rate=statistics.mean(r['any_action_flip'] for r in rr),return_changed_rate=statistics.mean(r['return_delta']!=0 for r in rr),mean_return_delta=statistics.mean(r['return_delta'] for r in rr),mean_absolute_return_delta=statistics.mean(abs(r['return_delta']) for r in rr)))
        atomic_json(out/'summary.json',dict(by_seed=groups,branches=len(records),noop_checks=checks,interpretation='conditional on factual visited caches; correlated branches, not independent episodes; no deployment timing claim'))
        atomic_json(out/'status.json',dict(stage='completed',seconds=time.monotonic()-start,branches=len(records),noop_checks=checks))
    except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
    finally:log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
    if a.smoke:s.update(output=s['output']+'_smoke',seeds=[0],methods=['dac'],conditions=[['slow',5.]],episodes=1)
    run(s)
