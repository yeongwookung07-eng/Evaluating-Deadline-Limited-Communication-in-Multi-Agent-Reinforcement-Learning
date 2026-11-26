"""Frozen broadcast policies with late candidates rejected during fresh rollouts."""
import argparse
from copy import deepcopy
import gc
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
import yaml
from ..day2.policy import DecayedPolicy
from ..receiver_adaptation.evaluate import SplitUtility, interval
from ..reviewer_validation.hidden_source import HiddenSource
from ..reviewer_validation.train_ten_seeds import configure_arm
from ..reviewer_validation.endpoints import EndpointHarness
from ..timing import configure_cpu
from ..tmlr_revision.experiments import setup, clone


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')


def executed_action(candidate, previous, missed, fallback):
    if fallback not in ('ordinary','hold','zero'):raise ValueError(fallback)
    if not missed or fallback=='ordinary':return np.asarray(candidate).copy()
    return np.asarray(previous).copy() if fallback=='hold' else np.zeros_like(candidate)


def load_model(c,seed,arm,scheduler,hashes):
    path=Path(c['checkpoint_template'].format(seed=seed));hashes[str(path)]=digest(path)
    base=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix')
    base.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    head=deepcopy(base.utility)
    if arm!='original':
        hp=Path(c['head_run'])/f'seed{seed}_{scheduler}_{arm}_head.pt';hashes[str(hp)]=digest(hp)
        head.load_state_dict(torch.load(hp,map_location='cpu',weights_only=True))
    model=deepcopy(base);model.utility=SplitUtility(base.utility,head);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    for name,value in base.state_dict().items():
        if not name.startswith('utility.'):assert torch.equal(value,model.state_dict()[name])
    return model


def rollout(model,reset,scheduler,fallback,cost,deadline_ms=2.,fixed=False,forced_miss=None):
    method,cfg=configure_arm(setup(['medium',deadline_ms]),'rr_cap',100000,deadline_ms)
    cfg['fixed_ready']=fixed or scheduler=='source_aware'
    env=HiddenSource();obs,_=env.reset(seed=reset)
    h=EndpointHarness(model,method,cfg,reset,True)
    previous=np.zeros(5,dtype=np.int64);trace=[]
    with torch.inference_mode():
        for t in range(env.episode_length):
            available=env.available_actions();overhead=0;chosen=None
            if scheduler=='source_aware':
                tick=time.perf_counter_ns()
                _,factual,logs=clone(h).step(obs.copy(),available,t)
                selected=sorted(m['sender'] for m in logs if m['accepted']);k=len(selected)
                chosen=sorted(([0]+[i for i in selected if i!=0])[:k]) if k else []
                overhead=time.perf_counter_ns()-tick
            def select(scores,kmax):
                result=torch.zeros(5,dtype=torch.bool);result[chosen]=True
                assert len(chosen)<=kmax
                return result
            tick=time.perf_counter_ns()
            if chosen is None:
                candidate,row,logs=h.step(obs,available,t)
            else:
                with patch.object(h.dac,'select',side_effect=select):
                    candidate,row,logs=h.step(obs,available,t)
                assert row['K_delivered']==factual['K_delivered']
            cpu=time.perf_counter_ns()-tick+overhead
            total=cpu+row['comm_model_ns']+cost*1000
            missed=total>cfg['deadline_ns'] if forced_miss is None else forced_miss
            action=executed_action(candidate,previous,missed,fallback)
            # Actual actuator output, not rejected candidate, is next recurrent input.
            h.previous=torch.nn.functional.one_hot(torch.as_tensor(action)[None],2).float()
            assert np.array_equal(h.previous.argmax(-1)[0].numpy(),action)
            bit=int(env.bits[0]);obs,reward,done,_,_=env.step(action)
            assert reward==float((action[1:]==bit).mean())
            trace.append(dict(t=t,bit=bit,candidate=candidate.tolist(),actions=action.tolist(),
                previous_executed=previous.tolist(),reward=reward,miss=bool(missed),
                late_candidate_applied=bool(missed and fallback=='ordinary'),
                fallback_selected=bool(missed and fallback!='ordinary'),
                candidate_changed=bool(not np.array_equal(action,candidate)),
                cpu_total_ns=cpu,gate_ns=row['gate_ns'],encode_ns=row['encode_ns'],post_cpu_ns=row['post_cpu_ns'],
                selection_overhead_ns=overhead,post_cost_ns=cost*1000,
                total_ns=total,wait_ns=row['comm_model_ns'],ready_ns=row['ready_ns'],
                offered_wire_B=row['wire_bytes'],delivered_payload_B=32*row['K_delivered'],
                offered_senders=[m['sender'] for m in logs],
                delivered_senders=sorted(m['sender'] for m in logs if m['accepted']),
                source_timestamps=h.source[0].tolist(),
                messages=logs))
            previous=action.copy()
            if done:break
    return trace


def behavioral(trace):
    return [{k:r[k] for k in ('t','bit','candidate','actions','reward','offered_wire_B','delivered_payload_B','delivered_senders','source_timestamps')} for r in trace]


def validate_controls(c,hashes,budget):
    checks=0;hist={}
    with (Path(c['head_run'])/'episodes.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['episode']<2:hist[row['seed'],row['scheduler'],row['arm'],row['episode']]=row['trace']
    for seed in c['seeds']:
        for scheduler in ('normal','source_aware'):
            for arm in c['heads']:
                model=load_model(c,seed,arm,scheduler,hashes)
                for ep in range(2):
                    budget();tr=rollout(model,17300000+ep,scheduler,'ordinary',0,fixed=True)
                    old=hist[seed,scheduler,arm,ep]
                    for a,b in zip(tr,old):
                        for key in ('t','bit','actions','reward','delivered_senders','source_timestamps'):assert a[key]==b[key],(seed,scheduler,arm,ep,key)
                        assert a['offered_wire_B']==b['offered_wire_bytes']
                        assert a['delivered_payload_B']==b['delivered_payload_bytes']
                    assert len(tr)==len(old);checks+=1
    model=load_model(c,0,'adapted','normal',hashes)
    ordinary=rollout(model,c['reset_base'],'normal','ordinary',0,fixed=True,forced_miss=False)
    for fb in ('hold','zero'):
        on_time=rollout(model,c['reset_base'],'normal',fb,0,fixed=True,forced_miss=False)
        assert behavioral(ordinary)==behavioral(on_time)
        late=rollout(model,c['reset_base'],'normal',fb,0,fixed=True,forced_miss=True)
        assert all(r['actions']==[0]*5 and not r['late_candidate_applied'] for r in late)
    return dict(historical_episodes_exact=checks,no_miss_identity=True,all_miss_fallback=True,executed_action_feedback=True)


def run(c,repeat,smoke=False):
    configure_cpu();root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    out=root/(f'repeat{repeat}' if not smoke else 'smoke');out.mkdir(exist_ok=False)
    write(out/'registration.json',dict(**c,repeat=repeat,smoke=smoke,entry_sha256=digest(__file__)))
    start=time.monotonic();hashes={};rows=[];episodes=0;steps=0
    def budget():
        if time.monotonic()-start>c['wall_seconds_per_repeat']:raise TimeoutError('Registered per-process CPU evaluation wall budget exhausted')
    write(out/'host.json',dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,threads=torch.get_num_threads(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
    try:
        if repeat==0:checks=validate_controls(c,hashes,budget)
        else:checks=dict(reference_controls='repeat0/validation.json')
        seeds=c['seeds'] if not smoke else [0]
        n_ep=c['episodes'] if not smoke else 2
        with gzip.open(out/'episodes.jsonl.gz','wt',encoding='utf-8') as stream:
            for seed in seeds:
                cells=list(itertools.product(['normal'],c['heads'],c['fallbacks'],c['post_cost_us']))
                if repeat==0 and c['source_aware_diagnostic']:
                    cells+=list(itertools.product(['source_aware'],c['heads'],c['fallbacks'],[0]))
                # Predeclared reproducible random permutation, independent of results.
                order=np.random.default_rng(261126+100*repeat+seed).permutation(len(cells))
                models={(sched,arm):load_model(c,seed,arm,sched,hashes) for sched,arm,_,_ in cells}
                for idx in order:
                    scheduler,arm,fb,cost=cells[idx];model=models[scheduler,arm];budget()
                    state={k:v.clone() for k,v in model.state_dict().items()}
                    gc.collect();gc.disable()
                    try:
                        for ep in range(c['warmup_episodes']):rollout(model,c['reset_base']+10000+ep,scheduler,fb,cost,c['deadline_ms'])
                        traces=[]
                        for ep in range(n_ep):
                            budget();reset=c['reset_base']+ep
                            tr=rollout(model,reset,scheduler,fb,cost,c['deadline_ms'])
                            record=dict(seed=seed,repeat=repeat,scheduler=scheduler,arm=arm,fallback=fb,post_cost_us=cost,
                                        episode=ep,reset=reset,return_=sum(r['reward'] for r in tr),trace=tr)
                            stream.write(json.dumps(record,separators=(',',':'))+'\n');traces.extend(tr)
                            episodes+=1;steps+=len(tr)
                    finally:gc.enable()
                    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
                    if fb!='ordinary':assert not any(r['late_candidate_applied'] for r in traces)
                    row=dict(seed=seed,repeat=repeat,scheduler=scheduler,arm=arm,fallback=fb,post_cost_us=cost,
                        accuracy=float(np.mean([r['reward'] for r in traces])),mean_return=sum(r['reward'] for r in traces)/n_ep,
                        miss_fraction=float(np.mean([r['miss'] for r in traces])),
                        replaced_fraction=float(np.mean([r['candidate_changed'] for r in traces])),
                        offered_B=float(np.mean([r['offered_wire_B'] for r in traces])),
                        payload_B=float(np.mean([r['delivered_payload_B'] for r in traces])),
                        ready_us=float(np.mean([r['ready_ns'] for r in traces]))/1000,
                        cpu_us=float(np.mean([r['cpu_total_ns'] for r in traces]))/1000,
                        total_p95_us=float(np.quantile([r['total_ns'] for r in traces],.95))/1000)
                    rows.append(row)
                write(out/'status.json',dict(stage='running',seed=seed,cells=len(rows),episodes=episodes,wall_seconds=time.monotonic()-start))
                print(json.dumps(dict(repeat=repeat,seed=seed,cells=len(rows),episodes=episodes,wall_seconds=round(time.monotonic()-start,2))),flush=True)
        for path,value in hashes.items():assert digest(path)==value
        write(out/'input_hashes.json',hashes);write(out/'results.json',dict(rows=rows,episodes=episodes,steps=steps))
        write(out/'validation.json',dict(passed=True,checks=checks,source_hashes_unchanged=True,weights_unchanged=True,no_late_candidates_applied_in_hard_arms=True))
        write(out/'status.json',dict(stage='completed',cells=len(rows),episodes=episodes,steps=steps,wall_seconds=time.monotonic()-start))
    except Exception as exc:
        write(out/'partial.json',rows);write(out/'status.json',dict(stage='failed',error=repr(exc),episodes=episodes,wall_seconds=time.monotonic()-start));raise


def summarize(c):
    root=Path(c['output']);rows=[]
    for repeat in range(c['repeats']):
        p=root/f'repeat{repeat}';assert json.loads((p/'status.json').read_text())['stage']=='completed'
        rows+=json.loads((p/'results.json').read_text())['rows']
    cells=[]
    for scheduler in ('normal','source_aware'):
        for fb,cost in itertools.product(c['fallbacks'],c['post_cost_us'] if scheduler=='normal' else [0]):
            group=[r for r in rows if (r['scheduler'],r['fallback'],r['post_cost_us'])==(scheduler,fb,cost)]
            if not group:continue
            reps=sorted(set(r['repeat'] for r in group));per_repeat=[]
            for repeat in reps:
                ix={(r['seed'],r['arm']):r for r in group if r['repeat']==repeat}
                per_repeat.append(dict(repeat=repeat,adapted_minus_original_pp=interval([100*(ix[s,'adapted']['accuracy']-ix[s,'original']['accuracy']) for s in c['seeds']]),
                                       adapted_minus_shuffled_pp=interval([100*(ix[s,'adapted']['accuracy']-ix[s,'shuffled']['accuracy']) for s in c['seeds']])))
            metrics={}
            for arm in c['heads']:
                metrics[arm]={k:interval([np.mean([r[k] for r in group if r['seed']==s and r['arm']==arm]) for s in c['seeds']]) for k in ['accuracy','mean_return','miss_fraction','replaced_fraction','offered_B','payload_B','cpu_us','ready_us','total_p95_us']}
            contrasts={}
            for comparator in ('original','shuffled'):
                diffs=[]
                for s in c['seeds']:
                    adapted=np.mean([r['accuracy'] for r in group if r['seed']==s and r['arm']=='adapted'])
                    other=np.mean([r['accuracy'] for r in group if r['seed']==s and r['arm']==comparator])
                    diffs.append(100*(adapted-other))
                contrasts['adapted_minus_'+comparator+'_pp']=interval(diffs)
            cells.append(dict(scheduler=scheduler,fallback=fb,post_cost_us=cost,metrics=metrics,contrasts=contrasts,per_repeat=per_repeat))
    primary=next(r for r in cells if (r['scheduler'],r['fallback'],r['post_cost_us'])==('normal','hold',0))
    passed=all(r[k]['ci95'][0]>0 for r in primary['per_repeat'] for k in ['adapted_minus_original_pp','adapted_minus_shuffled_pp'])
    write(root/'summary.json',dict(cells=cells,primary=primary,primary_robust_gain_pass=passed,training_seeds=10,process_repeats=c['repeats']))
    print(json.dumps(dict(primary_robust_gain_pass=passed,primary=primary),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--repeat',type=int,default=0);p.add_argument('--summarize',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text())
    if a.summarize:summarize(c)
    else:run(c,a.repeat,a.smoke)
