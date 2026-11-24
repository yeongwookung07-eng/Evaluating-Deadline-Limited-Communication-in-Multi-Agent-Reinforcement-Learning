"""Held-out zero-shot endpoints and exact shadow delivery-identity control."""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..common import load_config
from ..day2.io import atomic_json
from ..day2.policy import DecayedPolicy
from ..reviewer_validation.endpoints import IdealLink, NeverSend
from ..selection_validation.harness import SelectionHarness
from ..standard_benchmarks.smac_adapter import SMACAdapter
from ..timing import configure_cpu


def load_model(checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);m=saved['metadata'];d=m['dimensions']
    model=DecayedPolicy(d['obs'],d['state'],d['agents'],d['actions'],**m['policy'])
    model.load_state_dict(saved['model']);model.eval();model.decay_lambda=1.
    return model


def make_harness(model,arm,spec,condition,mode,seed):
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    cfg=deepcopy(base);setting=spec['evaluation']['conditions'][condition]
    cfg.update(deadline_ns=int(setting['deadline_ms']*1e6),r_post_ns=spec['evaluation']['r_post_ns'],training_ready_ns=spec['evaluation']['fixed_ready_ns'],fixed_ready=mode=='fixed',decay_lambda=spec['evaluation']['decay_lambda'])
    cfg['dac']['budget_bytes']=spec['evaluation']['budget_bytes'];cfg['link']['bandwidth_bytes_s']=setting['bandwidth']
    h=SelectionHarness(model,arm if arm not in ('ideal_full','no_comm') else 'full',cfg,seed,True)
    if arm=='ideal_full':h.link=IdealLink(float('inf'),0,h.link.header_bytes)
    if arm=='no_comm':h.gate=NeverSend(model.n_agents)
    return h


def digest(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def delivery_ids(messages):return sorted(m['sender'] for m in messages if m['accepted'])


def compare_states(a,b,aa,ab,ma,mb):
    return dict(delivery=delivery_ids(ma)==delivery_ids(mb),source=torch.equal(a.source,b.source),cache=torch.equal(a.cache,b.cache),hidden=torch.equal(a.hidden,b.hidden),action=np.array_equal(aa,ab))


@torch.inference_mode()
def rollout(model,arm,spec,condition,mode,reset_seed,step_sink=None,message_sink=None,shadow=False):
    # Each condition/method starts from a fresh native engine, not just a wrapper reseed.
    env=SMACAdapter(spec['scenario'],engine_seed=reset_seed)
    enabled=gc.isenabled();gc.collect();gc.disable()
    try:
        obs,_=env.reset(seed=reset_seed);h=make_harness(model,arm,spec,condition,mode,reset_seed)
        other=make_harness(model,'rr_cap',spec,condition,'fixed',reset_seed) if shadow else None
        trace=hashlib.sha256();checks=[];rows=[];total=0.;first_mismatch=None
        for t in range(env.episode_length):
            state=env.get_state();available=env.available_actions();a,row,msg=h.step(obs,available,t)
            if other is not None:
                b,_,omsg=other.step(obs,available,t);check=compare_states(h,other,a,b,msg,omsg);checks.append(check)
                if not all(check.values()) and first_mismatch is None:first_mismatch=dict(t=t,checks=check,full_senders=delivery_ids(msg),rr_senders=delivery_ids(omsg))
            if arm=='no_comm':assert row['K_attempt']==row['K_delivered']==0 and (h.source<0).all()
            if arm=='ideal_full':assert row['K_delivered']==env.n_agents
            next_obs,reward,term,trunc,info=env.step(a);total+=reward
            signature=digest(obs,state,available,a,h.hidden,h.cache,h.source,np.array([reward,term,trunc],dtype=np.float64),np.asarray(delivery_ids(msg),np.int64))
            trace.update(signature.encode())
            row.update(t=t,arm=arm,mode=mode,condition=condition,reset_seed=reset_seed,reward=reward,terminated=term,truncated=trunc,won=bool(info.get('battle_won',False)),payload_goodput_bytes=32*row['K_delivered'],delivered_senders=delivery_ids(msg),actions=a.tolist(),source_step=h.source[0].tolist(),trace_step_sha256=signature)
            rows.append(row)
            if step_sink:step_sink.write(json.dumps(row)+'\n')
            if message_sink:
                for m in msg:message_sink.write(json.dumps(dict(m,arm=arm,mode=mode,condition=condition,reset_seed=reset_seed))+'\n')
            obs=next_obs
            if term or trunc:break
        result=dict(arm=arm,mode=mode,condition=condition,reset_seed=reset_seed,return_=total,won=bool(info.get('battle_won',False)),length=len(rows),offered_bytes=sum(r['wire_bytes'] for r in rows),delivered_payload_bytes=sum(r['payload_goodput_bytes'] for r in rows),attempts=sum(r['K_attempt'] for r in rows),deliveries=sum(r['K_delivered'] for r in rows),action_misses=sum(r['action_miss'] for r in rows),trace_sha256=trace.hexdigest(),cpu_mean_us=float(np.mean([r['cpu_total_ns']/1000 for r in rows])),cpu_p95_us=float(np.percentile([r['cpu_total_ns']/1000 for r in rows],95)))
        if other is not None:result['identity']=dict(passed=all(all(c.values()) for c in checks),steps=len(checks),first_mismatch=first_mismatch,matching_steps={k:sum(c[k] for c in checks) for k in checks[0]},interpretation='Shadow shares factual environment inputs; only complete state/action identity supports unchanged trajectory. No alternate return is imputed after mismatch.')
        return result
    finally:
        env.close()
        if enabled:gc.enable()
        gc.collect()


def seed_summary(rows):
    nsteps=sum(r['length'] for r in rows);attempts=sum(r['attempts'] for r in rows);delivered=sum(r['deliveries'] for r in rows)
    return dict(return_mean=float(np.mean([r['return_'] for r in rows])),win_rate=float(np.mean([r['won'] for r in rows])),offered_b_per_step=sum(r['offered_bytes'] for r in rows)/nsteps,goodput_b_per_step=sum(r['delivered_payload_bytes'] for r in rows)/nsteps,waste_fraction=(attempts-delivered)/attempts if attempts else None,action_miss_rate=sum(r['action_misses'] for r in rows)/nsteps,cpu_episode_mean_us=float(np.mean([r['cpu_mean_us'] for r in rows])),episodes=len(rows),steps=nsteps)


def bootstrap_pairs(differences,samples=10000):
    d=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(20261121)
    means=d[rng.integers(0,len(d),size=(samples,len(d)))].mean(1)
    return dict(mean=float(d.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist(),differences=d.tolist(),n_training_seeds=len(d),scope='Exploratory percentile bootstrap of training-seed differences; 3 seeds have low precision, no equivalence or significance claim.')


def aggregate(spec,root,rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['training_seed'],r['condition'],r['mode'],r['arm'])].append(r)
    per_seed=[dict(training_seed=k[0],condition=k[1],mode=k[2],arm=k[3],**seed_summary(v)) for k,v in sorted(grouped.items())]
    with (root/'by_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
    groups=defaultdict(list)
    for r in per_seed:groups[(r['condition'],r['mode'],r['arm'])].append(r)
    means=[]
    for key,rr in sorted(groups.items()):
        cell=dict(condition=key[0],mode=key[1],arm=key[2],n_seeds=len(rr))
        for metric in ['return_mean','win_rate','offered_b_per_step','goodput_b_per_step','cpu_episode_mean_us']:
            vals=[r[metric] for r in rr];cell[metric]=float(np.mean(vals));cell[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        means.append(cell)
    paired=[]
    for cond in spec['evaluation']['conditions']:
        for mode in ['fixed','measured']:
            for a,b in [('ideal_full','no_comm'),('dac','rr_cap'),('full','rr_cap')]:
                for metric in ['return_mean','win_rate']:
                    d=[]
                    for seed in spec['seeds']:
                        index={r['arm']:r for r in per_seed if r['condition']==cond and r['mode']==mode and r['training_seed']==seed}
                        d.append(index[a][metric]-index[b][metric])
                    paired.append(dict(condition=cond,mode=mode,contrast=a+'-'+b,metric=metric,**bootstrap_pairs(d,spec['evaluation']['bootstrap_samples'])))
    identities=[dict(training_seed=r['training_seed'],reset_seed=r['reset_seed'],condition=r['condition'],**r['identity']) for r in rows if 'identity' in r]
    atomic_json(root/'identity.json',identities)
    summary=dict(stage='completed',cohort='All gated Ideal-trained checkpoints; held-out zero-shot deployments, no filtering',means=means,paired=paired,identity=dict(episodes=len(identities),passed=sum(r['passed'] for r in identities),failed=sum(not r['passed'] for r in identities)),scope='Standard SMACv2 Terran 5v5 on one learned backbone; hybrid serial-link model, no real radio guarantee. 3 training seeds; no native ranking or universal generalization.')
    atomic_json(root/'summary.json',summary)
    lines=['# SMACv2 통신 평가 결과','',summary['scope'],'','| 조건 | 모드 | 방식 | Return 평균 ± SD | 승률 | Wire B/step | Payload B/step |','|---|---|---|---:|---:|---:|---:|']
    for r in means:
        sd='NA' if r['return_mean_sd'] is None else f"{r['return_mean_sd']:.3f}"
        lines.append(f"| {r['condition']} | {r['mode']} | {r['arm']} | {r['return_mean']:.3f} ± {sd} | {100*r['win_rate']:.2f}% | {r['offered_b_per_step']:.2f} | {r['goodput_b_per_step']:.2f} |")
    lines.extend(['',f"Full–RR 동일 이력: {summary['identity']['passed']}/{len(identities)} 에피소드 통과. 불일치는 identity.json에 보존.",'','CI는 paired.json 상당의 summary.json paired 필드에 있으며 학습 seed 3개 기준 탐색적 구간입니다. 환경 reset 반복을 독립 학습 seed처럼 세지 않습니다.','학습 기준, 독립 엔진 재현 검사와 원시 episodes/steps/messages를 함께 확인해야 합니다.'])
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(spec,checkpoints,deadline):
    configure_cpu();out=Path(spec['output'])/'evaluation';out.mkdir(exist_ok=False)
    atomic_json(out/'manifest.json',dict(spec=spec,checkpoints=checkpoints,started=time.time(),timing_note='Serial evaluation; fixed readiness primary and measured sensitivity; no actual network sleep.'))
    all_rows=[]
    try:
        for seed in spec['seeds']:
            model=load_model(checkpoints[str(seed)])
            # Every checkpoint must reproduce a full native episode across fresh engines.
            cond=next(iter(spec['evaluation']['conditions']));test_seed=spec['heldout_seed_base']-100+seed
            first=rollout(model,'full',spec,cond,'fixed',test_seed,shadow=True)
            repeat=rollout(model,'full',spec,cond,'fixed',test_seed,shadow=True)
            atomic_json(out/f'seed{seed}_replay_check.json',dict(first=first,repeat=repeat,passed=first['trace_sha256']==repeat['trace_sha256']))
            if first['trace_sha256']!=repeat['trace_sha256']:raise AssertionError('Native full-episode replay mismatch; paired causal conclusions blocked')
            for condition in spec['evaluation']['conditions']:
                for mode in ['fixed','measured']:
                    for arm in spec['evaluation']['methods']:
                        if time.time()>deadline:raise TimeoutError('Campaign budget')
                        cell=out/f'seed{seed}_{condition}_{mode}_{arm}';cell.mkdir()
                        for j in range(spec['evaluation']['warmup_episodes']):
                            rollout(model,arm,spec,condition,mode,spec['heldout_seed_base']-1000+j)
                        result=[]
                        with (cell/'steps.jsonl').open('w',buffering=1) as sf,(cell/'messages.jsonl').open('w',buffering=1) as mf,(cell/'episodes.jsonl').open('w',buffering=1) as ef:
                            for ep in range(spec['evaluation_episodes']):
                                if time.time()>deadline:raise TimeoutError('Campaign budget')
                                r=rollout(model,arm,spec,condition,mode,spec['heldout_seed_base']+ep,sf,mf,shadow=mode=='fixed' and arm=='full')
                                r['training_seed']=seed;result.append(r);all_rows.append(r);ef.write(json.dumps(r)+'\n')
                                atomic_json(out/'status.json',dict(stage='evaluating',seed=seed,condition=condition,mode=mode,arm=arm,episode=ep+1,total_completed=len(all_rows)))
                        atomic_json(cell/'summary.json',seed_summary(result))
                        atomic_json(out/'partial_episodes.json',all_rows)
                        print('EVAL',seed,condition,mode,arm,seed_summary(result),flush=True)
        summary=aggregate(spec,out,all_rows)
        atomic_json(out/'status.json',dict(stage='completed',episodes=len(all_rows)));return summary
    except Exception as exc:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc),completed_episodes=len(all_rows)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoints',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    run(load_config(a.config),json.loads(Path(a.checkpoints).read_text()),a.deadline)
