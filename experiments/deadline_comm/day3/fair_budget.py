"""Frozen 2k-checkpoint evaluation and per-condition empirical Pareto analysis."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml
from ..day2.close_day2_gaps import controlled
from ..day2.finetune_comm_aware import load_model
from ..day2.aggregate_day2 import measure
from ..day2.io import atomic_json, CSVLog, JSONLLog
from ..adapters.hallway import Hallway
from ..timing import configure_cpu


def budget_config(cfg, budget):
    cfg=deepcopy(cfg)
    cfg['dac']['budget_bytes']=float(budget)
    # The old three-packet burst would silently restrict 160/240-byte trials.
    cfg['dac']['burst_packets']=min(5,math.ceil(budget/48))
    cfg['dac']['threshold']=0.
    return cfg


def nondominated(points, cost, benefit):
    return [p for p in points if not any(
        q[cost]<=p[cost] and q[benefit]>=p[benefit]
        and (q[cost]<p[cost] or q[benefit]>p[benefit]) for q in points)]


def calibration(model, metadata, cfg, spec, seed, method, budget):
    env=Hallway(**metadata['environment'])
    local=budget_config(cfg,budget)
    local['link']['bandwidth_bytes_s']=1000000
    local['deadline_ns']=5000000
    def rate(value):
        candidate=deepcopy(local)
        if method=='periodic':candidate['gate']['period']=int(value)
        elif method=='vbc':candidate['gate']['q_threshold']=float(value)
        elif method=='tmc':candidate['gate']['message_threshold']=float(value)
        count=0;total=0
        for ep in range(spec['calibration_episodes']):
            rows,_,_=controlled(env,model,method,candidate,spec['calibration_seed_start']+seed*10000+ep)
            total+=sum(r['attempted_bytes'] for r in rows);count+=len(rows)
        return dict(parameter=value,offered_B_step=total/count,error=abs(total/count-budget)),candidate
    candidates=[]
    if method=='periodic':
        values=range(1,7)
        for value in values:candidates.append(rate(value))
    elif method=='dac':
        candidates.append(rate(0))
    else:
        # Fixed-grid anchors plus local refinement: retain all evidence if response
        # is nonmonotonic due to policy/cache feedback.
        values=[0.,.01,.03,.1,.3,1.,3.,10.,30.,100.]
        for value in values:candidates.append(rate(value))
        for _ in range(spec['calibration_iterations']):
            ordered=sorted(candidates,key=lambda pair:pair[0]['parameter'])
            best=min(range(len(ordered)),key=lambda i:ordered[i][0]['error'])
            neighbors=[]
            if best>0:neighbors.append((ordered[best-1][0]['parameter']+ordered[best][0]['parameter'])/2)
            if best+1<len(ordered):neighbors.append((ordered[best][0]['parameter']+ordered[best+1][0]['parameter'])/2)
            if not neighbors:break
            existing={x[0]['parameter'] for x in candidates}
            fresh=[v for v in neighbors if v not in existing]
            if not fresh:break
            for value in fresh:
                candidates.append(rate(value))
    best,cfg=min(candidates,key=lambda pair:pair[0]['error'])
    return cfg,dict(method=method,budget=budget,selected=best,eligible=best['error']<=.05*budget,
                    candidates=[r for r,_ in candidates],reference='fast_5ms')


def evaluate(model, metadata, cfg, spec, seed, method, budget, link, deadline, out):
    env=Hallway(**metadata['environment'])
    cfg=deepcopy(cfg)
    cfg['deadline_ns']=int(deadline*1e6)
    cfg['link']['bandwidth_bytes_s']=spec['links'][link]
    for ep in range(spec['warmup_episodes']):
        controlled(env,model,method,cfg,5300000+ep)
    destination=out/f'{link}_{deadline:g}ms'
    destination.mkdir()
    steps=[];episodes=[]
    sl,el=CSVLog(destination/'steps.csv'),CSVLog(destination/'episodes.csv')
    ml=JSONLLog(destination/'messages.jsonl')
    try:
        for ep in range(spec['episodes']):
            ss,mm,ee=controlled(env,model,method,cfg,spec['evaluation_seed_start']+ep)
            tags=dict(episode=ep,seed=seed,budget=budget)
            sl.append([{**tags,**r} for r in ss]);el.append([{**tags,**ee}])
            ml.append([{**tags,**r} for r in mm])
            steps.extend(ss);episodes.append(ee)
            assert all(abs(r['cpu_total_ns']+r['comm_model_ns']-r['step_total_ns'])<1e-5 for r in ss)
            if method=='dac':
                assert sum(r['attempted_bytes'] for r in ss)<=budget*len(ss)+1e-8
                assert all(r['K_attempt']<=r['k_max'] for r in ss)
        result=measure(episodes,steps)
        result.update(seed=seed,method=method,budget=budget,condition=f'{link}_{deadline:g}ms',
                      budget_eligible=abs(result['offered_B_step']-budget)<=.05*budget,
                      step_p95_us=float(np.quantile([r['step_total_ns']/1000 for r in steps],.95)),
                      cpu_p95_us=float(np.quantile([r['cpu_total_ns']/1000 for r in steps],.95)))
        atomic_json(destination/'summary.json',result)
        return result
    finally:sl.close();el.close();ml.close()


def analyze(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={}
    for row in records:groups.setdefault((row['condition'],row['method'],row['budget']),[]).append(row)
    summaries=[]
    for (condition,method,budget),rows in sorted(groups.items()):
        if len(rows)!=3:continue
        s=dict(condition=condition,method=method,budget=budget,all_budget_eligible=all(r['budget_eligible'] for r in rows))
        for metric in ['mean_return','offered_B_step','goodput_payload_B_step','wasted_pct','step_p95_us','cpu_p95_us']:
            values=[r[metric] for r in rows if r[metric] is not None]
            s[metric]=float(np.mean(values)) if values else None
            s[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(s)
    atomic_json(out/'aggregate.json',summaries)
    if not summaries:return
    with (out/'aggregate.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    fronts=[]
    colors=dict(periodic='#0072B2',vbc='#D55E00',vbc_batched='#E69F00',tmc='#CC79A7',dac='#009E73')
    for link in ['fast','medium','slow']:
        fig,axes=plt.subplots(3,4,figsize=(17,11))
        for col,deadline in enumerate([5.,2.5,2.,1.5]):
            condition=f'{link}_{deadline:g}ms'
            points=[r for r in summaries if r['condition']==condition]
            for row,cost in enumerate(['offered_B_step','step_p95_us','goodput_payload_B_step']):
                ax=axes[row,col]
                for method,color in colors.items():
                    ps=[r for r in points if r['method']==method]
                    for p in ps:
                        ax.errorbar(p[cost],p['mean_return'],xerr=p[cost+'_sd'],yerr=p['mean_return_sd'],
                                    color=color,alpha=.35,capsize=2)
                        ax.scatter(p[cost],p['mean_return'],edgecolors=color,
                                   facecolors=color if p['all_budget_eligible'] else 'none',label=method if p==ps[0] else None)
                if row<2:
                    front=sorted(nondominated(points,cost,'mean_return'),key=lambda r:r[cost])
                    ax.plot([r[cost] for r in front],[r['mean_return'] for r in front],color='black',ls='--',lw=1)
                    fronts.extend(dict(condition=condition,cost=cost,method=p['method'],budget=p['budget']) for p in front)
                ax.set_title(condition);ax.set_xlabel({'offered_B_step':'Actual offered B/step (minimize)',
                    'step_p95_us':'Step p95 us (minimize)','goodput_payload_B_step':'Payload B/step (maximize)'}[cost])
                ax.set_ylabel('Return (maximize)');ax.grid(alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5)
        fig.suptitle(f'{link}: 2k shared checkpoints, mean ± seed SD; hollow = budget ineligible',y=.96)
        fig.tight_layout(rect=[0,0,1,.93])
        for ext in ['png','pdf']:fig.savefig(out/f'figure3_pareto_{link}.{ext}',dpi=180)
        plt.close(fig)
    atomic_json(out/'empirical_frontiers.json',fronts)
    lines=['# Matched 2k checkpoint evaluation and budget sweep','',
           'Three training seeds, twenty paired evaluation resets per condition; one timing pass per reset.',
           'All methods use the same Full-Comm 2k weights per seed. VBC/TMC-style gates are diagnostic ports.',
           'Calibration uses disjoint resets at fast 5ms; all parameters are frozen before the matrix.',
           'Periodic retains integer periods. 96/160 B may be unattainable without changing that baseline.',
           'DAC burst cap scales as ceil(budget/48), capped at N. A retained three-packet cap would invalidate high-budget trials.',
           'Budget eligibility is assessed against each target in every seed, not against the across-seed average.',
           'Frontiers are empirical non-dominated mean points within one link/deadline; they do not imply statistical superiority.',
           'Goodput is maximized, not treated as a resource cost. p95 values are computed per seed, then averaged.',
           'CPU timing is single-threaded with GC disabled per rollout and two warmups; shared-host noise remains possible.',
           '## Matched 120 B target','',
           '| Condition | Method | Return mean ± SD | Actual B | Payload B | Waste % | Eligible all seeds |',
           '|---|---|---|---|---|---|---|']
    for r in summaries:
        if r['budget']!=120:continue
        waste='NA' if r['wasted_pct'] is None else f"{r['wasted_pct']:.2f}"
        lines.append(f"| {r['condition']} | {r['method']} | {r['mean_return']:.3f} ± {r['mean_return_sd']:.3f} | {r['offered_B_step']:.2f} | {r['goodput_payload_B_step']:.2f} | {waste} | {r['all_budget_eligible']} |")
    (out/'FAIR_BUDGET_REPORT.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();spec=yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        spec.update(output=spec['output']+'_smoke',seeds=[0],budgets=[120,240],links={'fast':1000000},
                    deadlines_ms=[5.],episodes=1,calibration_episodes=1,calibration_iterations=1)
    configure_cpu();out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    checkpoints={seed:Path(spec['checkpoint_root'])/f'seed{seed}/checkpoint_0500.pt' for seed in spec['seeds']}
    state=dict(status='running',config=spec,start_unix=time.time(),completed_conditions=0,
               question='Matched 2k deployment comparison and observed rate-return tradeoffs',
               falsification='Budget failures and inferior/identical performance remain negative results',
               checkpoint_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.values()},
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'manifest.json',state)
    records=[]
    try:
        for seed in spec['seeds']:
            model,metadata=load_model(checkpoints[seed])
            cfg=json.loads((Path('experiments/deadline_comm/runs/day2_dac_marl_main_20261118_001')/f'seed{seed}/frozen_harness.json').read_text())
            seedout=out/f'seed{seed}';seedout.mkdir()
            for budget in spec['budgets']:
                calibrated={}
                for method in spec['methods']:
                    dest=seedout/f'{method}_{budget}B';dest.mkdir()
                    if method=='vbc_batched':
                        selected,record=calibrated['vbc']
                    else:selected,record=calibration(model,metadata,cfg,spec,seed,method,budget)
                    calibrated[method]=(selected,record)
                    atomic_json(dest/'calibration.json',record)
                    print('CALIBRATED',seed,budget,method,record['selected'],flush=True)
                # Rotate method ordering to reduce a systematic timing order effect.
                for link in spec['links']:
                    for d in spec['deadlines_ms']:
                        methods=spec['methods']
                        offset=(seed+list(spec['links']).index(link)+spec['deadlines_ms'].index(d))%len(methods)
                        for method in methods[offset:]+methods[:offset]:
                            dest=seedout/f'{method}_{budget}B'
                            records.append(evaluate(model,metadata,calibrated[method][0],spec,seed,method,budget,link,d,dest))
                            state.update(seed=seed,budget=budget,condition=f'{link}_{d}ms',method=method,
                                         completed_conditions=len(records))
                            atomic_json(out/'records.json',records);atomic_json(out/'manifest.json',state)
                            print('DONE',seed,budget,link,d,method,len(records),flush=True)
        analyze(out,records);state.update(status='completed',finish_unix=time.time())
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:atomic_json(out/'manifest.json',state)


if __name__=='__main__':main()
