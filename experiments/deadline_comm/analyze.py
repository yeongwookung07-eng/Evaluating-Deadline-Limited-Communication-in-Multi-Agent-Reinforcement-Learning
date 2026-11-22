"""Descriptive ranks, seed-level return intervals and matched-budget eligibility."""
import argparse
import csv
import itertools
from pathlib import Path
import numpy as np
from .common import save_json


def analyze(path, tolerance=.05):
    path=Path(path)
    steps=list(csv.DictReader((path/'steps.csv').open()))
    episodes=list(csv.DictReader((path/'episodes.csv').open()))
    groups={}
    for condition,method in sorted({(r['condition'],r['method']) for r in steps}):
        ss=[r for r in steps if r['condition']==condition and r['method']==method]
        ee=[r for r in episodes if r['condition']==condition and r['method']==method]
        seed_returns=[np.mean([float(r['return']) for r in ee if r['training_seed']==seed]) for seed in sorted({r['training_seed'] for r in ee})]
        rng=np.random.default_rng(581)
        ci=np.quantile([np.mean(rng.choice(seed_returns,len(seed_returns),replace=True)) for _ in range(1000)],[.025,.975]).tolist() if len(seed_returns)>=3 else None
        timings={k:{'mean':float(np.mean(v)),'p50':float(np.quantile(v,.5)), 'p95':float(np.quantile(v,.95)), 'p99':float(np.quantile(v,.99))}
            for k in ('encode_ns','gate_ns','cache_ns','act_ns','other_ns','cpu_total_ns','comm_model_ns','step_total_ns','serialize_ns','post_cpu_ns')
            for v in [[float(r[k]) for r in ss]]}
        groups[(condition,method)]=dict(condition=condition,method=method,episodes=len(ee),training_seeds=len(seed_returns),
            mean_return=float(np.mean(seed_returns)),return_ci95_seed_bootstrap=ci,
            mean_wire_bytes=float(np.mean([float(r['wire_bytes']) for r in ss])),
            action_miss_rate=float(np.mean([r['action_miss']=='True' for r in ss])),
            message_drop_rate=sum(int(r['drop_count']) for r in ss)/max(1,sum(int(r['K_attempt']) for r in ss)),
            mean_K=float(np.mean([int(r['K_attempt']) for r in ss])),
            p95_K=float(np.quantile([int(r['K_attempt']) for r in ss],.95)),
            mean_age=float(np.mean([float(r['mean_age']) for r in ss if r['mean_age']])) if any(r['mean_age'] for r in ss) else None,
            timing=timings)
    comparisons=[]
    correlations=[]
    for condition in sorted({k[0] for k in groups}):
        arms=[v for k,v in groups.items() if k[0]==condition]
        for arm in arms:
            arm['volume_rank']=1+sum(v['mean_wire_bytes']<arm['mean_wire_bytes'] for v in arms)
            arm['deadline_rank']=1+sum(v['action_miss_rate']<arm['action_miss_rate'] for v in arms)
        pairs=list(itertools.combinations(arms,2))
        signs=[(np.sign(a['mean_wire_bytes']-b['mean_wire_bytes']),np.sign(a['action_miss_rate']-b['action_miss_rate'])) for a,b in pairs]
        denominator=float(np.sqrt(sum(x!=0 for x,y in signs)*sum(y!=0 for x,y in signs)))
        correlations.append(dict(condition=condition,kendall_tau_b=float(sum(x*y for x,y in signs))/denominator if denominator else None))
        for a,b in pairs:
            dv=a['mean_wire_bytes']-b['mean_wire_bytes']; dm=a['action_miss_rate']-b['action_miss_rate']
            difference=abs(dv)/max(a['mean_wire_bytes'],b['mean_wire_bytes'],1)
            paired=[]
            seeds=sorted({r['training_seed'] for r in episodes if r['condition']==condition})
            for seed in seeds:
                va=[float(r['return']) for r in episodes if r['condition']==condition and r['training_seed']==seed and r['method']==a['method']]
                vb=[float(r['return']) for r in episodes if r['condition']==condition and r['training_seed']==seed and r['method']==b['method']]
                if va and vb: paired.append(float(np.mean(va)-np.mean(vb)))
            rng=np.random.default_rng(582)
            delta_ci=np.quantile([np.mean(rng.choice(paired,len(paired),replace=True)) for _ in range(1000)],[.025,.975]).tolist() if len(paired)>=3 else None
            comparisons.append(dict(condition=condition,a=a['method'],b=b['method'],rank_inversion=dv*dm<0,
                rank_tie=dv*dm==0,budget_relative_difference=difference,matched_budget_eligible=difference<=tolerance,
                return_difference=a['mean_return']-b['mean_return'],return_difference_ci95_seed_bootstrap=delta_ci,
                interpretation='descriptive only; budget match is post-evaluation eligibility, not calibration'))
    result=dict(groups=list(groups.values()),comparisons=comparisons,rank_correlations=correlations,budget_tolerance=tolerance,
        limitations=['CPU timings are real; link delay is modeled.', 'No cross-policy causal guarantee from descriptive decomposition.',
            'One checkpoint or short smoke cannot establish learning superiority.', 'Thresholds must be frozen on separate calibration data.'])
    save_json(path/'summary.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); ap.add_argument('--budget-tolerance',type=float,default=.05)
    a=ap.parse_args(); analyze(a.run,a.budget_tolerance)
if __name__=='__main__': main()
