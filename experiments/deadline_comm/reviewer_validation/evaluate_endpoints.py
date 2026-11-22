"""Paired endpoint controls on every retained checkpoint, no checkpoint selection."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from ..timing import configure_cpu
from ..common import load_config
from ..day2.io import atomic_json,CSVLog,JSONLLog
from ..day2 import followup_harness as hm
from ..day2.finetune_comm_aware import load_model
from ..day2.aggregate_day2 import measure
from ..adapters.hallway import Hallway
from ..training_controls.run import controlled
from .endpoints import EndpointHarness
from .stats import paired_bootstrap


def checkpoints(spec):
    root=Path('experiments/deadline_comm/runs');items=[]
    for seed in spec['seeds']:
        for label,relative in [('base_1500',f'day2_dac_marl_main_20261118_001/seed{seed}/train/checkpoint_1500.pt'),
                               ('full_2000',f'day2_gap_closure_20261118_001/seed{seed}/checkpoint_0500.pt'),
                               ('adapted_2000',f'day2_followup_20261118_001/seed{seed}/checkpoint_0500.pt')]:
            items.append(dict(cohort=label,seed=seed,path=str(root/relative)))
    for file in sorted(Path(spec['parent_run']).glob('*/training_complete.json')):
        saved=json.loads(file.read_text());job=saved['job']
        if job['seed'] in spec['seeds']:
            items.append(dict(cohort=job['family']+'_'+job['arm'],seed=job['seed'],path=saved['final_checkpoint']))
    return [i for i in items if i['cohort'] in spec['cohorts']] if spec.get('cohorts') else items


def run(spec,smoke=False):
    configure_cpu();out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    parent=Path(spec['parent_run'])/'status.json'
    if not smoke and json.loads(parent.read_text())['stage']!='completed':
        recovery=parent.parent.with_name(parent.parent.name+'_evaluation_verified_002')/'status.json'
        if not recovery.exists() or json.loads(recovery.read_text())['stage']!='completed':
            raise RuntimeError('Parent or its documented evaluation recovery must complete')
    items=checkpoints(spec)
    if smoke:items=items[:1]
    atomic_json(out/'registration.json',dict(spec=spec,smoke=smoke,checkpoints=[dict(**i,sha256=hashlib.sha256(Path(i['path']).read_bytes()).hexdigest()) for i in items]))
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    records=[];returns={};start=time.monotonic()
    try:
        for item in items:
            model,meta=load_model(item['path']);env=Hallway(**meta['environment'])
            for link,bandwidth in spec['links'].items():
                for deadline in spec['deadlines_ms']:
                    for method in ('no_comm','ideal_full','periodic','dac','dac_zero_payload','dac_rotated_payload'):
                        if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Endpoint wall budget')
                        cfg=deepcopy(base);cfg['link']['bandwidth_bytes_s']=bandwidth;cfg['deadline_ns']=int(deadline*1e6)
                        cfg['endpoint']=method if method in ('no_comm','ideal_full') else None
                        cfg['message_intervention']='zero' if method=='dac_zero_payload' else 'rotate' if method=='dac_rotated_payload' else None
                        condition=f'{link}_{deadline:g}ms';label=f"{item['cohort']}_seed{item['seed']}_{condition}_{method}"
                        dest=out/label
                        prior=Path(spec['reuse_run'])/label if spec.get('reuse_run') else None
                        if prior is not None and (prior/'summary.json').exists():
                            row=json.loads((prior/'summary.json').read_text())
                            records.append(row)
                            dest.symlink_to(prior.resolve(),target_is_directory=True)
                            atomic_json(out/'status.json',dict(stage='evaluation',completed=len(records),reused=True))
                            continue
                        dest.mkdir()
                        def collect(reset):
                            from .rollouts import rollout
                            return controlled(rollout,env,model,'full' if cfg['endpoint'] else 'dac' if method.startswith('dac_') else method,cfg,reset,True)
                        for ep in range(spec['warmups']):collect(spec['reset_seed_start']-100+ep)
                        steps=[];episodes=[]
                        sl,el,ml=CSVLog(dest/'steps.csv'),CSVLog(dest/'episodes.csv'),JSONLLog(dest/'messages.jsonl')
                        try:
                            for ep in range(spec['episodes']):
                                ss,mm,ee=collect(spec['reset_seed_start']+ep);ee['method']=method
                                sl.append([dict(episode=ep,**r) for r in ss]);el.append([dict(episode=ep,**ee)]);ml.append([dict(episode=ep,**m) for m in mm])
                                steps.extend(ss);episodes.append(ee)
                            row=dict(cohort=item['cohort'],seed=item['seed'],condition=condition,method=method,**measure(episodes,steps))
                            records.append(row);atomic_json(dest/'summary.json',row)
                            returns[(item['cohort'],condition,method,item['seed'])]=row['mean_return']
                            atomic_json(out/'status.json',dict(stage='evaluation',completed=len(records),elapsed_seconds=time.monotonic()-start))
                        finally:sl.close();el.close();ml.close()
            print('ENDPOINTS',item['cohort'],item['seed'],flush=True)
        comparisons=[]
        for cohort,condition in sorted({(r['cohort'],r['condition']) for r in records}):
            for a,b in [('ideal_full','no_comm'),('dac','no_comm'),('periodic','no_comm'),('dac','periodic'),('dac','dac_zero_payload'),('dac','dac_rotated_payload')]:
                left={r['seed']:r['mean_return'] for r in records if r['cohort']==cohort and r['condition']==condition and r['method']==a}
                right={r['seed']:r['mean_return'] for r in records if r['cohort']==cohort and r['condition']==condition and r['method']==b}
                comparisons.append(dict(cohort=cohort,condition=condition,left=a,right=b,**paired_bootstrap(left,right,spec['bootstrap_resamples'])))
        atomic_json(out/'summary.json',dict(records=records,paired_comparisons=comparisons))
        for name,rows in [('by_seed.csv',records),('paired_ci.csv',comparisons)]:
            log=CSVLog(out/name);log.append(rows);log.close()
        lines=['# Communication endpoint validation','', 'Same weights and reset cohort. Ideal Full is an ideal-delivery reference, not a return optimum. No-Comm deployment differs from No-Comm training. Three-seed intervals are descriptive; episode repeats do not create independent training seeds.','', '|Checkpoint cohort|Condition|Difference|Mean|95% paired CI|n|','|---|---|---|---|---|---|']
        for r in comparisons:lines.append(f"|{r['cohort']}|{r['condition']}|{r['left']} minus {r['right']}|{r['mean_difference']:.2f}|[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]|{r['n_training_seeds']}|")
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        atomic_json(out/'status.json',dict(stage='completed',conditions=len(records),elapsed_seconds=time.monotonic()-start))
    except Exception as exc:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc),completed=len(records)));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();spec=load_config(a.config)
    if a.smoke:spec.update(output=spec['output']+'_smoke',seeds=[0],links={'slow':20000},deadlines_ms=[5.],episodes=2,warmups=1)
    run(spec,a.smoke)
