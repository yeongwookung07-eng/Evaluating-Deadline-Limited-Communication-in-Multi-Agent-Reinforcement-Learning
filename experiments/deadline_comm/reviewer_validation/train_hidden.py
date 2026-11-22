"""Demonstration-free hidden-source training; scaling conditional on N5 learning."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..timing import configure_cpu
from ..common import load_config
from ..day2.io import atomic_json,JSONLLog,CSVLog
from .hidden_source import HiddenSource
from .train_ten_seeds import trajectory,configure_arm
from .stats import paired_bootstrap


def train(spec,job,out,device):
    import numpy as np
    import torch
    from ..day2.policy import DecayedPolicy
    from ..oracle_bc_learner import OracleBCLearner
    from ..replay import CommReplay
    from ..training_controls.run import controlled
    torch.manual_seed(job['seed']);env=HiddenSource(job['n'],job['task'],spec['horizon'])
    model=DecayedPolicy(4,env.state_dim,env.n_agents,2,hidden_dim=64,message_dim=8,mixer='qmix')
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    method,cfg=configure_arm(base,job['method'],100000,2.)
    model.decay_lambda=cfg['decay_lambda'];model.to(device)
    lc=dict(optimizer='adam',learning_rate=5e-4,gamma=.99,grad_clip=10.,target_update_interval=100,bootstrap_timeouts=False,bc_weight=0.)
    learner=OracleBCLearner(model,lc);roll=deepcopy(model).cpu();replay=CommReplay(1000,job['seed'])
    meta=dict(job=job,device=device,learner=lc,oracle_episodes=0,bc_weight=0,updates=spec['updates'])
    atomic_json(out/'manifest.json',meta)
    def collect(index,eps):
        roll.load_state_dict(model.state_dict())
        return controlled(trajectory,env,roll,method,cfg,9100000+job['seed']*10000+index,False,eps)
    for i in range(spec['batch_size']):replay.insert(collect(i,1.))
    start=time.monotonic();log=JSONLLog(out/'training.jsonl')
    try:
        for u in range(spec['updates']):
            record=collect(100+u,1.-.95*u/max(1,spec['updates']-1));replay.insert(record)
            metrics=learner.train_batch(replay.sample(spec['batch_size']))
            log.append([dict(update=u+1,episode_return=float(record.episode.rewards.sum()),elapsed=time.monotonic()-start,**metrics)])
            if (u+1)%250==0:print(job,u+1,metrics,flush=True)
        torch.save(dict(model=model.cpu().state_dict(),metadata=meta),out/'final.pt')
        atomic_json(out/'complete.json',dict(seconds=time.monotonic()-start,**meta))
    finally:log.close()


def evaluate(spec,jobs,out):
    import torch
    from ..day2.policy import DecayedPolicy
    from ..day2.aggregate_day2 import measure
    from ..training_controls.run import controlled
    from .rollouts import rollout
    records=[]
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    for j in jobs:
        env=HiddenSource(j['n'],j['task'],spec['horizon']);model=DecayedPolicy(4,env.state_dim,j['n'],2,hidden_dim=64,message_dim=8,mixer='qmix')
        model.load_state_dict(torch.load(out/j['name']/'final.pt',weights_only=True,map_location='cpu')['model'])
        for link,bandwidth,deadline in [('medium',100000,5.),('medium',100000,2.),('slow',20000,5.)]:
            method,cfg=configure_arm(base,j['method'],bandwidth,deadline)
            dest=out/j['name']/f'eval_{link}_{deadline:g}ms';dest.mkdir()
            for ep in range(2):controlled(rollout,env,model,method,cfg,9400000+ep,True)
            sl,el=CSVLog(dest/'steps.csv'),CSVLog(dest/'episodes.csv');steps=[];episodes=[]
            try:
                for ep in range(spec['evaluation_episodes']):
                    ss,_,ee=controlled(rollout,env,model,method,cfg,9500000+ep,True)
                    sl.append([dict(episode=ep,**r) for r in ss]);el.append([dict(episode=ep,**ee)])
                    steps.extend(ss);episodes.append(ee)
                row=dict(task=j['task'],n_agents=j['n'],seed=j['seed'],method=j['method'],condition=f'{link}_{deadline:g}ms',**measure(episodes,steps))
                row['accuracy']=row['mean_return']/spec['horizon'];records.append(row);atomic_json(dest/'summary.json',row)
            finally:sl.close();el.close()
    return records


def main():
    configure_cpu();ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--worker');ap.add_argument('--output');ap.add_argument('--device');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    spec=load_config(a.config)
    if a.worker:train(spec,json.loads(a.worker),Path(a.output),a.device);return
    if a.smoke:spec.update(output=spec['output']+'_smoke',seeds=[0],agent_counts=[5],updates=2,batch_size=4,evaluation_episodes=2,devices=['cpu'])
    out=Path(spec['output']);out.mkdir(exist_ok=False)
    import yaml
    config=out/'resolved.yaml';config.write_text(yaml.safe_dump(spec))
    atomic_json(out/'registration.json',dict(spec=spec,max_jobs=320,max_updates=320000,scope='Fresh custom information tasks, not native benchmark reproduction'))
    records=[];start=time.monotonic();eligible=list(spec['tasks']);running=[]
    try:
        for n in spec['agent_counts']:
            jobs=[dict(name=f'{task}_n{n}_seed{seed}_{method}',task=task,n=n,seed=seed,method=method) for task in eligible for seed in spec['seeds'] for method in spec['methods']]
            pending=list(jobs)
            while pending or running:
                if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Hidden-task wall budget')
                used={r[2] for r in running}
                for device in spec['devices']:
                    if device in used or not pending:continue
                    j=pending.pop(0);dest=out/j['name'];dest.mkdir();log=(dest/'stdout.log').open('x')
                    cmd=[sys.executable,'-u','-m','experiments.deadline_comm.reviewer_validation.train_hidden','--config',str(config),'--worker',json.dumps(j),'--output',str(dest),'--device',device]
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);running.append((p,j,device,log))
                for r in list(running):
                    code=r[0].poll()
                    if code is not None:
                        r[3].close();running.remove(r)
                        if code:raise RuntimeError(f"Worker failed: {r[1]}")
                atomic_json(out/'status.json',dict(stage='training',n_agents=n,running=[r[1]['name'] for r in running],pending=len(pending)))
                time.sleep(2)
            atomic_json(out/'status.json',dict(stage='cpu_evaluation',n_agents=n))
            records.extend(evaluate(spec,jobs,out));atomic_json(out/'records.json',records)
            if n==5:
                eligible=[task for task in eligible if sum(r['accuracy']>=spec['min_ideal_accuracy'] for r in records if r['task']==task and r['method']=='ideal_full' and r['condition']=='medium_5ms')>=spec['min_ideal_successful_seeds']]
                atomic_json(out/'scaling_gate.json',dict(eligible_tasks=eligible,criterion='N5 ideal link accuracy >= .75 in >=8 of 10 seeds',skipped_tasks=[t for t in spec['tasks'] if t not in eligible]))
                if not eligible:break
        ci=[]
        for task,n,condition in sorted({(r['task'],r['n_agents'],r['condition']) for r in records}):
            for a,b in [('dac','no_comm'),('ideal_full','no_comm'),('dac','periodic')]:
                left={r['seed']:r['mean_return'] for r in records if (r['task'],r['n_agents'],r['condition'],r['method'])==(task,n,condition,a)}
                right={r['seed']:r['mean_return'] for r in records if (r['task'],r['n_agents'],r['condition'],r['method'])==(task,n,condition,b)}
                ci.append(dict(task=task,n_agents=n,condition=condition,left=a,right=b,**paired_bootstrap(left,right)))
        atomic_json(out/'summary.json',dict(records=records,paired_ci=ci,scaling_eligible_tasks=eligible))
        atomic_json(out/'status.json',dict(stage='completed',elapsed_seconds=time.monotonic()-start))
    except Exception as exc:
        for p,j,d,log in running:p.terminate()
        for p,j,d,log in running:p.wait();log.close()
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise

if __name__=='__main__':main()
