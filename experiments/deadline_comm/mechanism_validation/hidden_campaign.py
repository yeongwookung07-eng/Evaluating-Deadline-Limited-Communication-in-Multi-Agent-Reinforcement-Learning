"""Bounded N5 campaign using existing hidden-task learner; adds equal-cap RR."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..common import load_config
from ..timing import configure_cpu
from ..day2.io import atomic_json
from ..reviewer_validation.train_hidden import evaluate
from ..reviewer_validation.stats import paired_bootstrap


def main(spec):
    configure_cpu();out=Path(spec['output']);out.mkdir(exist_ok=False)
    jobs=[dict(name=f'{task}_n5_seed{seed}_{method}',task=task,n=5,seed=seed,method=method) for seed in spec['seeds'] for task in spec['tasks'] for method in spec['methods']]
    atomic_json(out/'registration.json',dict(spec=spec,max_jobs=len(jobs),max_updates=len(jobs)*spec['updates'],scope='N5 only; no demonstrations; no scaling or native scheduler'))
    import yaml
    resolved=out/'resolved.yaml';resolved.write_text(yaml.safe_dump(spec))
    pending=list(jobs);running=[];completed=[];start=time.monotonic()
    try:
        while pending or running:
            if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Training wall budget')
            used={r[2] for r in running}
            for device in spec['devices']:
                if device in used or not pending:continue
                j=pending.pop(0);dest=out/j['name'];dest.mkdir();log=(dest/'stdout.log').open('x')
                cmd=[sys.executable,'-u','-m','experiments.deadline_comm.reviewer_validation.train_hidden','--config',str(resolved),'--worker',json.dumps(j),'--output',str(dest),'--device',device]
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);running.append((proc,j,device,log))
                print('START',j['name'],device,flush=True)
            for r in list(running):
                code=r[0].poll()
                if code is not None:
                    r[3].close();running.remove(r)
                    if code:raise RuntimeError(f"Worker {r[1]['name']} exit {code}")
                    completed.append(r[1]['name']);print('COMPLETE',r[1]['name'],len(completed),'/',len(jobs),flush=True)
            atomic_json(out/'status.json',dict(stage='training',completed=len(completed),total=len(jobs),running=[dict(job=r[1]['name'],device=r[2],pid=r[0].pid) for r in running],pending=len(pending),elapsed_seconds=time.monotonic()-start))
            time.sleep(2)
        records=[]
        for j in jobs:
            if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Campaign wall budget')
            atomic_json(out/'status.json',dict(stage='cpu_evaluation',job=j['name'],completed_cells=len(records),total_cells=3*len(jobs)))
            records.extend(evaluate(spec,[j],out));atomic_json(out/'records.json',records)
        ci=[]
        for task,condition in sorted({(r['task'],r['condition']) for r in records}):
            for left,right in [('ideal_full','no_comm'),('dac','no_comm'),('dac','periodic'),('dac','rr_cap')]:
                get=lambda method:{r['seed']:r['mean_return'] for r in records if (r['task'],r['condition'],r['method'])==(task,condition,method)}
                ci.append(dict(task=task,condition=condition,left=left,right=right,**paired_bootstrap(get(left),get(right))))
        atomic_json(out/'summary.json',dict(records=records,paired_ci=ci,limitations='Custom N5 tasks; no demonstrations; ideal-link learning must be checked before interpreting selector effects'))
        atomic_json(out/'status.json',dict(stage='completed',jobs=len(jobs),cells=len(records),elapsed_seconds=time.monotonic()-start))
    except Exception as exc:
        for proc,j,d,log in running:proc.terminate()
        for proc,j,d,log in running:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc),completed_jobs=completed));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();s=load_config(a.config)
    if a.smoke:s.update(output=s['output']+'_smoke',seeds=[0],updates=2,batch_size=4,evaluation_episodes=2,devices=['cuda:0','cuda:1'])
    main(s)
