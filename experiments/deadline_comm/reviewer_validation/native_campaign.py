"""Bounded native-author learner validation, separate task/protocol from DAC."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from ..common import load_config
from ..day2.io import atomic_json,CSVLog
from .stats import paired_bootstrap

p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();spec=load_config(a.config)
out=Path(spec['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',spec)
root=Path(__file__).resolve().parent;exe=root/'native_env/bin/python';start=time.monotonic();rows=[]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
try:
    for seed in spec['seeds']:
        for control in spec['controls']:
            remaining=spec['wall_hours']*3600-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Native wall budget')
            dest=out/f'seed{seed}_{control}'
            with (out/f'seed{seed}_{control}.log').open('x') as log:
                subprocess.run([str(exe),str(root/'native_launcher.py'),'--output',str(dest),'--seed',str(seed),'--control',control,'--steps',str(spec['steps']),'--testing-steps',str(spec['testing_steps'])],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=remaining)
            text=(out/f'seed{seed}_{control}.log').read_text()
            matches=re.findall(r'Test result: Average steps to capture:\s+\S+\s+([-+\d.eE]+)\s+([-+\d.eE]+)',text)
            if not matches:raise RuntimeError('Missing native evaluation output')
            steps,reward=map(float,matches[-1]);rows.append(dict(seed=seed,control=control,average_steps=steps,reported_reward=reward))
            atomic_json(out/'progress.json',rows)
    left={r['seed']:r['reported_reward'] for r in rows if r['control']=='native'};right={r['seed']:r['reported_reward'] for r in rows if r['control']=='round_robin'}
    atomic_json(out/'summary.json',dict(records=rows,paired=paired_bootstrap(left,right),scope=spec['scope']))
    log=CSVLog(out/'by_seed.csv');log.append(rows);log.close();atomic_json(out/'status.json',dict(stage='completed'))
except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
