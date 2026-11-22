"""Wait for the registered main run, then execute bounded follow-up experiments."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from ..timing import configure_cpu
from ..adapters.hallway import Hallway
from .finetune_comm_aware import load_model, finetune
from .followup_harness import rollout
from .aggregate_day2 import measure, report
from .io import atomic_json, CSVLog, JSONLLog


def evaluate(model, metadata, cfg, spec, seed, arm, out):
    env = Hallway(**metadata['environment'])
    method = 'dac_decentralized' if arm=='dac_decentralized' else 'dac'
    records=[]
    def run(local, env_seed):
        gc.collect()
        enabled=gc.isenabled()
        gc.disable()
        try:return rollout(env,model,method,local,env_seed,True)
        finally:
            if enabled:gc.enable()
    for link, bandwidth in spec['links'].items():
        for deadline in spec['deadlines_ms']:
            label=f'{link}_{deadline:g}ms'
            local=deepcopy(cfg)
            local['deadline_ns']=int(deadline*1e6)
            local['link']['bandwidth_bytes_s']=bandwidth
            destination=out/f'{arm}_{label}'
            destination.mkdir()
            for ep in range(2):run(local,2500000+ep)
            sl,el,ml=CSVLog(destination/'steps.csv'),CSVLog(destination/'episodes.csv'),JSONLLog(destination/'messages.jsonl')
            steps,episodes=[],[]
            try:
                for repeat in range(spec['repeats']):
                    for ep in range(spec['episodes']):
                        ss,mm,ee=run(local,2200000+ep)
                        tags=dict(repeat=repeat,episode=ep)
                        sl.append([{**tags,**r} for r in ss])
                        el.append([{**tags,**ee}]);ml.append([{**tags,**r} for r in mm])
                        steps.extend(ss);episodes.append(ee)
                record=dict(seed=seed,category='extension',condition=label,method=arm,**measure(episodes,steps))
                records.append(record);atomic_json(destination/'summary.json',record)
                print('EVALUATED',seed,arm,label,flush=True)
            finally:sl.close();el.close();ml.close()
    return records


def worker(spec, seed, out):
    main=Path(spec['main_run'])
    source=main/f'seed{seed}'
    checkpoint=source/'train/checkpoint_1500.pt'
    cfg=json.loads((source/'frozen_harness.json').read_text())
    model,metadata=load_model(checkpoint)
    records=evaluate(model,metadata,cfg,spec,seed,'dac_decentralized',out)
    atomic_json(out/'evaluation.json',records)
    model=finetune(checkpoint,cfg,out,seed,spec['updates'],f'cuda:{seed}')
    records.extend(evaluate(model,metadata,cfg,spec,seed,'dac_comm_aware',out))
    atomic_json(out/'evaluation.json',records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    spec=yaml.safe_load(Path(args.config).read_text())
    configure_cpu()
    out=Path(spec['output'])
    if args.seed is not None:
        worker(spec,args.seed,out/f'seed{args.seed}')
        return
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting',config=spec)
    atomic_json(out/'status.json',state)
    try:
        while True:
            source=json.loads((Path(spec['main_run'])/'manifest.json').read_text())
            if source['status']=='completed':break
            if source['status']=='failed':raise RuntimeError(source.get('error','Main failed'))
            print('WAITING',source['phase'],source['seed'],flush=True)
            time.sleep(60)
        report(Path(spec['main_run']),out)
        for seed in spec['seeds']:
            # Sequential CPU timing avoids contention between our evaluation jobs.
            (out/f'seed{seed}').mkdir()
            state.update(status='running',seed=seed)
            atomic_json(out/'status.json',state)
            with (out/f'seed{seed}/worker.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m',__package__+'.post_pipeline','--config',args.config,'--seed',str(seed)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        report(Path(spec['main_run']),out,True)
        state.update(status='completed')
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:atomic_json(out/'status.json',state)


if __name__=='__main__':main()
