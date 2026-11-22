"""Registered sequential gap closure, preserving upstream code and run artifacts.

Question: distinguish additional updates, decay at deployment, and scored slot
selection. Controls: matched +500 Full-Comm updates and score-free public slots.
All seeds are primary; convergence-selected estimates are descriptive supplements.
No improvement is assumed. CPU samples are measured, link waiting is modeled.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import yaml
from ..timing import configure_cpu
from ..adapters.hallway import Hallway
from . import finetune_comm_aware as ft
from . import followup_harness as harness_module
from .aggregate_day2 import aggregate, measure, read_main
from .decentralized_slot_only import DecentralizedSlotOnlyGate
from .io import atomic_json, CSVLog, JSONLLog

OriginalHarness = harness_module.Day2Harness
PHASES = ('encode_ns','gate_ns','cache_ns','act_ns','other_ns','cpu_total_ns','step_total_ns')


class SlotHarness(OriginalHarness):
    def __init__(self, model, method, cfg, seed, measured=True):
        super().__init__(model, method, cfg, seed, measured)
        if cfg.get('slot_only'):
            self.dac = DecentralizedSlotOnlyGate(model.n_agents, seed=seed,
                         probability=cfg['slot_probability'], **cfg['dac'])


class FullTrainingHarness(OriginalHarness):
    def __init__(self, model, method, cfg, seed, measured=True):
        cfg = deepcopy(cfg)
        cfg['link'].update(bandwidth_bytes_s=1000000000, latency_ns=0)
        cfg['deadline_ns'] = 1000000000
        super().__init__(model, 'full', cfg, seed, measured)

    def step(self, *args):
        result = super().step(*args)
        assert result[1]['K_delivered'] == self.model.n_agents
        return result


def train_full(checkpoint, cfg, out, seed, updates, device):
    cfg = deepcopy(cfg)
    cfg['decay_lambda'] = 1.
    # Dependency substitution is scoped to this sequential process, never a file edit.
    with patch.object(ft, 'Day2Harness', FullTrainingHarness):
        return ft.finetune(checkpoint, cfg, out, seed, updates, device)


def evaluation_config(cfg, arm):
    cfg = deepcopy(cfg)
    if arm == 'comm_aware_lambda1':
        cfg['decay_lambda'] = 1.
    return cfg


def controlled(env, model, method, cfg, seed, measured=True):
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with patch.object(harness_module, 'Day2Harness', SlotHarness):
            return harness_module.rollout(env, model, method, cfg, seed, measured)
    finally:
        if enabled:
            gc.enable()


def calibrate(model, metadata, cfg, spec, seed, slot, out):
    env = Hallway(**metadata['environment'])
    candidates = []
    low, high = (0., 1.) if slot else (-10., 20.)
    for index in range(spec['calibration_iterations']+2):
        value = low if index == 0 else high if index == 1 else (low+high)/2
        local = deepcopy(cfg)
        local['slot_only'] = slot
        if slot:
            local['slot_probability'] = value
        else:
            local['dac']['threshold'] = value
        local['deadline_ns'] = 5000000
        local['link']['bandwidth_bytes_s'] = 100000
        rows = []
        for ep in range(spec['calibration_episodes']):
            steps, _, _ = controlled(env,model,'dac_decentralized',local,4100000+seed*10000+ep)
            rows.extend(steps)
        rate = sum(r['attempted_bytes'] for r in rows)/len(rows)
        candidates.append(dict(value=value,offered_B_step=rate,error=abs(rate-120)))
        if index >= 2:
            if (rate < 120) == slot:
                low = value
            else:
                high = value
    best = min(candidates,key=lambda row:row['error'])
    result = deepcopy(cfg)
    result['slot_only'] = slot
    if slot:
        result['slot_probability'] = best['value']
    else:
        result['dac']['threshold'] = best['value']
    atomic_json(out/('slot_calibration.json' if slot else 'scored_calibration.json'),
                dict(candidates=candidates,selected=best,eligible=best['error']<=6,
                     frozen_harness=result,calibration_condition='medium_5ms'))
    return result


def evaluate(model, metadata, cfg, spec, seed, arm, out, conditions):
    cfg = evaluation_config(cfg,arm)
    method = 'full' if arm=='full_2k' else 'dac_decentralized' if arm in ('scored_calibrated','slot_only') else 'dac'
    env = Hallway(**metadata['environment'])
    records = []
    for link, deadline in conditions:
        label = f'{link}_{deadline:g}ms'
        local = deepcopy(cfg)
        local['deadline_ns'] = int(deadline*1e6)
        local['link']['bandwidth_bytes_s'] = spec['links'][link]
        destination = out/f'{arm}_{label}'
        destination.mkdir()
        for ep in range(2):
            controlled(env,model,method,local,2500000+ep)
        sl, el = CSVLog(destination/'steps.csv'), CSVLog(destination/'episodes.csv')
        ml = JSONLLog(destination/'messages.jsonl')
        steps, episodes = [], []
        try:
            for repeat in range(spec['repeats']):
                for ep in range(spec['episodes']):
                    ss, mm, ee = controlled(env,model,method,local,2200000+ep)
                    tags = dict(repeat=repeat,episode=ep,arm=arm)
                    sl.append([{**tags,**r} for r in ss])
                    el.append([{**tags,**ee}])
                    ml.append([{**tags,**r} for r in mm])
                    steps.extend(ss)
                    episodes.append(ee)
            row = dict(seed=seed,category='gap_closure',condition=label,method=arm,**measure(episodes,steps))
            atomic_json(destination/'summary.json',row)
            records.append(row)
            print('EVALUATED',seed,arm,label,flush=True)
        finally:
            sl.close(); el.close(); ml.close()
    return records


def timing_statistics(rows):
    return {phase:dict(mean=float(np.mean([float(r[phase])/1000 for r in rows])),
                       median=float(np.median([float(r[phase])/1000 for r in rows])),
                       p95=float(np.quantile([float(r[phase])/1000 for r in rows],.95)))
            for phase in PHASES}


def timing_report(roots):
    result = []
    for root in roots:
        for path in sorted(root.glob('seed*/*/steps.csv')):
            groups = {}
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    method = row.get('arm',row['method'])
                    groups.setdefault(method,[]).append(row)
            for method, rows in groups.items():
                result.append(dict(run=root.name,seed=int(path.parts[-3][4:]),
                                   condition=path.parent.name,method=method,
                                   timings_us=timing_statistics(rows)))
    return result


def write_report(spec, out, records):
    main, convergence = read_main(Path(spec['main_run']))
    follow = []
    for file in Path(spec['followup_run']).glob('seed*/evaluation.json'):
        follow.extend(json.loads(file.read_text()))
    tables = aggregate(main+follow+records,convergence)
    timings = timing_report([Path(spec['main_run']),Path(spec['followup_run']),out])
    atomic_json(out/'final_tables.json',dict(convergence=convergence,tables=tables,records=records,timings=timings))
    lines = ['# Day 2 gap closure','',
             'Primary estimates include all three seeds. Standard deviations are across independent training seeds.',
             'Conditional Return >=5 estimates are selected descriptive supplements. Timing repeats reuse environment seeds.',
             'Full-Comm continuation matches optimizer reset, target reset, learning rate, Oracle ratio and update count.',
             'Oracle observations/actions are matched by generator seeds; delivery histories differ with training condition.',
             'Slot-only and scored slots have separate medium-5ms calibration. Eligibility is measured in every condition.',
             'The public slot schedule assumes synchronized clock and known IDs. No optimality or scalability guarantee follows.',
             'Evaluation is sequential, CPU single-thread, R_post=450us, GC disabled inside episodes, two warmups excluded.',
             'Training uses the configured CUDA device. Timing smoke values are wiring checks only.',
             'Raw CPU time and virtual communication wait are distinct. step_total includes the modeled wait.','']
    for cohort in ('all_seeds','converged_seeds'):
        lines += ['## '+cohort,'','| Condition | Method | Seeds | Return | Offered B/step | Payload B/step | Waste % | Invalid % | Drop % | Budget eligible |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for row in tables:
            if row['cohort'] != cohort:continue
            values=[]
            for metric in ('mean_return','offered_B_step','goodput_payload_B_step','wasted_pct','cache_invalid_pct','drop_pct'):
                value=row['metrics'][metric]
                values.append('NA' if value['mean'] is None else f"{value['mean']:.3f} ± "+('NA' if value['std'] is None else f"{value['std']:.3f}"))
            lines.append('| '+' | '.join([row['condition'],row['method'],str(row['seeds'])]+values+[str(row['all_budget_eligible'])])+' |')
        lines.append('')
    lines += ['## CPU breakdown per seed and condition','',
              '| Run | Seed | Condition | Method | Phase | Mean us | Median us | p95 us |',
              '|---|---|---|---|---|---|---|---|']
    for row in timings:
        if row['method'] not in ('vbc','vbc_batched','dac','dac_decentralized','scored_calibrated','slot_only'):continue
        for phase, values in row['timings_us'].items():
            lines.append(f"| {row['run']} | {row['seed']} | {row['condition']} | {row['method']} | {phase} | {values['mean']:.3f} | {values['median']:.3f} | {values['p95']:.3f} |")
    (out/'DAY2_GAP_CLOSURE_REPORT.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    spec=yaml.safe_load(Path(args.config).read_text())
    configure_cpu()
    if args.smoke:
        spec.update(output=spec['output']+'_smoke',seeds=[0],updates=1,episodes=1,repeats=1,
                    calibration_episodes=1,calibration_iterations=1,
                    links={'medium':100000,'slow':20000},deadlines_ms=[5.])
    out=Path(spec['output'])
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting',config=spec,smoke=args.smoke)
    atomic_json(out/'status.json',state)
    atomic_json(out/'registration.json',dict(question=__doc__,config=spec,
        control='Matched Full-Comm +500 updates and slot-only gate',
        falsification='No improvement, budget drift, or decay removal gains remain negative findings.',
        budget='3 seeds x500 updates; 4 arms x12 conditions plus3 no-decay conditions;20 resets x3 timing repeats',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(args.config),Path(__file__).with_name('decentralized_slot_only.py')]}))
    records=[]
    try:
        if not args.smoke:
            while True:
                status=json.loads((Path(spec['followup_run'])/'status.json').read_text())
                if status['status']=='completed':break
                if status['status']=='failed':raise RuntimeError(status.get('error','Followup failed'))
                print('WAITING',status['status'],flush=True)
                time.sleep(60)
        for seed in spec['seeds']:
            state.update(status='running',seed=seed)
            atomic_json(out/'status.json',state)
            dest=out/f'seed{seed}'
            dest.mkdir()
            source=Path(spec['main_run'])/f'seed{seed}'
            checkpoint=source/'train/checkpoint_1500.pt'
            model,metadata=ft.load_model(checkpoint)
            cfg=json.loads((source/'frozen_harness.json').read_text())
            conditions=[(link,d) for link in spec['links'] for d in spec['deadlines_ms']]
            for slot,arm in [(False,'scored_calibrated'),(True,'slot_only')]:
                calibrated=calibrate(model,metadata,cfg,spec,seed,slot,dest)
                records.extend(evaluate(model,metadata,calibrated,spec,seed,arm,dest,conditions))
                atomic_json(out/'evaluation.json',records)
            trained=train_full(checkpoint,cfg,dest,seed,spec['updates'],spec['devices'][seed])
            for arm in ('full_2k','dac_on_2k'):
                records.extend(evaluate(trained,metadata,cfg,spec,seed,arm,dest,conditions))
                atomic_json(out/'evaluation.json',records)
            adapted=Path(spec['followup_run'])/f'seed{seed}/checkpoint_0500.pt'
            if args.smoke and not adapted.exists():
                adapted=dest/'checkpoint_0001.pt'
                atomic_json(dest/'smoke_checkpoint_substitution.json',dict(reason='Comm-aware checkpoint not available yet',used=str(adapted),scientific_result=False))
            model,metadata=ft.load_model(adapted)
            records.extend(evaluate(model,metadata,cfg,spec,seed,'comm_aware_lambda1',dest,spec['bottlenecks']))
            atomic_json(out/'evaluation.json',records)
        write_report(spec,out,records)
        state.update(status='completed')
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        atomic_json(out/'status.json',state)


if __name__=='__main__':main()
