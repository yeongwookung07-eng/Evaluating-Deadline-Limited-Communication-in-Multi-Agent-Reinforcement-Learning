"""Persistent-network policy stress test; same fixed weights across gates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
from ..common import load_config
from ..timing import configure_cpu
from ..adapters.hallway import Hallway
from ..day2.finetune_comm_aware import load_model
from ..day2.io import atomic_json,CSVLog
from ..training_controls.run import controlled
from .queued_harness import QueuedHarness
from .stats import paired_bootstrap


def episode(env,model,method,cfg,seed):
    obs,_=env.reset(seed=seed);h=QueuedHarness(model,method,cfg,seed,True);rows=[];total=0.
    for t in range(env.episode_length):
        a,r=h.step(obs,env.available_actions(),t);obs,reward,term,trunc,_=env.step(a)
        rows.append(r);total+=reward
        if term or trunc:break
    return rows,total


def main(spec):
    configure_cpu();out=Path(spec['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',spec)
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness'];records=[];start=time.monotonic()
    for seed in spec['seeds']:
        model,meta=load_model(spec['checkpoint_template'].format(seed=seed));env=Hallway(**meta['environment'])
        for link,deadline in spec['conditions']:
            for loss in spec['loss']:
                for retries in spec['retries']:
                    for method in ('no_comm','full','periodic','dac'):
                        if time.monotonic()-start>spec['wall_hours']*3600:raise TimeoutError('Queued evaluation wall budget')
                        cfg=deepcopy(base);cfg.update(loss=loss,retries=retries,deadline_ns=int(deadline*1e6));cfg['link']['bandwidth_bytes_s']=spec['links'][link]
                        label=f'seed{seed}_{link}_{deadline:g}ms_p{loss}_r{retries}_{method}';dest=out/label;dest.mkdir()
                        for i in range(2):controlled(episode,env,model,method,cfg,9600000+i)
                        sl,el=CSVLog(dest/'steps.csv'),CSVLog(dest/'episodes.csv');rows=[];scores=[]
                        try:
                            for ep in range(spec['episodes']):
                                rr,ret=controlled(episode,env,model,method,cfg,9700000+ep)
                                sl.append([dict(episode=ep,**r) for r in rr]);el.append([dict(episode=ep,return_value=ret)]);rows.extend(rr);scores.append(ret)
                            row=dict(seed=seed,condition=f'{link}_{deadline:g}ms_p{loss}_r{retries}',method=method,mean_return=statistics.mean(scores),
                                     offered_B_step=statistics.mean(r['offered_bytes'] for r in rows),payload_B_step=statistics.mean(r['received_payload_bytes'] for r in rows),
                                     wire_started_B_step=statistics.mean(r['wire_bytes_started'] for r in rows),retry_B_step=statistics.mean(r['retry_bytes'] for r in rows),
                                     mean_backlog_packets=statistics.mean(r['backlog_packets'] for r in rows),invalid_fraction=statistics.mean(r['invalid_fraction'] for r in rows),
                                     old_packet_arrivals=sum(r['received_old_packets'] for r in rows),action_misses=sum(r['action_miss'] for r in rows))
                            records.append(row);atomic_json(dest/'summary.json',row);atomic_json(out/'progress.json',records)
                        finally:sl.close();el.close()
    ci=[]
    for condition in sorted({r['condition'] for r in records}):
        for a,b in [('dac','periodic'),('dac','no_comm'),('full','no_comm')]:
            left={r['seed']:r['mean_return'] for r in records if r['condition']==condition and r['method']==a};right={r['seed']:r['mean_return'] for r in records if r['condition']==condition and r['method']==b}
            ci.append(dict(condition=condition,left=a,right=b,**paired_bootstrap(left,right)))
    atomic_json(out/'summary.json',dict(records=records,paired_ci=ci,scope='Persistent simulated channel, fixed-weight deployment stress test. Busy-radio waits to cutoff; not the original latency protocol. Retransmission/ACK energy and actual wireless traces excluded.'))
    log=CSVLog(out/'by_seed.csv');log.append(records);log.close()
    atomic_json(out/'status.json',dict(stage='completed',seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();s=load_config(a.config)
    if a.smoke:s.update(output=s['output']+'_smoke',checkpoint_template='experiments/deadline_comm/runs/day2_followup_20261118_001/seed{seed}/checkpoint_0500.pt',seeds=[0],conditions=[['slow',5.]],loss=[.1],retries=[2],episodes=2)
    main(s)
