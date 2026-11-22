"""Analytic-task and network scaling checks, not trained-policy evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..common import load_config
from ..day2.io import atomic_json,CSVLog
from .hidden_source import HiddenSource
from .queued_link import QueuedLink


def run(spec):
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False);atomic_json(out/'registration.json',spec)
    tasks=[];network=[]
    for n in spec['agents']:
        for seed in spec['seeds']:
            for task in ('broadcast','xor'):
                env=HiddenSource(n,task);random_scores=[];oracle_scores=[]
                rng=np.random.default_rng(seed)
                for ep in range(spec['hidden_episodes']):
                    for oracle,store in [(False,random_scores),(True,oracle_scores)]:
                        env.reset(seed=900000+seed*10000+ep)
                        for _ in range(env.episode_length):
                            a=env.oracle_actions() if oracle else rng.integers(0,2,n)
                            _,r,_,_,_=env.step(a);store.append(r)
                tasks.append(dict(n_agents=n,seed=seed,task=task,random_accuracy=float(np.mean(random_scores)),oracle_accuracy=float(np.mean(oracle_scores)),no_comm_expected_upper_bound=.5,learned_policy=False))
            for p in spec['loss_probabilities']:
                for retries in spec['retries']:
                    for method in ('full','periodic','cap'):
                        q=QueuedLink(spec['bandwidth_bytes_s'],p,retries,.00005,seed)
                        source=np.full(n,-1);delivered=0;late=0;ages=[];offered=0
                        for t in range(spec['network_steps']):
                            def receive(events):
                                nonlocal delivered,late
                                for event in events:
                                    delivered+=1;late+=int(event['arrival_s']>(event['source_step']+1)*spec['step_seconds']-.00045)
                                    i=event['sender'];source[i]=max(source[i],event['source_step'])
                            receive(q.advance(t*spec['step_seconds']+.0005))
                            senders=range(n) if method=='full' else [i for i in range(n) if (i+t)%2==0] if method=='periodic' else [t%n]
                            for i in senders:q.submit(i,t,48);offered+=48
                            receive(q.advance((t+1)*spec['step_seconds']-.00045))
                            valid=source>=0
                            if valid.any():ages.extend((t-source[valid]).tolist())
                        network.append(dict(n_agents=n,seed=seed,method=method,loss=p,max_retries=retries,
                                            offered_B_step=offered/spec['network_steps'],received_payload_B_step=32*delivered/spec['network_steps'],
                                            late_arrivals=late,mean_age=float(np.mean(ages)) if ages else None,**q.snapshot()))
    for name,rows in [('hidden_tasks.csv',tasks),('network_scaling.csv',network)]:
        log=CSVLog(out/name);log.append(rows);log.close()
    atomic_json(out/'summary.json',dict(status='completed',scope='Untrained task checks and modeled queue scaling, not MARL scalability validation',hidden=tasks,network=network))
    (out/'REPORT.md').write_text('# Diagnostic checks\n\nHidden broadcast and XOR: fresh private bits, theoretical No-Comm expected follower accuracy <= 0.5, bit-sharing Oracle accuracy 1.0. See hidden_tasks.csv. These are custom diagnostic tasks, not standard benchmark reproductions.\n\nnetwork_scaling.csv includes N5/10/20/50, ten loss seeds, bounded retries and persistent queues. It does not contain learned control returns or real-network traces. Bytes started can include a partially transmitted last packet; counters are explicitly named.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(load_config(a.config))
