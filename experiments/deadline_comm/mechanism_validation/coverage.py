"""Sender coverage and reward-adjacent freshness; descriptive, not causal."""
import csv
import json
from pathlib import Path
import statistics
from ..day2.io import atomic_json,CSVLog
from ..common import load_config


def main():
    spec=load_config('experiments/deadline_comm/configs/sender_coverage.yaml')
    src=Path(spec['source'])
    out=Path(spec['output']);out.mkdir(exist_ok=False)
    atomic_json(out/'registration.json',dict(source=str(src),question='Does RR avoid missing senders and stale reward-adjacent caches?',budget='Existing36 cells only, no rollouts',limitations='Episode termination censors gaps; reward-adjacent freshness is association, not importance'))
    sender_rows=[];episode_rows=[];reward_rows=[]
    records=json.loads((src/'summary.json').read_text())['records']
    for cell in sorted(src.iterdir()):
        if not cell.is_dir():continue
        meta=json.loads((cell/'summary.json').read_text())
        keys={k:meta[k] for k in ['seed','condition','mode','method']}
        steps=list(csv.DictReader((cell/'steps.csv').open()))
        messages=[json.loads(l) for l in (cell/'messages.jsonl').read_text().splitlines()]
        for episode in sorted({int(r['episode']) for r in steps}):
            ss=sorted([r for r in steps if int(r['episode'])==episode],key=lambda r:int(r['t']))
            length=len(ss);n=len(json.loads(ss[0]['source_steps']));ever=set();first=[]
            for i in range(n):
                times=sorted({r['t'] for r in messages if r['episode']==episode and r['sender']==i and r['accepted']})
                gaps=[b-a for a,b in zip(times,times[1:])]
                for r in ss:
                    latest=max((t for t in times if t<=int(r['t'])),default=-1)
                    assert latest==json.loads(r['source_steps'])[i]
                row=dict(**keys,episode=episode,sender=i,length=length,first_receive=times[0] if times else None,never_received=not times,n_received=len(times),mean_complete_gap=statistics.mean(gaps) if gaps else None,max_complete_gap=max(gaps) if gaps else None,tail_unreceived_steps=length-1-times[-1] if times else length)
                sender_rows.append(row)
                if times:ever.add(i);first.append(times[0])
            event_rows=[]
            for r in ss:
                if float(r['reward'])<=0:continue
                t=int(r['t']);source=json.loads(r['source_steps'])
                rr=dict(**keys,episode=episode,t=t,reward=float(r['reward']),fresh_sender_fraction=sum(s==t for s in source)/n,invalid_sender_fraction=sum(s<0 for s in source)/n,older_than_one_fraction=sum(s>=0 and t-s>1 for s in source)/n)
                reward_rows.append(rr);event_rows.append(rr)
            episode_rows.append(dict(**keys,episode=episode,length=length,episode_return=sum(float(r['reward']) for r in ss),never_received_fraction=1-len(ever)/n,first_receive_mean_received_only=statistics.mean(first) if first else None,reward_events=len(event_rows)))
    for name,rows in [('sender_episode.csv',sender_rows),('episode_coverage.csv',episode_rows),('reward_adjacency.csv',reward_rows)]:
        log=CSVLog(out/name);log.append(rows);log.close()
    summaries=[]
    for meta in records:
        keys={k:meta[k] for k in ['seed','condition','mode','method']}
        selected=lambda rows:[r for r in rows if all(r[k]==v for k,v in keys.items())]
        ep=selected(episode_rows);sr=selected(sender_rows);rr=selected(reward_rows)
        avg=lambda rows,k:statistics.mean(r[k] for r in rows if r[k] is not None) if any(r[k] is not None for r in rows) else None
        summaries.append(dict(**keys,episodes=len(ep),return_mean=avg(ep,'episode_return'),mean_length=avg(ep,'length'),never_received_fraction=avg(ep,'never_received_fraction'),first_receive_conditional=avg(sr,'first_receive'),complete_gap_conditional=avg(sr,'mean_complete_gap'),tail_unreceived_steps=avg(sr,'tail_unreceived_steps'),reward_events=len(rr),fresh_at_reward=avg(rr,'fresh_sender_fraction'),invalid_at_reward=avg(rr,'invalid_sender_fraction')))
    atomic_json(out/'summary.json',dict(by_seed=summaries,episodes=len(episode_rows),sender_episodes=len(sender_rows),source_timestamp_checks='all passed',limitations='Never-received and gaps depend on episode length; reward events are selected outcomes; no causal conclusion from these correlations'))
    atomic_json(out/'status.json',dict(stage='completed',cells=len(summaries)))
    print(out)

if __name__=='__main__':main()
