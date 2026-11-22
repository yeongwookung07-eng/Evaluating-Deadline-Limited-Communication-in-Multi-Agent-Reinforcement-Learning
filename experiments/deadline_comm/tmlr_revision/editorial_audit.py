"""Recompute manuscript denominators, treatment checks and costs from frozen logs."""
import json,itertools,statistics,collections,csv
from pathlib import Path
import numpy as np
from scipy.stats import t
from ..adapters.hallway import Hallway
BASE=Path('experiments/deadline_comm');HERE=BASE/'tmlr_revision'
def lines(p):
    with p.open() as f:
        for line in f:yield json.loads(line)
def interval(a):
    a=np.asarray(a,float);m=float(a.mean());d=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return [m,m-d,m+d]
def main():
    fresh=collections.defaultdict(list)
    for r in lines(BASE/'runs/tmlr_fresh_20261119_001/states.jsonl'):
        assert sorted(tuple(v['senders']) for v in r['variants'])==list(itertools.combinations(range(5),r['k']))
        original=next(v for v in r['variants'] if v['senders']==r['normal_senders'])
        assert original['return_']==r['baseline_return'] and not original['action_flip']
        fresh[r['condition'],r['checkpoint_arm']].append(r)
    fs=[]
    for (cond,arm),rows in fresh.items():
        byseed=[[r for r in rows if r['seed']==seed] for seed in range(10)]
        metrics={}
        for name,fn in [('range',lambda r:r['range_return']),('regret',lambda r:max(v['return_'] for v in r['variants'])-r['baseline_return']),('effect',lambda r:100*(r['range_return']>0)),('random_regret',lambda r:max(v['return_'] for v in r['variants'])-statistics.mean(v['return_'] for v in r['variants']))]:
            metrics[name]=interval([statistics.mean(fn(r) for r in rr) for rr in byseed])
        fs.append(dict(condition=cond,arm=arm,states=len(rows),branches=sum(len(r['variants']) for r in rows),seed_states=[len(rr) for rr in byseed],seed_branches=[sum(len(r['variants']) for r in rr) for rr in byseed],metrics=metrics,postcoverage_states=sum(r['initial_invalid']==0 for r in rows),postcoverage_range_mean=statistics.mean([r['range_return'] for r in rows if r['initial_invalid']==0]) if any(r['initial_invalid']==0 for r in rows) else None))
    groups=collections.defaultdict(list);env=Hallway(group_ids=[0,0,1,1,1])
    for r in lines(BASE/'runs/tmlr_prefix_20261119_001/episodes.jsonl'):
        env.reset(seed=14100000+r['episode'])
        for x in r['prefix']:
            _,reward,_,_,_=env.step(x['actions']);assert reward==x['reward'] and env.get_state().tolist()==x['state']
        a,b=r['paired']['rr_cap']['trace'],r['paired']['dac']['trace']
        groups[r['condition'],r['checkpoint_arm']].append(dict(seed=r['seed'],survived=r['survived'],remaining=14-len(r['prefix']),active_groups=sum(env.env.active_group),common_steps=min(len(a),len(b)),set_diff_steps=sum(x['senders']!=y['senders'] for x,y in zip(a,b)),action_diff_steps=sum(x['actions']!=y['actions'] for x,y in zip(a,b)),different_lengths=len(a)!=len(b),rr_steps=len(a),dac_steps=len(b),effect=r['paired']['dac']['suffix_return']-r['paired']['rr_cap']['suffix_return']))
    prefix=[]
    for (cond,arm),rows in groups.items():
        survivors=[r for r in rows if r['survived']];seedmeans=[statistics.mean(r['effect'] for r in survivors if r['seed']==seed) for seed in range(10)]
        prefix.append(dict(condition=cond,arm=arm,episodes=len(rows),survivors=len(survivors),set_diff_episodes=sum(r['set_diff_steps']>0 for r in rows),action_diff_episodes=sum(r['action_diff_steps']>0 for r in rows),set_diff_steps=sum(r['set_diff_steps'] for r in rows),action_diff_steps=sum(r['action_diff_steps'] for r in rows),common_steps=sum(r['common_steps'] for r in rows),different_lengths=sum(r['different_lengths'] for r in rows),mean_remaining=statistics.mean(r['remaining'] for r in survivors),mean_rr_steps=statistics.mean(r['rr_steps'] for r in survivors),mean_dac_steps=statistics.mean(r['dac_steps'] for r in survivors),unresolved_group_histogram=dict(collections.Counter(r['active_groups'] for r in survivors)),survivor_effect=interval(seedmeans)))
    timing=[]
    for arm in ['rr_cap','dac']:
        counter=collections.Counter();total=0
        for r in lines(BASE/'runs/tmlr_readiness_20261119_001/episodes.jsonl'):
            if (r['checkpoint_arm'],r['deadline_ms'],r['mode'])!=(arm,2.25,'measured'):continue
            credit=0
            for x in r['trace']:
                credit=min(144,credit+120);budget=int(credit//48);attempt=x['wire']//48;credit-=x['wire'];total+=1
                if budget<3:counter['credit_below_three']+=1
                else:
                    counter['credit_at_least_three']+=1;counter['ready_le310_given_credit']+=x['ready_ns']<=310000
                    if attempt<3:counter['admission_below_three_with_credit']+=1
                    elif x['k']<3:counter['attempt_three_but_deliver_less']+=1
                    else:counter['deliver_three']+=1
        timing.append(dict(arm=arm,total_steps=total,**counter))
    b=collections.defaultdict(list)
    for r in lines(BASE/'runs/tmlr_broadcast_20261119_001/steps.jsonl'):
        # Deterministic attempted counts under the recorded fixed-readiness gate contract.
        # This is reconstructed accounting, not a newly measured CPU trace.
        if r['scheduler']=='source_aware':attempt=r['k']
        elif r['method']=='periodic':attempt=sum((r['t']+i)%2==0 for i in range(5))
        elif r['method']=='ideal_full':attempt=5
        elif r['method']=='no_comm':attempt=0
        else:attempt=r['k'] # cap includes propagation; threshold zero admits all five.
        b[r['condition'],r['method'],r['scheduler'],r['seed']].append(dict(k=r['k'],attempt=attempt,accuracy=r['followers_accuracy'],q=r['source_delivered']))
    broadcast=[]
    for cond in ['slow_5ms','medium_2ms','medium_5ms']:
        for method in ['periodic','rr_cap','dac']:
            means={mode:[{k:statistics.mean(x[k] for x in b[cond,method,mode,seed]) for k in ['k','attempt','accuracy','q']} for seed in range(10)] for mode in ['normal','source_aware']}
            broadcast.append(dict(condition=cond,method=method,normal={k:statistics.mean(x[k] for x in means['normal']) for k in means['normal'][0]},source_aware={k:statistics.mean(x[k] for x in means['source_aware']) for k in means['source_aware'][0]},gain=interval([100*(y['accuracy']-x['accuracy']) for x,y in zip(means['normal'],means['source_aware'])])))
    hidden_root=BASE/'runs/hidden_mechanism_20261119_001'
    hidden_records=json.loads((hidden_root/'records.json').read_text());checked=0
    for r in hidden_records:
        path=hidden_root/f"{r['task']}_n5_seed{r['seed']}_{r['method']}"/f"eval_{r['condition']}"/'episodes.csv'
        with path.open() as f:episodes=list(csv.DictReader(f))
        assert len(episodes)==50 and [int(x['seed']) for x in episodes]==list(range(9500000,9500050))
        accuracy=statistics.mean(float(x['episode_return'])/int(x['length']) for x in episodes)
        assert abs(accuracy-r['accuracy'])<1e-12
        checked+=1
    result=dict(hidden_original_csv_verified_cells=checked,fresh=fs,prefix=prefix,timing=timing,broadcast=broadcast,scope='Seed-level t intervals are descriptive and unadjusted. Prefix unresolved groups reconstructed by replaying logged prefix actions, not inferred from active-agent flags. Broadcast attempted counts reconstructed from deterministic gate semantics; CPU overhead of diagnostic source selection was not measured.')
    (HERE/'results/editorial_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(fresh_states=sum(r['states'] for r in fs),prefix_episodes=sum(r['episodes'] for r in prefix),timing=timing)))
if __name__=='__main__':main()
