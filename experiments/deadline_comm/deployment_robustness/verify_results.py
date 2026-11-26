"""Independent accounting audit of all saved robustness outcomes (no training)."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t


def ci(x):
    x=np.asarray(x,float);m=x.mean();d=student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(m),ci95=[float(m-d),float(m+d)],seed_values=x.tolist())


def audit_mnist(root):
    r=json.loads((root/'results.json').read_text());reg=json.loads((root/'registration.json').read_text())
    hashes=json.loads((root/'input_hashes.json').read_text())
    for p,h in hashes['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    count=0
    for seed in reg['seeds']:
        with np.load(root/f'seed{seed}_cells.npz',allow_pickle=False) as a:
            for idx,(d,sd,b,c) in enumerate(a['cells']):
                if sd:
                    probability=np.array([.5*math.erfc(-(d-220-c-ready-b)/(sd*math.sqrt(2))) for ready in a['ready_us']])
                else:probability=(a['ready_us']+b+c+220<=d).astype(float)
                np.testing.assert_allclose(probability,a['offer_probability'][idx],atol=1e-14)
                gains=a['full_reward']-a['zero_reward'];timely=a['ready_us']+220+c<=d;cheap=a['ready_us']+220<=d
                informed=a['zero_reward']+probability*timely*gains
                other=a['zero_reward']+probability.mean()*cheap*gains
                same=a['zero_reward']+probability.mean()*timely*gains
                for name,val in [('informed',informed),('random_no_extra_cost',other),('random_same_cost',same)]:np.testing.assert_allclose(val,a[name][idx],atol=1e-14)
                row=next(x for x in r['rows'] if (x['seed'],x['deadline_us'],x['noise_sd_us'],x['bias_us'],x['decision_cost_us'])==(seed,d,sd,b,c))
                assert abs(row['gain_practical_pp']-100*(informed-other).mean())<1e-11
                assert abs(row['offered_B']-17*probability.mean())<1e-12
                count+=len(probability)
    for cell in r['summary']:
        group=[x for x in r['rows'] if all(x[k]==cell[k] for k in ('deadline_us','noise_sd_us','bias_us','decision_cost_us'))]
        for key,m in cell['metrics'].items():
            calc=ci([x[key] for x in group]);np.testing.assert_allclose(calc['ci95'],m['t95'],atol=1e-12)
    return dict(passed=True,example_cells=count,probabilities_by_independent_erfc=True,all_summary_intervals=True,source_hashes=True)


def audit_broadcast(root):
    summaries=json.loads((root/'summary.json').read_text());all_rows=[];steps=0;episodes=0
    for rep in range(summaries['process_repeats']):
        p=root/f'repeat{rep}';rows=json.loads((p/'results.json').read_text())['rows'];all_rows+=rows
        for f,h in json.loads((p/'input_hashes.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        buckets={}
        with gzip.open(p/'episodes.jsonl.gz','rt') as stream:
            for line in stream:
                ep=json.loads(line);trace=ep['trace'];previous=[0]*5;timestamps=[-1]*5;rng=np.random.default_rng(ep['reset']);rewards=[]
                key=(ep['seed'],ep['scheduler'],ep['arm'],ep['fallback'],ep['post_cost_us']);bucket=buckets.setdefault(key,[])
                assert len(trace)==20
                for t,row in enumerate(trace):
                    bit=int(rng.integers(0,2,size=1)[0]);assert row['bit']==bit and row['t']==t
                    assert row['previous_executed']==previous
                    miss=row['cpu_total_ns']+row['wait_ns']+row['post_cost_ns']>2000000
                    assert row['miss']==miss and row['total_ns']==row['cpu_total_ns']+row['wait_ns']+row['post_cost_ns']
                    assert row['post_cost_ns']==ep['post_cost_us']*1000
                    action=row['candidate']
                    if miss and ep['fallback']=='hold':action=previous
                    if miss and ep['fallback']=='zero':action=[0]*5
                    assert row['actions']==action
                    assert row['late_candidate_applied']==(miss and ep['fallback']=='ordinary')
                    reward=sum(x==bit for x in action[1:])/4
                    assert row['reward']==reward
                    previous=action;rewards.append(reward)
                    messages=row['messages'];delivered=sorted(m['sender'] for m in messages if m['accepted'])
                    assert row['delivered_senders']==delivered
                    assert row['offered_wire_B']==48*len(messages)
                    assert row['delivered_payload_B']==32*len(delivered)
                    for queue_index,m in enumerate(messages):
                        assert m['arrival_ns']==row['ready_ns']+(queue_index+1)*480000+50000
                        assert m['accepted']==(m['arrival_ns']<=1550000)
                        if m['accepted']:timestamps[m['sender']]=t
                    assert row['source_timestamps']==timestamps
                    bucket.append(row);steps+=1
                assert ep['return_']==sum(rewards);episodes+=1
        for row in rows:
            tr=buckets[row['seed'],row['scheduler'],row['arm'],row['fallback'],row['post_cost_us']]
            assert abs(np.mean([x['reward'] for x in tr])-row['accuracy'])<1e-12
            assert abs(np.mean([x['miss'] for x in tr])-row['miss_fraction'])<1e-12
            assert abs(np.mean([x['offered_wire_B'] for x in tr])-row['offered_B'])<1e-12
    for cell in summaries['cells']:
        rows=[r for r in all_rows if all(r[k]==cell[k] for k in ('scheduler','fallback','post_cost_us'))]
        for comparator in ('original','shuffled'):
            differences=[]
            for seed in range(10):
                means={a:np.mean([r['accuracy'] for r in rows if r['seed']==seed and r['arm']==a]) for a in ('adapted',comparator)}
                differences.append(100*(means['adapted']-means[comparator]))
            val=ci(differences);saved=cell['contrasts']['adapted_minus_'+comparator+'_pp'];np.testing.assert_allclose(val['ci95'],saved['ci95'],atol=1e-12)
    return dict(passed=True,episodes=episodes,steps=steps,fallbacks_rewards_timing_delivery_verified=True,seed_level_aggregation=True,source_hashes=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mnist',required=True);p.add_argument('--broadcast',required=True);a=p.parse_args()
    result=dict(mnist=audit_mnist(Path(a.mnist)),broadcast=audit_broadcast(Path(a.broadcast)))
    Path(a.broadcast,'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
