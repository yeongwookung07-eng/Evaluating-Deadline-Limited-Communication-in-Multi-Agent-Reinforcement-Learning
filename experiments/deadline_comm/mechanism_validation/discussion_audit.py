"""Existing-log checks of discussion explanations; no new scientific observations."""
from pathlib import Path
import csv,json,statistics
from collections import defaultdict
from ..common import load_config
from ..day2.io import atomic_json,CSVLog


def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    spec=load_config('experiments/deadline_comm/configs/discussion_evidence_audit.yaml')
    out=Path(spec['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',spec)
    src=Path(spec['coverage_source']);cells={}
    for p in src.glob('*/summary.json'):
        m=json.loads(p.read_text())
        if m['mode']!='fixed_ready':continue
        steps=readcsv(p.parent/'steps.csv');episodes=defaultdict(list)
        for r in steps:episodes[int(r['episode'])].append(r)
        cells[(m['seed'],m['condition'],m['method'])]={ep:sorted(rr,key=lambda r:int(r['t'])) for ep,rr in episodes.items()}
    coverage=[]
    for seed in spec['seeds']:
        for cond in ['slow_5ms','medium_2ms','medium_5ms']:
            rr=cells[(seed,cond,'rr_cap')];dac=cells[(seed,cond,'dac')]
            assert set(rr)==set(dac)
            for window in spec['windows']:
                admitted=[ep for ep in rr if len(rr[ep])>=window and len(dac[ep])>=window]
                for method,episodes in [('rr_cap',rr),('dac',dac)]:
                    invalid=[];offered=[];age=[]
                    for ep in admitted:
                        rows=episodes[ep][:window];assert [int(r['t']) for r in rows]==list(range(window))
                        n=len(json.loads(rows[-1]['source_steps']))
                        invalid.append(sum(x<0 for x in json.loads(rows[-1]['source_steps']))/n)
                        offered.append(statistics.mean(float(r['wire_bytes']) for r in rows))
                        valid_ages=[window-1-x for x in json.loads(rows[-1]['source_steps']) if x>=0]
                        age.append(statistics.mean(valid_ages) if valid_ages else None)
                    avg=lambda x:statistics.mean(x) if x else None
                    coverage.append(dict(seed=seed,condition=cond,window_steps=window,method=method,eligible_pairs=len(admitted),excluded_pairs=len(rr)-len(admitted),invalid_at_window_end=avg(invalid),wire_B_step=avg(offered),valid_age_at_end=avg([x for x in age if x is not None])))
    branches=[json.loads(l) for l in (Path(spec['branch_source'])/'branches.jsonl').read_text().splitlines()]
    effects=[]
    groups=defaultdict(list)
    for r in branches:
        bucket='fresh_0' if r['age']==0 else 'age_1' if r['age']==1 else 'age_2plus'
        groups[(r['seed'],r['condition'],r['method'],r['intervention'],bucket)].append(r)
    for key,rr in sorted(groups.items()):
        effects.append(dict(zip(['seed','condition','method','intervention','age_group'],key),n=len(rr),benefit_count=sum(r['return_delta']>0 for r in rr),harm_count=sum(r['return_delta']<0 for r in rr),no_change_count=sum(r['return_delta']==0 for r in rr),benefit_fraction=statistics.mean(r['return_delta']>0 for r in rr),harm_fraction=statistics.mean(r['return_delta']<0 for r in rr),mean_return_delta=statistics.mean(r['return_delta'] for r in rr)))
    # Same factual state for zero versus norm-preserving reversal; no unpaired trajectory pooling.
    interventions={}
    for r in branches:
        key=tuple(r[k] for k in ['seed','condition','method','episode','t','sender'])
        interventions.setdefault(key,{})[r['intervention']]=r
    assert all(set(d)=={'zero','negate'} for d in interventions.values())
    intervention_pairs=[]
    for key,d in interventions.items():
        assert d['zero']['baseline_return']==d['negate']['baseline_return']
        intervention_pairs.append(dict(seed=key[0],condition=key[1],method=key[2],episode=key[3],t=key[4],sender=key[5],age=d['zero']['age'],zero_delta=d['zero']['return_delta'],negate_delta=d['negate']['return_delta']))
    raw=readcsv(Path(spec['training_source'])/'by_seed.csv');load=[]
    for cond in ['slow_5ms','medium_2ms','medium_5ms']:
        for seed in range(10):
            get=lambda method:next(r for r in raw if r['family']=='scratch_'+cond and r['condition']==cond and r['arm']==method and int(r['seed'])==seed)
            a,b=get('dac'),get('rr_cap');da,rb=float(a['offered_B_step']),float(b['offered_B_step'])
            load.append(dict(seed=seed,condition=cond,dac_B_step=da,rr_B_step=rb,difference_B_step=da-rb,exactly_equal=abs(da-rb)<1e-9,relative_difference=(da-rb)/rb,return_difference=float(a['mean_return'])-float(b['mean_return'])))
    for name,rows in [('matched_window_coverage.csv',coverage),('age_stratified_interventions.csv',effects),('paired_interventions.csv',intervention_pairs),('same_cap_actual_load.csv',load)]:
        log=CSVLog(out/name);log.append(rows);log.close()
    atomic_json(out/'summary.json',dict(coverage=coverage,effects=effects,load=load,paired_content_interventions=len(intervention_pairs),limitations=['Surviving paired trajectories at window>1 are selected; matched horizon removes duration differences only within the retained cohort.','Age strata are factual-state associations, not randomized age interventions.','Three checkpoint seeds; branch events are correlated.','Same cap does not mean exactly identical actual bytes.']))
    atomic_json(out/'status.json',dict(stage='completed'))
    print(out)

if __name__=='__main__':main()
