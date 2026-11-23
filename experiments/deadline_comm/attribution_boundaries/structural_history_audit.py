"""Existing-log structural identity, joint-history and deadline boundary audit."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import time
import yaml
from scipy.stats import t as student_t


def rows(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def stats(values):
    values = list(values)
    m = st.mean(values)
    d = student_t.ppf(.975, len(values)-1)*st.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0
    return dict(mean=m, t95=[float(m-d), float(m+d)], n=len(values), values=values)


def summarize(items):
    byseed=defaultdict(list)
    for r in items: byseed[r['seed']].append(r)
    keys=[k for k in items[0] if k not in ['seed','episode']]
    return {k:stats(st.mean(r[k] for r in byseed[s]) for s in sorted(byseed)) for k in keys}


def pair(x,y,seed,episode):
    a,b=x['trace'],y['trace'];common=list(zip(a,b))
    eq=lambda key:len(a)==len(b) and all(u[key]==v[key] for u,v in common)
    ax=sum(v['attempts'] for v in a);ay=sum(v['attempts'] for v in b)
    sets_equal=eq('senders')
    counts_equal=len(a)==len(b) and all(len(u['senders'])==len(v['senders']) for u,v in common)
    return dict(seed=seed,episode=episode,return_difference=y['episode_return']-x['episode_return'],attempt_saving=1-ay/ax,offered_bytes_per_step_saving=1-(ay/len(b))/(ax/len(a)),full_steps=len(a),alternative_steps=len(b),common_count_disagreement_steps=sum(len(u['senders'])!=len(v['senders']) for u,v in common),first_sender_mismatch=next((u['t'] for u,v in common if u['senders']!=v['senders']),-1),return_equal=x['episode_return']==y['episode_return'],sender_history_equal=sets_equal,count_history_equal=counts_equal,counts_equal_senders_different=counts_equal and not sets_equal,cache_history_equal=eq('cache') and eq('source'),actions_equal=eq('actions'),rewards_equal=eq('reward'),return_equal_actions_different=x['episode_return']==y['episode_return'] and not eq('actions'),return_equal_senders_different=x['episode_return']==y['episode_return'] and not sets_equal,full_rotation_singleton=all(u['senders']==[u['t']%5] for u in a),alternative_rotation_singleton=all(v['senders']==[v['t']%5] for v in b))


def main(config):
    started=time.monotonic();s=yaml.safe_load(Path(config).read_text());out=Path(s['output']);out.mkdir(exist_ok=False)
    (out/'registration.json').write_text(json.dumps(s,indent=2)+'\n')
    inputs={}
    def read(path):
        path=Path(path);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert time.monotonic()-started<s['budget_seconds'],'budget exceeded'
        return list(rows(path))
    grouped=defaultdict(list)
    raw=[]
    identity=Path(s['identity_run'])
    for f in sorted(identity.glob('seed*_full.jsonl')):
        seed=int(f.name.split('_')[0][4:]);condition='_'.join(f.name.split('_')[1:3]);decay=f.name.split('_')[3][6:]
        a=read(f)
        for method in ['rr_cap','dac']:
            b=read(f.with_name(f.name.replace('_full.jsonl','_'+method+'.jsonl')))
            assert len(a)==len(b)==20
            for x,y in zip(a,b):
                assert x['reset']==y['reset']
                r=pair(x,y,seed,x['reset']);grouped[condition,decay,method].append(r)
                raw.append(dict(condition=condition,decay=decay,method=method,**r))
    identity_summary=[]
    for (condition,decay,method),rr in sorted(grouped.items()):
        identity_summary.append(dict(condition=condition,decay=decay,method=method,episodes=len(rr),metrics=summarize(rr),counts={k:sum(r[k] for r in rr) for k in rr[0] if isinstance(rr[0][k],bool)}))
    prefix=defaultdict(list)
    for r in read(Path(s['prefix_run'])/'episodes.jsonl'):
        a=r['paired']['rr_cap'];b=r['paired']['dac'];orig=r['originals'];common=list(zip(a['trace'],b['trace']))
        prefix[r['condition'],r['checkpoint_arm']].append(dict(seed=r['seed'],episode=r['episode'],prefix_length=len(r['prefix']),survived=r['survived'],cold_difference=orig['dac']['return_']-orig['rr_cap']['return_'],prefix_difference=b['suffix_return']-a['suffix_return'],changed_return=b['suffix_return']!=a['suffix_return'],different_sender_on_common_steps=any(x['senders']!=y['senders'] for x,y in common),different_actions_on_common_steps=any(x['actions']!=y['actions'] for x,y in common),different_suffix_length=len(a['trace'])!=len(b['trace'])))
    prefix_summary=[dict(condition=k[0],weights=k[1],episodes=len(v),metrics=summarize(v),counts={q:sum(r[q] for r in v) for q in ['survived','changed_return','different_sender_on_common_steps','different_actions_on_common_steps','different_suffix_length']}) for k,v in sorted(prefix.items())]
    readiness=[];snapshot=[]
    roots=[('original',Path(s['readiness_run']))]+[(f'repeat{i}',Path(s['repeat_run'])/f'repeat{i}'/'rollouts') for i in range(3)]
    for proc,root in roots:
        timing=defaultdict(list);stepvalues=defaultdict(lambda:defaultdict(list))
        for r in read(root/'episodes.jsonl'):
            tr=r['trace'];threshold=r['deadline_ms']*1000-450-50-3*480
            credit=0;opportunity=[]
            for q in tr:
                credit=min(144,credit+120);opportunity.append(credit>=144 and q['ready_ns']/1000<=threshold);credit-=q['wire']
                assert credit>=0
            rr=dict(seed=r['seed'],episode=r['episode'],return_=r['return_'],k=st.mean(q['k'] for q in tr),ready_us=st.mean(q['ready_ns']/1000 for q in tr),slack_three_us=st.mean(threshold-q['ready_ns']/1000 for q in tr),ready_for_three_rate=st.mean(q['ready_ns']/1000<=threshold for q in tr),credit_and_ready_three_rate=st.mean(opportunity),delivered_three_rate=st.mean(q['k']>=3 for q in tr),action_miss_rate=st.mean(q['miss'] for q in tr))
            key=(r['checkpoint_arm'],r['deadline_ms'],r['mode']);timing[key].append(rr)
            stepvalues[key][r['seed']].extend(q['k'] for q in tr)
        for (arm,deadline,mode),rr in sorted(timing.items()):
            readiness.append(dict(process=proc,weights=arm,deadline_ms=deadline,mode=mode,episodes=len(rr),three_packet_ready_threshold_us=deadline*1000-1940,episode_primary=summarize(rr),step_secondary_k=stats(st.mean(v) for _,v in sorted(stepvalues[arm,deadline,mode].items()))))
    for i in range(3):
        matched=defaultdict(dict)
        for r in read(Path(s['repeat_run'])/f'repeat{i}'/'snapshots.jsonl'):
            matched[r['seed'],r['weights'],r['episode'],r['t']][r['selector']]=r
        rr=[]
        for (seed,weights,ep,t),r in matched.items():
            a,b=r['rr_cap'],r['dac']
            rr.append(dict(seed=seed,episode=(weights,ep),gate_difference_us=(b['gate_ns']-a['gate_ns'])/1000,ready_difference_us=(b['ready_ns']-a['ready_ns'])/1000,k_difference=b['K_delivered']-a['K_delivered']))
        # Average snapshots within each episode/weights pair first, then within seed.
        epgroup=defaultdict(list)
        for r in rr: epgroup[r['seed'],r['episode']].append(r)
        episode_rows=[dict(seed=k[0],episode=str(k[1]),**{q:st.mean(r[q] for r in v) for q in ['gate_difference_us','ready_difference_us','k_difference']}) for k,v in epgroup.items()]
        snapshot.append(dict(process=f'repeat{i}',snapshots=len(rr),episode_primary=summarize(episode_rows),state_secondary=summarize(rr)))
    result=dict(scope='Retained-log reanalysis only; no new rollouts, training or timing measurements.',identity=identity_summary,prefix=prefix_summary,readiness=readiness,snapshots=snapshot,structural_conditions={'general':'Same initial full state, policy weights, cache transformation and random draws; equal delivered payload/sender/timestamp inputs each step imply equal subsequent states/actions/rewards inductively. Ordered input equality is sufficient; for order-invariant aggregation it can be weakened to sender-keyed equality.','specialization':'Full queue begins at t mod N; RR starts at cursor 0 and advances by admitted count. With exactly one successful packet/step, no skipped admission, matching payload/cache/readiness, both deliver t mod N. For constant 1<=k<N, Full begins at t while RR begins at kt; equal sets require (k-1)t=0 mod N. k=1 guarantees equality; equal count alone does not.','offered_bytes':'5 Full attempts versus 1 RR attempt yields 80% offered-byte reduction. Actual serialized bytes, RF energy and physical channel occupancy were not recorded.','prefix':'Shared RR prefix equalizes full joint history at branching. Cold versus prefix comparisons cannot isolate a unique cache mediator.','timing':'3 packets at 100000 B/s and 48 B each require readiness <= deadline -450us reserve -50us propagation -1440us serialization; opportunity also requires credit and actual admission.'},elapsed_seconds=time.monotonic()-started)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'input_sha256.json').write_text(json.dumps(inputs,indent=2)+'\n');(out/'paired_identity.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
    tex=['% Generated from existing frozen-checkpoint identity traces; no new rollouts.',r'\begin{tabular}{llrrrr}',r'\hline',r'Condition & Access & Pairs & Same counts & Same senders & Same return \\',r'\hline']
    for condition in sorted(set(r['condition'] for r in identity_summary)):
        for method in ['rr_cap','dac']:
            rr=[r for r in identity_summary if r['condition']==condition and r['method']==method]
            cc=lambda k:sum(r['counts'][k] for r in rr)
            tex.append(f"{condition.replace('_','/')} & {method.replace('_','-')} & {sum(r['episodes'] for r in rr)} & {cc('count_history_equal')} & {cc('sender_history_equal')} & {cc('return_equal')} "+r'\\')
    tex.extend([r'\hline',r'\end{tabular}']);(out/'same_cohort_table.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(output=str(out),identity_cells=len(identity_summary),identity_pairs=len(raw),prefix_pairs=sum(len(v) for v in prefix.values()),timing_cells=len(readiness),elapsed_seconds=result['elapsed_seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
