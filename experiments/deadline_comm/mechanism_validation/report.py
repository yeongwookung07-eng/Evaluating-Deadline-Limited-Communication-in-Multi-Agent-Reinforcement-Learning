"""Summarize mechanism runs without promoting incomplete jobs to results."""
import json
from pathlib import Path
import statistics
import time
import argparse

BASE=Path('experiments/deadline_comm/runs')

def generate():
    lines=['# Message mechanism follow-up','', 'All prior artifacts retained. Branches use fixed readiness and visited-state cache interventions; they are not deployment timing measurements. Coverage correlations are not causal explanations. Hidden tasks use fresh weights without demonstrations.','']
    coverage=json.loads((BASE/'sender_coverage_20261119_001/summary.json').read_text())['by_seed']
    lines += ['## Existing coverage, fixed readiness, all 3 seeds','', '|Condition|Method|Return|Never received %|Conditional first receive step|Complete gap|Invalid at positive reward %|','|---|---|---:|---:|---:|---:|---:|']
    for cond in sorted({r['condition'] for r in coverage}):
        for method in ['rr_cap','dac']:
            rows=[r for r in coverage if r['condition']==cond and r['mode']=='fixed_ready' and r['method']==method]
            avg=lambda k:statistics.mean(r[k] for r in rows if r[k] is not None)
            lines.append(f"|{cond}|{method}|{avg('return_mean'):.3f}|{100*avg('never_received_fraction'):.2f}|{avg('first_receive_conditional'):.3f}|{avg('complete_gap_conditional'):.3f}|{100*avg('invalid_at_reward'):.2f}|")
    lines += ['', 'First-reception means exclude never-received senders; complete gaps exclude censored boundaries. Episode lengths differ. Positive-reward states are outcome-selected; do not infer that missing messages caused failure.','']
    branch=BASE/'mechanism_branches_20261119_001'
    if (branch/'summary.json').exists():
        summary=json.loads((branch/'summary.json').read_text());rows=summary['by_seed']
        lines += ['## Single-message branches','',f"Branches: {summary['branches']}; no-op full suffix checks: {summary['noop_checks']}.",'','|Condition|Method|Intervention|Action flip % (seed mean)|Return changed %|Mean suffix Return delta|','|---|---|---|---:|---:|---:|']
        for cond,method,kind in sorted({(r['condition'],r['method'],r['intervention']) for r in rows}):
            selected=[r for r in rows if (r['condition'],r['method'],r['intervention'])==(cond,method,kind)]
            avg=lambda k:statistics.mean(r[k] for r in selected)
            lines.append(f"|{cond}|{method}|{kind}|{100*avg('action_flip_rate'):.3f}|{100*avg('return_changed_rate'):.3f}|{avg('mean_return_delta'):.5f}|")
        # Causal cached-content harm versus heuristic score: descriptive within-state ranking only.
        groups={}
        for line in (branch/'branches.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['intervention']=='zero':groups.setdefault((r['seed'],r['condition'],r['method'],r['episode'],r['t']),[]).append(r)
        ranking=[]
        for key,rr in groups.items():
            if len(rr)<2:continue
            harm=[-r['return_delta'] for r in rr]
            if max(harm)<=0 or max(harm)==min(harm):continue
            selected=max(rr,key=lambda r:(r['priority'],-r['sender']))
            ranking.append(dict(seed=key[0],condition=key[1],method=key[2],matches_max_harm=-selected['return_delta']==max(harm),qgap_matches_max_harm=-max(rr,key=lambda r:(r['q_gap'],-r['sender']))['return_delta']==max(harm),uniform_hit_probability=harm.count(max(harm))/len(harm),n_cached_senders=len(rr)))
        (branch/'priority_vs_cache_harm.json').write_text(json.dumps(dict(states=ranking,limitation='Q-gap+AoI rank among valid cached senders, not unreceived candidate packets; post-hoc diagnostic, not scheduler performance'),indent=2)+'\n')
        for method in ['dac','rr_cap']:
            subset=[r for r in ranking if r['method']==method]
            lines += ['',f"{method}: positive-harm, non-tied states {len(subset)}; Q-gap top matches {sum(r['qgap_matches_max_harm'] for r in subset)}; Q-gap+AoI top matches {sum(r['matches_max_harm'] for r in subset)}; uniform expected matches {sum(r['uniform_hit_probability'] for r in subset):.2f}. Selected subset of valid cached messages; not a new-packet scheduler comparison."]
        lines += ['',f'Non-tied states with positive zero-payload harm: {len(ranking)}. Raw per-seed branch results and priority-vs-cache-harm diagnostics are retained. A null effect only bounds these interventions on these visited caches.','']
    else:lines += ['## Branches','', 'Running; no completed branch result claimed.','']
    hidden=BASE/'hidden_mechanism_20261119_001'
    if (hidden/'summary.json').exists():
        rows=json.loads((hidden/'summary.json').read_text())['records']
        lines += ['## Hidden-source final accuracy','', '|Task|Condition|Method|Seeds|Accuracy mean ± SD|','|---|---|---|---:|---|']
        for task,cond,method in sorted({(r['task'],r['condition'],r['method']) for r in rows}):
            vals=[r['accuracy'] for r in rows if (r['task'],r['condition'],r['method'])==(task,cond,method)]
            lines.append(f'|{task}|{cond}|{method}|{len(vals)}|{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}|')
        lines += ['', 'Check ideal Full accuracy before interpreting DAC-versus-RR: failed learning is not evidence of communication irrelevance. Seed-paired CIs are in the hidden summary.','']
    elif (hidden/'status.json').exists():lines += ['## Hidden-source status','', '```json', (hidden/'status.json').read_text().strip(),'```','']
    dest=Path('experiments/deadline_comm/mechanism_validation/RESULTS.md')
    temp=dest.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();start=time.monotonic()
    while True:
        generate()
        if not a.watch:break
        stages=[]
        for name in ['mechanism_branches_20261119_001','hidden_mechanism_20261119_001']:
            status=BASE/name/'status.json';stages.append(json.loads(status.read_text()).get('stage') if status.exists() else 'pending')
        if all(s in ('completed','failed') for s in stages) or time.monotonic()-start>25*3600:break
        time.sleep(30)
