"""Aggregate independent seed means, preserving budget failures and failed seeds."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from .io import atomic_json

METRICS = ('mean_return','offered_B_step','goodput_payload_B_step','wasted_pct','cache_invalid_pct','drop_pct','deadline_misses')


def measure(episodes, steps):
    n = len(steps)
    offered = sum(float(s['attempted_bytes']) for s in steps)
    delivered = sum(float(s['delivered_bytes']) for s in steps)
    attempts = sum(int(s['K_attempt']) for s in steps)
    drops = sum(int(s['drop_count']) for s in steps)
    rate = offered/n
    return dict(mean_return=statistics.mean(float(e['episode_return']) for e in episodes),
                offered_B_step=rate,goodput_payload_B_step=32*sum(int(s['K_delivered']) for s in steps)/n,
                wasted_pct=100*(offered-delivered)/offered if offered else None,
                cache_invalid_pct=100*statistics.mean(float(s['invalid_cache_fraction']) for s in steps),
                drop_pct=100*drops/attempts if attempts else None,
                deadline_misses=sum(str(s['action_miss']).lower()=='true' for s in steps),
                budget_eligible=114<=rate<=126)


def read_main(run):
    records, convergence = [], {}
    for seed_dir in sorted(run.glob('seed[0-9]*')):
        seed = int(seed_dir.name[4:])
        evaluations = list(csv.DictReader((seed_dir/'train/evaluations.csv').open()))
        last = max(int(r['update']) for r in evaluations)
        convergence[seed] = statistics.mean(float(r['return']) for r in evaluations if int(r['update'])==last)
        for summary in seed_dir.glob('*/summary.json'):
            if not (summary.parent/'steps.csv').exists(): continue
            steps = list(csv.DictReader((summary.parent/'steps.csv').open()))
            episodes = list(csv.DictReader((summary.parent/'episodes.csv').open()))
            for row in json.loads(summary.read_text()):
                method = row['method']
                records.append(dict(seed=seed,condition=row['condition'],method=method,category=row['category'],
                                    **measure([e for e in episodes if e['method']==method],
                                              [s for s in steps if s['method']==method])))
    return records, convergence


def aggregate(records, convergence):
    groups = defaultdict(list)
    for row in records:
        groups[('all_seeds',row['category'],row['condition'],row['method'])].append(row)
        if convergence[row['seed']]>=5:
            groups[('converged_seeds',row['category'],row['condition'],row['method'])].append(row)
    result = []
    for (cohort,category,condition,method), rows in sorted(groups.items()):
        values = {}
        for metric in METRICS:
            sample = [r[metric] for r in rows if r[metric] is not None]
            values[metric] = dict(mean=statistics.mean(sample) if sample else None,
                                 std=statistics.stdev(sample) if len(sample)>1 else None)
        result.append(dict(cohort=cohort,category=category,condition=condition,method=method,
                           seeds=[r['seed'] for r in rows],all_budget_eligible=all(r['budget_eligible'] for r in rows),metrics=values))
    return result


def report(main, out, final=False):
    records, convergence = read_main(main)
    if final:
        for file in sorted(out.glob('seed*/evaluation.json')):
            records.extend(json.loads(file.read_text()))
    tables = aggregate(records,convergence)
    atomic_json(out/('final_tables.json' if final else 'main_tables.json'),
                dict(convergence=convergence,records=records,tables=tables))
    lines = ['# Day 2 results','',
             'Independent training seeds determine the sample standard deviation. Converged means final-checkpoint Full-Comm evaluation Return >= 5.',
             'Payload goodput counts 32 bytes per delivered broadcast once. Offered load includes the 16-byte header.',
             'Zero attempts leave waste/drop undefined. Slow-link infeasible budgets remain ineligible.',
             'Central Top-K is a centralized reference. Decentralized slots assume shared clock, IDs and configured timing reserve.',
             'Fine-tuning uses 500 additional updates, fresh optimizer/target initialized from final weights, and DAC-conditioned Oracle demonstrations.',
             'Observed differences do not isolate extra training from communication adaptation.','',
             str(convergence),'']
    sections = [
        ('Budget comparison',lambda r:r['category']=='main' and r['condition'] in ('medium_5ms','slow_5ms')),
        ('Deadline sweep',lambda r:r['category']=='main'),
        ('Ablation',lambda r:r['category']=='ablation' or (r['method']=='dac' and r['condition'] in ('slow_5ms','medium_1.5ms','medium_2ms'))),
        ('Extensions',lambda r:r['category']=='extension' or r['method']=='dac')]
    for title, predicate in sections:
        lines += ['## '+title,'','| Cohort | Condition | Method | Seeds | Return | Offered B | Payload B | Waste % | Invalid % | Drop % | Misses | Budget eligible |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for row in tables:
            if not predicate(row):continue
            cells = [row['cohort'],row['condition'],row['method'],str(row['seeds'])]
            for metric in METRICS:
                value=row['metrics'][metric]
                cells.append('NA' if value['mean'] is None else f"{value['mean']:.3f} ± "+('NA' if value['std'] is None else f"{value['std']:.3f}"))
            lines.append('| '+' | '.join(cells+[str(row['all_budget_eligible'])])+' |')
        lines.append('')
    (out/('DAY2_FINAL_REPORT.md' if final else 'DAY2_MAIN_REPORT.md')).write_text('\n'.join(lines))
    with (out/'aggregated_3seed_summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['cohort','category','condition','method','seeds','all_budget_eligible']+
                              [f'{m}_{s}' for m in METRICS for s in ('mean','std')])
        writer.writeheader()
        for row in tables:
            flat={k:v for k,v in row.items() if k!='metrics'}
            flat.update({f'{m}_{s}':row['metrics'][m][s] for m in METRICS for s in ('mean','std')})
            writer.writerow(flat)
