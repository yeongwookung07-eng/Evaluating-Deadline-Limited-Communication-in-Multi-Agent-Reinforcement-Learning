"""Recheck the two transfer studies from their saved raw observations and rewards."""
import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import groupby
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev
import struct


RUNS = next((p / 'experiments/deadline_comm/runs' for p in Path(__file__).resolve().parents
             if (p / 'experiments/deadline_comm/runs').is_dir()), Path('runs'))
TERRAN = RUNS / 'terran_sender_branch_replay_20261128_001'
PREDATOR = RUNS / 'native_predator_sender_20261128_001'
TRAINING_SEEDS = (0, 1, 2)
RESET_BASE = 28112800
T95_DF2 = 4.302652729696142


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def read_lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def close(actual, expected):
    assert isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def seed_stats(seed_values):
    assert sorted(seed_values) == list(TRAINING_SEEDS)
    values = [seed_values[seed] for seed in TRAINING_SEEDS]
    point = mean(values)
    half_width = T95_DF2 * stdev(values) / sqrt(3)
    return {'per_seed': values, 'mean': point, 'ci95': [point-half_width, point+half_width]}


def terran_diagnosis(path):
    """Compare each separately saved transition/field fingerprint before reading the summary."""
    logged = {(r['adapter'], r['repeat']): r for r in read_json(path / 'diagnosis.json')}
    metrics = {}
    legacy = {'pre_obs', 'pre_state', 'pre_mask', 'action', 'payload', 'ts',
              'hidden', 'sender', 'reward', 'terminal', 'timeout'}
    for adapter in ('SMACAdapter', 'ScopedSMAC'):
        traces = [read_json(path / f'diagnostic_{adapter}_{r}.json.gz') for r in range(3)]
        assert all(len(trace) == 38 for trace in traces)
        first_differences = []
        for r, trace in enumerate(traces):
            mismatches = [(t, {k for k in row if row[k] != reference[k]})
                          for t, (row, reference) in enumerate(zip(trace, traces[0]))]
            entry = {}
            for label, fields in [('exact', set(trace[0])), ('legacy_fields', legacy)]:
                first = next(((t, sorted(keys & fields)) for t, keys in mismatches if keys & fields), None)
                reported = logged[adapter, r][label]
                assert first == (None if reported is None else (reported['t'], sorted(reported['fields'])))
                entry[label] = first
            first_differences.append(entry)
        metrics[adapter] = first_differences
    assert all(r['exact'] is None for r in metrics['ScopedSMAC'])
    assert any(r['exact'] is not None for r in metrics['SMACAdapter'])
    return metrics


def native_rewards(trace):
    """Check rewards using predator sight flags and grid occupancy from the saved next state."""
    assert 0 < len(trace) <= 50
    values = []
    for t, row in enumerate(trace):
        if t:
            prior = trace[t - 1]
            assert prior['next_state'] == row['state'] and prior['next_obs'] == row['obs']
        invisible = sum(obs[2] == 0 for obs in row['next_obs'][:4])
        predator_reward = -0.1 * invisible if invisible else 1.0
        positions = [tuple(round(7 * x) for x in row['next_state'][i:i+2]) for i in range(0, 10, 2)]
        assert len(set(positions)) == 5
        x, y = positions[-1]
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        prey_reward = -int(all(min(point) < 0 or max(point) > 6 or point in positions[:4]
                               for point in neighbors))
        rewards = row['rewards']
        assert len(rewards) == len(row['actions']) == 5
        for reward in rewards[:4]:
            close(predator_reward, reward)
        close(prey_reward, rewards[4])
        assert row['done'] == [not invisible] * 5
        if not invisible:
            assert t == len(trace)-1
        values.append(sum(rewards))
    assert any(trace[-1]['done']) or len(trace) == 50
    return values


def indexed(path, fields):
    result = {}
    for row in read_lines(path):
        key = tuple(row[k] for k in fields)
        assert key not in result
        result[key] = row
    return result


def native_study(path):
    summary = read_json(path / 'summary.json')
    manifest = read_json(path / 'manifest.json')
    assert manifest['status'] == 'complete' and not manifest['smoke']
    episode_rows = indexed(path / 'episodes.jsonl', ['seed', 'reset', 'condition'])
    branch_rows = indexed(path / 'branches.jsonl', ['seed', 'reset', 'step', 'sender'])
    metrics = defaultdict(lambda: defaultdict(list))
    predator_only_gaps = defaultdict(list)
    predator_only_sensitive = defaultdict(int)
    observed = set()
    by_step, count = defaultdict(int), defaultdict(int)
    for (seed, reset), group in groupby(read_lines(path / 'traces.jsonl.gz'), lambda x: (x['seed'], x['reset'])):
        assert (seed, reset) not in observed
        observed.add((seed, reset))
        episodes, decisions = {}, defaultdict(dict)
        for record in group:
            trace = record['trace']
            rewards = native_rewards(trace)
            count['transitions'] += len(trace)
            condition = record['condition']
            if condition == 'branch':
                t, sender = record['step'], record['sender']
                assert sender not in decisions[t]
                decisions[t][sender] = (trace, sum(rewards[t:]))
                continue
            assert condition not in episodes
            episodes[condition] = (trace, sum(rewards))
            report = episode_rows.pop((seed, reset, condition))
            close(report['total_return'], sum(rewards))
            assert report['steps'] == len(trace) and report['terminal'] == any(trace[-1]['done'])
            metrics[condition+'_return'][seed].append(sum(rewards))
            metrics[condition+'_capture'][seed].append(int(report['terminal']))
        assert set(episodes) == {'factual', 'sham', 'random', 'zero'}
        factual, factual_return = episodes['factual']
        assert factual == episodes['sham'][0]
        count['sham_exact'] += 1
        for control in ('zero', 'random'):
            other, other_return = episodes[control]
            for k in ('state', 'obs', 'native_sender'):
                assert other[0][k] == factual[0][k]
            metrics['factual_minus_'+control][seed].append(factual_return-other_return)
        assert set(decisions) == {t for t in (0, 20) if t < len(factual)}
        ep_effects = defaultdict(list)
        four_predator_effects = []
        for t, alternatives in decisions.items():
            assert sorted(alternatives) == list(range(4))
            by_step[t] += 1
            returns, immediate_changes, action_changes = [], [], []
            predator_only = {}
            fact_sender = factual[t]['sender']
            assert alternatives[fact_sender][0] == factual
            count['factual_forced_exact'] += 1
            for sender, (trace, value) in alternatives.items():
                report = branch_rows.pop((seed, reset, t, sender))
                assert trace[:t] == factual[:t] and all(trace[t][k] == factual[t][k]
                                                      for k in ('state', 'obs', 'native_sender'))
                assert trace[t]['sender'] == sender and report['factual_sender'] == fact_sender
                assert all(i == t or x['sender'] == x['native_sender'] for i, x in enumerate(trace))
                assert report['prefix_match'] and report['scheduled_count'] == 1
                assert report['factual_forced_exact'] == (True if sender == fact_sender else None)
                assert report['terminal'] == any(trace[-1]['done']) and report['steps'] == len(trace)
                count['prefix_exact'] += 1
                close(value, report['suffix'])
                close(alternatives[fact_sender][1], report['factual_suffix'])
                immediate = trace[t]['actions'][:4] != factual[t]['actions'][:4]
                suffix_actions = [q['actions'][:4] for q in trace[t:]] != [q['actions'][:4] for q in factual[t:]]
                assert immediate == report['target_predator_actions_changed']
                assert suffix_actions == report['suffix_actions_changed']
                immediate_changes.append(immediate)
                action_changes.append(suffix_actions)
                returns.append(value)
                predator_only[sender] = sum(sum(q['rewards'][:4]) for q in trace[t:])
            four_predator_effects.append(predator_only[fact_sender]-mean(predator_only.values()))
            predator_only_sensitive[seed] += max(predator_only.values())-min(predator_only.values()) > 1e-8
            effects = {'advantage': alternatives[fact_sender][1]-mean(returns),
                       'sensitivity': max(returns)-min(returns) > 1e-8,
                       'action_sensitivity': any(action_changes),
                       'immediate_action_sensitivity': any(immediate_changes)}
            for name, value in effects.items():
                metrics['state_'+name][seed].append(value)
                ep_effects[name].append(value)
        for name, values in ep_effects.items():
            metrics['episode_'+name][seed].append(mean(values))
        predator_only_gaps[seed].append(mean(four_predator_effects))
    assert observed == {(s, r) for s in TRAINING_SEEDS for r in range(RESET_BASE, RESET_BASE+24)}
    assert not episode_rows and not branch_rows
    assert summary['checks'] == {k: count[k] for k in ('sham_exact', 'prefix_exact', 'factual_forced_exact')}
    assert summary['rollouts'] == 4*len(observed) + count['prefix_exact']
    assert summary['branches'] == count['prefix_exact'] and summary['episodes'] == len(observed)
    assert summary['decisions'] == sum(by_step.values()) and summary['checkpoints'] == 3
    assert {int(k): v for k, v in summary['decisions_by_step'].items()} == by_step
    aggregates = {key: seed_stats({s: mean(v) for s, v in vals.items()}) for key, vals in metrics.items()}
    assert set(aggregates) == set(summary['aggregates'])
    for key, estimate in aggregates.items():
        reported = summary['aggregates'][key]
        for seed, value in enumerate(estimate['per_seed']):
            close(value, reported['seed_values'][seed])
            close(value, summary['per_seed'][str(seed)][key])
        close(estimate['mean'], reported['mean'])
        close(estimate['ci95'][0], reported['low'])
        close(estimate['ci95'][1], reported['high'])
    gates = {k: min(aggregates[k]['per_seed']) > 0 and aggregates[k]['ci95'][0] > 0
             for k in ('factual_minus_zero', 'factual_minus_random', 'episode_advantage')}
    assert gates == summary['positive_gates']
    return {'verified': dict(count), 'decisions_by_step': dict(by_step), 'metrics': aggregates,
            'post_hoc_four_predator_reward': {
                'sensitive_decisions_per_seed': [predator_only_sensitive[s] for s in TRAINING_SEEDS],
                'sensitive_decisions': sum(predator_only_sensitive.values()),
                'decisions': sum(by_step.values()),
                'episode_weighted_advantage': seed_stats({s: mean(v) for s, v in predator_only_gaps.items()})},
            'counts_by_seed': {k: [int(sum(v[s])) for s in TRAINING_SEEDS] for k, v in metrics.items()
                               if k in ('state_sensitivity', 'state_action_sensitivity',
                                        'state_immediate_action_sensitivity', 'factual_capture',
                                        'random_capture', 'zero_capture')}, 'positive_gates': gates}


def terran_chain(rows, finished):
    assert rows
    for i, row in enumerate(rows):
        assert row['t'] == i
        assert len(row['sender']) == 1 and len(row['action']) == 5
        if row['terminal'] or row['timeout']:
            assert i+1 == len(rows)
        if i:
            assert all(rows[i-1]['post_'+k] == row['pre_'+k] for k in ('obs', 'state', 'mask', 'native'))
    assert bool(rows[-1]['terminal'] or rows[-1]['timeout']) == finished


def terran_cell(path, row):
    """Use digests for simulator continuity and policy no-ops, raw numerical rewards for effects."""
    cell = path / f"{row['mode']}_s{row['seed']}_r{row['reset']}"
    prefix = read_json(cell / 'prefix.json.gz')
    t = len(prefix)
    assert row['branch_step'] == t <= 8
    close(row['prefix_reward'], sum(x['reward'] for x in prefix))
    if not row['reached']:
        terran_chain(prefix, finished=True)
        assert row['valid'] and not row['branches'] and not row['action_changed']
        close(row['factual_minus_uniform'], 0.0)
        return dict(valid=True, reached=0, action_changed=0, nonfactual_action_changes=0,
                    nonfactual_return_changes=0, return_changed=0, win_changed=0, advantage=0.)
    assert t == 8
    common = read_json(cell / 'common_actions.json')
    assert all(common[k] == row[k] for k in common)
    natural = read_json(cell / 'natural.json.gz')
    terran_chain(prefix+natural, finished=True)
    close(row['factual_suffix'], sum(x['reward'] for x in natural))
    assert row['factual_suffix_steps'] == len(natural)
    assert natural[0]['pre_obs'] == common['input_observation']
    assert natural[0]['pre_state'] == common['input_state']
    for key, previous_key in [('hidden', 'hidden'), ('cache', 'payload'), ('timestamps', 'ts'),
                              ('previous', 'previous')]:
        assert common['prefix_'+key] == prefix[-1][previous_key]
    assert natural[0]['pre_mask'] == hashlib.sha256(bytes(v for r in common['available'] for v in r)).hexdigest()
    factual = row['factual_sender']
    assert natural[0]['sender'] == [factual] and natural[0]['action'] == row['natural_action']
    candidates = {r['sender']: r for r in common['candidates']}
    assert sorted(candidates) == list(range(5)) and len(common['candidates']) == 5
    assert candidates[factual]['action'] == natural[0]['action']
    assert len({(c['cursor_after'], c['budget_after']) for c in candidates.values()}) == 1
    previous_timestamps = [max(x['t'] for x in prefix if x['sender'] == [s]) for s in range(5)]
    for sender, candidate in candidates.items():
        assert candidate['offered_bytes'] == 48 and candidate['payload_bytes'] == 32
        timestamps = previous_timestamps.copy()
        timestamps[sender] = t
        assert [timestamps] == candidate['source']
        assert all(common['available'][i][a] for i, a in enumerate(candidate['action']))
    branches = {b['sender']: b for b in row['branches']}
    checks = {b['sender']: b for b in row['replay_checks']}
    assert branches.keys() == checks.keys()
    values, wins, prefix_matches, mismatch = {}, {}, [], None
    natural_actions = [x['action'] for x in natural]
    changed_action_traces = 0
    identical_physical_steps = 0
    for sender, branch in branches.items():
        alt = read_json(cell / f's{sender}.json.gz')
        if row['mode'] == 'fresh':
            branch_prefix = read_json(cell / f's{sender}_prefix.json.gz')
            terran_chain(branch_prefix+alt, finished=True)
            prefix_ok = prefix == branch_prefix and all(
                alt[0][k] == natural[0][k]
                for k in ('pre_obs', 'pre_state', 'pre_mask', 'pre_native', 'wrapper_visibility'))
            assert prefix_ok == (checks[sender]['prefix_or_live_state'] is None)
        else:
            prefix_ok = checks[sender]['prefix_or_live_state'] is None
        assert alt[0]['sender'] == [sender]
        assert alt[0]['ts'] == hashlib.sha256(struct.pack('<5q', *candidates[sender]['source'][0])).hexdigest()
        # After the substituted decision the deployed scheduler resumes its factual cursor.
        assert all(r['sender'] == [r['t'] % 5] for r in alt[1:])
        action_ok = alt[0]['action'] == candidates[sender]['action']
        assert checks[sender]['same_as_physical_action'] == action_ok
        if row['mode'] == 'fresh' and prefix_ok:
            for f, a in zip(natural, alt):
                if f['action'] != a['action']:
                    break
                assert all(f[k] == a[k] for k in ('pre_native', 'pre_state', 'pre_mask', 'pre_obs',
                           'wrapper_visibility', 'post_native', 'post_state', 'post_mask', 'post_obs',
                           'reward', 'terminal', 'timeout', 'won'))
                identical_physical_steps += 1
        noop_ok = sender != factual or alt == natural
        if sender == factual and not noop_ok:
            first = next((i for i, (a, b) in enumerate(zip(natural, alt)) if a != b), None)
            if first is None:
                mismatch = {'lengths': [len(natural), len(alt)]}
            else:
                mismatch = {'t': t+first, 'fields': sorted(k for k in natural[first] if natural[first][k] != alt[first][k])}
                report = checks[sender]['factual_replay']
                assert mismatch['t'] == report['t'] and mismatch['fields'] == sorted(report['fields'])
        assert (checks[sender]['factual_replay'] is None) == noop_ok
        assert branch['offered_bytes'] == 48 and branch['delivered_payload_bytes'] == 32
        assert alt[0]['action'] == branch['action'] and len(alt) == branch['steps']
        assert alt[-1]['won'] == branch['won']
        changed_action_traces += [x['action'] for x in alt] != natural_actions
        values[sender] = sum(r['reward'] for r in alt)
        wins[sender] = alt[-1]['won']
        close(values[sender], branch['return_'])
        prefix_matches.append(prefix_ok and action_ok and noop_ok)
    valid = len(values) == 5 and all(prefix_matches)
    assert valid == row['valid']
    action_count = sum(c['action'] != natural[0]['action'] for c in candidates.values())
    assert bool(action_count) == row['action_changed']
    result = dict(valid=valid, reached=1, action_changed=int(bool(action_count)), nonfactual_action_changes=action_count,
                  forced_factual_steps=len(natural), prefix_replays=len(prefix_matches),
                  prefix_matches=sum(prefix_matches), physical_steps_until_action_divergence=identical_physical_steps,
                  first_noop_mismatch=mismatch)
    if valid:
        effect = values[factual]-mean(values.values())
        close(effect, row['factual_minus_uniform'])
        nonzero = [abs(r-values[factual]) for r in values.values() if r != values[factual]]
        count = len(nonzero)
        above_tolerance = sum(delta > 1e-8 for delta in nonzero)
        return_sensitive = max(values.values())-min(values.values()) > 1e-8
        result.update(nonfactual_return_changes=count, return_changed=int(bool(count)),
                      nonfactual_return_changes_above_tolerance=above_tolerance,
                      return_range_above_tolerance=int(return_sensitive),
                      identical_immediate_actions_different_return=int(not action_count and return_sensitive),
                      suffix_action_changed=int(bool(changed_action_traces)), nonfactual_suffix_action_changes=changed_action_traces,
                      win_changed=int(len(set(wins.values())) > 1), factual_win=int(wins[factual]),
                      smallest_nonzero_return_change=min(nonzero, default=None), advantage=effect)
    else:
        assert row['factual_minus_uniform'] is None
    return result


def terran_study(path):
    diag = terran_diagnosis(path)
    attempts = {}
    summary = read_json(path / 'summary.json')
    selected = summary['mode']
    for mode in ['quick', 'fresh']:
        raw_path = path / f'{mode}_episodes.json'
        if not raw_path.is_file():
            continue
        raw = read_json(raw_path)
        rows = {(r['seed'], r['reset']): terran_cell(path, r) for r in raw}
        assert len(rows) == len(raw)
        if mode != selected:
            assert mode == 'quick' and any(not r['valid'] for r in rows.values())
            attempts[mode] = list(rows.values())
            continue
        assert set(rows) == {(s, r) for s in TRAINING_SEEDS for r in range(RESET_BASE, RESET_BASE+10)}
        valid = all(r['valid'] for r in rows.values())
        assert valid == summary['valid']
        by_seed, gap = {}, {}
        for s in TRAINING_SEEDS:
            cells = [v for (seed, _), v in rows.items() if seed == s]
            keys = ['reached', 'action_changed', 'nonfactual_action_changes', 'prefix_replays', 'prefix_matches']
            keys.append('physical_steps_until_action_divergence')
            if valid:
                keys += ['nonfactual_return_changes', 'return_changed', 'nonfactual_return_changes_above_tolerance',
                         'return_range_above_tolerance', 'suffix_action_changed', 'nonfactual_suffix_action_changes',
                         'identical_immediate_actions_different_return', 'win_changed', 'factual_win']
            counts = {k: sum(r.get(k, 0) for r in cells) for k in keys}
            counts['episodes'] = len(cells)
            by_seed[s] = counts
            report = next(r for r in summary['per_seed'] if r['seed'] == s)
            for a, b in [('episodes','episodes'), ('reached','reached'), ('action_changed','episodes_action_changed'),
                         ('nonfactual_action_changes','changed_nonfactual_actions')]:
                assert counts[a] == report[b]
            assert 4*counts['reached'] == report['nonfactual_branches']
            if valid:
                assert counts['nonfactual_return_changes'] == report['changed_nonfactual_returns']
                gap[s] = mean(r['advantage'] for r in cells)
                close(gap[s], report['mean_factual_minus_uniform'])
            else:
                assert report['changed_nonfactual_returns'] is None and report['mean_factual_minus_uniform'] is None
        result = {'valid': valid, 'per_seed': by_seed}
        if valid:
            result['advantage'] = seed_stats(gap)
            result['return_tolerance'] = 1e-8
            positive = [r['smallest_nonzero_return_change'] for r in rows.values()
                        if r.get('smallest_nonzero_return_change') is not None]
            result['smallest_nonzero_return_change'] = min(positive, default=None)
            close(result['advantage']['mean'], summary['mean_factual_minus_uniform'])
            for actual, reported in zip(result['advantage']['ci95'], summary['seed_t95']):
                close(actual, reported)
        else:
            assert 'mean_factual_minus_uniform' not in summary and 'seed_t95' not in summary
        attempts[mode] = result
    assert selected in attempts
    return {'diagnosis': diag, 'selected': selected, 'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=RUNS, help='directory containing the two archived run directories')
    parser.add_argument('--arm', choices=('both', 'predator', 'terran'), default='both')
    parser.add_argument('--output', type=Path, help='write the verification report as JSON')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Run with Python assertions enabled')
    report = {}
    if args.arm in ('predator', 'both'):
        report['predator'] = native_study(args.runs / PREDATOR.name)
    if args.arm in ('terran', 'both'):
        report['terran'] = terran_study(args.runs / TERRAN.name)
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
