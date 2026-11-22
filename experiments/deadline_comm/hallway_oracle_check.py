"""Bounded reward-path verification; no training or communication harness changes."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import numpy as np
from .adapters.hallway import Hallway, SOURCE
from .common import load_config, output_dir, save_json, save_csv, hash_file


def oracle_actions(observation, group_ids):
    """Privileged group coordination: gather at distance 1; finish groups separately."""
    positions = observation[:, 0].astype(int)
    active = observation[:, 1].astype(bool)
    groups = np.asarray(group_ids)
    actions = np.zeros(len(positions), dtype=np.int64)
    actions[active & (positions > 1)] = 1
    remaining = sorted(set(groups[active].tolist()))
    if remaining:
        members = (groups == remaining[0]) & active
        if np.all(positions[members] == 1):
            actions[members] = 1
    return actions


def check_episode(env, group_ids, method, seed, positions=None):
    obs, _ = env.reset(seed=seed)
    if positions is not None:
        env.env.state_n[:] = positions
        obs = np.asarray(env.env.get_obs(), np.float32)
    initial = obs[:, 0].astype(int).tolist()
    # Same native class/state, bypassing wrapper.step. Not an independent reimplementation.
    native = deepcopy(env.env)
    total, trace = 0., []
    for t in range(env.episode_length):
        if method == 'straight': actions = np.ones(env.n_agents, np.int64)
        elif method == 'wait': actions = np.zeros(env.n_agents, np.int64)
        elif method == 'synchronized_staggered': actions = oracle_actions(obs, group_ids)
        else: raise ValueError(method)
        before = obs[:, 0].astype(int).tolist()
        obs, reward, term, trunc, info = env.step(actions)
        raw_reward, raw_done, raw_info = native.step(actions.tolist())
        if reward != raw_reward or bool(term or trunc) != bool(raw_done):
            raise AssertionError('Reward/done wrapper mismatch')
        np.testing.assert_array_equal(obs, np.asarray(native.get_obs(), np.float32))
        np.testing.assert_array_equal(env.get_state(), np.asarray(native.get_state(), np.float32))
        np.testing.assert_array_equal(env.available_actions(), np.asarray(native.get_avail_actions(), bool))
        if info['success'] != bool(raw_info['battle_won']):
            raise AssertionError('Success wrapper mismatch')
        total += reward
        trace.append(dict(t=t,positions_before=before,actions=actions.tolist(),
            positions_after=obs[:,0].astype(int).tolist(),reward=reward,
            terminated=term,truncated=trunc,success=info['success'],
            won_groups=env.env._win_group,failed_groups=env.env._fail_group))
        if term or trunc: break
    result = dict(method=method,seed=seed,initial_positions=json.dumps(initial),
        episode_return=total,length=len(trace),success=info['success'],
        terminated=term,truncated=trunc,positive_reward_steps=sum(x['reward']>0 for x in trace),
        negative_reward_steps=sum(x['reward']<0 for x in trace),upstream_parity=True)
    return result, trace


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/deadline_comm/configs/hallway_oracle_check.yaml')
    ap.add_argument('--run-name')
    args=ap.parse_args(); cfg=load_config(args.config)
    out=output_dir(args.run_name or cfg['output_name'])
    meta=dict(status='running',config=cfg,source_hashes={str(p.resolve()):hash_file(p) for p in
        [Path(__file__),SOURCE,Path(__file__).parent/'adapters/hallway.py',Path(args.config)]},
        scope='Privileged oracle reward-path validation; no policy training or network timing')
    save_json(out/'manifest.json',meta)
    env=Hallway(**cfg['environment']); groups=cfg['environment']['group_ids']
    rows=[]; traces=[]
    cases=[('seeded',s,None) for s in cfg['seeds']]
    if cfg['exhaustive_initial_positions']:
        cases.extend(('exhaustive',0,pos) for pos in itertools.product(*[range(1,n+1) for n in cfg['environment']['state_numbers']]))
    if len(cases)*len(cfg['policies'])>cfg['budget']['max_episodes']:
        raise ValueError('Episode budget exceeded')
    try:
        for case_id,(kind,seed,positions) in enumerate(cases):
            for method in cfg['policies']:
                row,trace=check_episode(env,groups,method,seed,positions)
                row.update(case_id=case_id,kind=kind); rows.append(row)
                traces.append(dict(case_id=case_id,kind=kind,method=method,seed=seed,
                    initial_positions=json.loads(row['initial_positions']),steps=trace))
        checks=dict(oracle_all_positive_success=all(r['episode_return']==20 and r['success'] and r['terminated'] and not r['truncated'] for r in rows if r['method']=='synchronized_staggered'),
            wait_all_zero=all(r['episode_return']==0 and not r['success'] for r in rows if r['method']=='wait'),
            all_upstream_parity=all(r['upstream_parity'] for r in rows),
            within_step_budget=sum(r['length'] for r in rows)<=cfg['budget']['max_wrapper_steps'])
        aggregates=[]
        for kind in ['seeded','exhaustive']:
            for method in cfg['policies']:
                selected=[r for r in rows if r['kind']==kind and r['method']==method]
                if not selected: continue
                aggregates.append(dict(kind=kind,method=method,episodes=len(selected),
                    mean_return=float(np.mean([r['episode_return'] for r in selected])),
                    positive_return_count=sum(r['episode_return']>0 for r in selected),
                    negative_return_count=sum(r['episode_return']<0 for r in selected),
                    success_count=sum(r['success'] for r in selected),
                    max_length=max(r['length'] for r in selected),
                    return_distribution={str(v):sum(r['episode_return']==v for r in selected) for v in sorted({r['episode_return'] for r in selected})}))
        save_csv(out/'episodes.csv',rows)
        with (out/'traces.jsonl').open('w') as f:
            for row in traces: f.write(json.dumps(row)+'\n')
        save_json(out/'summary.json',dict(checks=checks,groups=aggregates,
            limitations=['Centralized oracle uses joint observations/group IDs; not a learned or decentralized policy.',
                        'Direct source parity uses the same loaded upstream class, not an independent native reproduction.',
                        'Does not identify why learned policies fail or establish VBC/TMC/GC causal hypotheses.']))
        if not all(checks.values()): raise AssertionError(checks)
        meta.update(status='completed',episodes=len(rows),wrapper_steps=sum(r['length'] for r in rows),checks=checks)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc)); raise
    finally: save_json(out/'manifest.json',meta)
    print(out)

if __name__=='__main__': main()
