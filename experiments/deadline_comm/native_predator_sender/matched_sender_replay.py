"""Frozen, same-reset, one-decision sender substitutions in native SchedNet Predator–Prey."""
import argparse
import configparser
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
import warnings
from unittest.mock import patch

for key in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy import stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(file, record):
    file.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')


def interval(values):
    mean = float(np.mean(values))
    margin = (float(stats.t.ppf(.975, len(values)-1) * stats.sem(values))
              if len(values) > 1 else 0.0)
    return dict(seed_values=values, mean=mean, low=mean-margin, high=mean+margin)


def summarize(episodes, branches):
    by_seed = {}
    for ep in episodes:
        by_seed.setdefault(ep['seed'], {}).setdefault(ep['reset'], {})[ep['condition']] = ep
    decisions = {}
    for row in branches:
        decisions.setdefault((row['seed'], row['reset'], row['step']), []).append(row)
    results = {}
    for seed, resets in by_seed.items():
        metrics = {}
        def add(key, value):
            metrics.setdefault(key, []).append(float(value))
        for reset, runs in resets.items():
            base = runs['factual']
            for condition, rec in runs.items():
                add(condition + '_return', rec['total_return'])
                add(condition + '_capture', rec['terminal'])
            for control in ['zero', 'random']:
                add('factual_minus_' + control, base['total_return'] - runs[control]['total_return'])
            obs = []
            for (s, r, _), rows in decisions.items():
                if (s, r) != (seed, reset):
                    continue
                assert sorted(x['sender'] for x in rows) == list(range(4))
                factual = rows[0]['factual_suffix']
                counterfactuals = [x['suffix'] for x in rows]
                effect = factual - np.mean(counterfactuals)
                sensitive = max(counterfactuals) - min(counterfactuals) > 1e-8
                action = any(x['suffix_actions_changed'] for x in rows)
                immediate = any(x['target_predator_actions_changed'] for x in rows)
                obs.append([effect, sensitive, action, immediate])
            for key, vals in zip(['advantage', 'sensitivity', 'action_sensitivity',
                                  'immediate_action_sensitivity'], zip(*obs)):
                add('episode_' + key, np.mean(vals))
                for val in vals:
                    add('state_' + key, val)
        results[seed] = {k: float(np.mean(v)) for k, v in metrics.items()}
    aggregates = {k: interval([m[k] for m in results.values()]) for k in next(iter(results.values()))}
    gates = {key: bool(min(aggregates[key]['seed_values']) > 0 and aggregates[key]['low'] > 0)
             for key in ['factual_minus_zero', 'factual_minus_random', 'episode_advantage']}
    return dict(per_seed=results, aggregates=aggregates, positive_gates=gates,
                checkpoints=len(results), episodes=len(episodes)//4, rollouts=len(episodes)+len(branches),
                decisions=len(decisions), branches=len(branches),
                decisions_by_step={t: sum(k[2] == t for k in decisions)
                                   for t in sorted({k[2] for k in decisions})},
                checks={'sham_exact':sum(r.get('sham_trace_match', False) for r in episodes),
                        'prefix_exact':sum(r['prefix_match'] for r in branches),
                        'factual_forced_exact':sum(r['factual_forced_exact'] is True for r in branches)})


def run_traces(cfg, out, source, weights, smoke, manifest):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    sys.modules['tensorflow'] = tf
    sys.modules['ConfigParser'] = configparser
    sys.path.insert(0, str(source))
    sys.argv = ['native_schednet', '--seed=0', '--training_step=50000', '--eval_on_train=False',
                '--gui=False', '--max_step=' + str(cfg['horizon'])]
    import config
    import make_env
    from agents.schednet.trainer import Trainer
    flags = config.flags.FLAGS
    assert (flags.scenario, flags.sch_type, flags.s_num, flags.n_predator, flags.n_prey) == (
        'predator_prey_obs', 'top', 1, 4, 1)
    manifest.update(native_flags={k: getattr(flags, k) for k in (
        'scenario','sch_type','s_num','n_predator','n_prey','hetero','map_size','obs_range','max_step')},
                    tensorflow=tf.__version__, numpy=np.__version__)
    factory = lambda: make_env.make_env(flags.scenario)
    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    trainer = Trainer(factory())
    agent = trainer._predator_agent
    actor = agent.action_selector
    selected = [op.outputs[0] for op in agent.sess.graph.get_operations()
                if op.name.startswith('actor') and op.name.endswith('/scheduled')]
    assert len(selected) == 4, [t.name for t in selected]
    assert all(t.shape.as_list() == [None, 2] for t in selected)
    manifest['scheduled_tensors'] = [t.name for t in selected]

    def rollout(reset, condition, force_step=None, force_sender=None):
        start, cpu = time.monotonic(), time.process_time()
        np.random.seed(reset)
        random.seed(reset)
        env = factory()
        obs_raw = env.reset()
        h = np.zeros(4)
        obs, state, h = trainer.get_obs_state_with_schedule(obs_raw, env.get_info(), h, init=True)
        trace = []

        def substituted_actor(state_input, schedule):
            if condition == 'random':
                return np.full((1, 20), .2)
            feeds = {actor.state_ph:state_input, actor.schedule_ph:schedule, actor.is_training_ph:False}
            if condition == 'sham':
                payloads = agent.sess.run(selected, feed_dict=feeds)
            else:
                payloads = [np.zeros((1,2), dtype=np.float32) for _ in selected]
            feeds.update(zip(selected, payloads))
            return agent.sess.run(actor.actions, feed_dict=feeds)

        # The native agent draws predator and prey actions in the same order in every arm.
        with patch.object(actor, 'action_for_state', actor.action_for_state if condition == 'factual'
                          else substituted_actor):
            for step in range(cfg['horizon']):
                schedule, _ = trainer.get_schedule(obs, step+1, flags.sched, train=False)
                native_sender = int(np.argmax(schedule))
                if step == force_step:
                    schedule = np.eye(4)[force_sender]
                assert np.count_nonzero(schedule) == 1 and sum(schedule) == 1
                actions = trainer.get_action(obs, schedule, step+1, train=False)
                raw_next, rewards, done, info = env.step(actions)
                next_obs, next_state, h = trainer.get_obs_state_with_schedule(raw_next, info, h, schedule)
                trace.append(dict(state=state.tolist(), obs=[np.asarray(o).tolist() for o in obs],
                                  native_sender=native_sender, sender=int(np.argmax(schedule)),
                                  actions=actions.tolist(), rewards=list(map(float, rewards)),
                                  done=list(map(bool, done)), next_state=next_state.tolist(),
                                  next_obs=[np.asarray(o).tolist() for o in next_obs]))
                obs, state = next_obs, next_state
                if any(done):
                    break
        return trace, {'elapsed_seconds':time.monotonic()-start, 'cpu_seconds':time.process_time()-cpu}

    def suffix(trace, t=0):
        return sum(sum(x['rewards']) for x in trace[t:])

    episodes, branches = [], []
    with (out/'episodes.jsonl').open('x') as eps, (out/'branches.jsonl').open('x') as br, \
            gzip.open(out/'traces.jsonl.gz', 'xt', compresslevel=3) as raw:
        for seed in cfg['seeds'][:1] if smoke else cfg['seeds']:
            ckpt, = (weights/f'seed{seed}_native/results/nn').glob('*-50000.index')
            agent.saver.restore(agent.sess, str(ckpt.with_suffix('')))
            manifest.setdefault('checkpoints', {})[seed] = {
                p.name:digest(p) for p in sorted(ckpt.parent.glob(ckpt.stem + '.*'))}
            resets = [17203155] if smoke else range(cfg['first_reset'], cfg['first_reset'] + cfg['resets'])
            for reset in resets:
                key = {'seed':seed,'reset':reset}
                for condition in ['factual','sham','zero','random']:
                    trace, timing = rollout(reset, condition)
                    if condition == 'factual':
                        factual = trace
                    ep = dict(**key, condition=condition, steps=len(trace), terminal=any(trace[-1]['done']),
                              total_return=suffix(trace), **timing)
                    if condition == 'sham':
                        ep['sham_trace_match'] = trace == factual
                        assert ep['sham_trace_match'], key
                    episodes.append(ep)
                    write_json(eps, ep)
                    write_json(raw, dict(**key, condition=condition, trace=trace))
                for t in cfg['branch_steps']:
                    if t >= len(factual):
                        continue
                    for sender in range(4):
                        trace, timing = rollout(reset, 'factual', t, sender)
                        prefix = trace[:t] == factual[:t] and all(
                            trace[t][k] == factual[t][k] for k in ['obs','state','native_sender'])
                        assert prefix and trace[t]['sender'] == sender, (key, t, sender)
                        same = sender == factual[t]['sender']
                        if same:
                            assert trace == factual, (key, t)
                        row = dict(**key, step=t, sender=sender, factual_sender=factual[t]['sender'],
                                   prefix_match=prefix, factual_forced_exact=trace == factual if same else None,
                                   scheduled_count=1, suffix=suffix(trace,t), factual_suffix=suffix(factual,t),
                                   target_predator_actions_changed=trace[t]['actions'][:4] != factual[t]['actions'][:4],
                                   suffix_actions_changed=[x['actions'][:4] for x in trace[t:]] !=
                                                          [x['actions'][:4] for x in factual[t:]],
                                   terminal=any(trace[-1]['done']), steps=len(trace), **timing)
                        branches.append(row)
                        write_json(br, row)
                        write_json(raw, dict(**key, condition='branch', step=t, sender=sender, trace=trace))
                eps.flush()
                br.flush()
            print(f'Finished native checkpoint {seed}', flush=True)
    agent.sess.close()
    return summarize(episodes, branches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--smoke', action='store_true', help='Separate one-reset correctness run')
    ap.add_argument('--output', type=Path, help='Fresh directory override')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[3]
    source = root/cfg['source']
    weights = root/cfg['checkpoint_root']
    out = (args.output or root/(cfg['output'] + ('_smoke' if args.smoke else ''))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'results/eval').mkdir(parents=True)
    (out/'results/nn').mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    manifest = dict(config=cfg, smoke=args.smoke, script_sha256=digest(Path(__file__)),
                    config_sha256=digest(args.config), status='running', device='cpu', threads=1)
    source_manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text())
    for entry in source_manifest['files']:
        assert digest(source/entry['path']) == entry['sha256'], entry
    manifest['source_commit'] = source_manifest['commit']
    manifest['source_manifest_sha256'] = digest(source/'SOURCE_MANIFEST.json')
    os.chdir(out)
    def expired(signum, frame):
        raise TimeoutError('Registered wall clock limit exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg['runtime_seconds'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        result = run_traces(cfg, out, source, weights, args.smoke, manifest)
        (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        manifest['status'] = 'complete'
    except BaseException as e:
        manifest.update(status='failed', error=repr(e))
        raise
    finally:
        manifest.update(seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
