"""Read-only historical audit and bounded fresh exploration diagnostics."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .timing import configure_cpu
from .adapters.hallway import Hallway, SOURCE
from .common import ROOT, load_config, output_dir, save_json, save_csv, hash_file


class DiagnosticHallway(Hallway):
    def reset(self, seed=None):
        self.events=[]
        return super().reset(seed)

    def step(self, actions):
        before=self.env.state_n.copy()
        active=np.asarray(self.env.active_group,bool).copy()
        failed_before=self.env._fail_group
        obs,reward,term,trunc,info=super().step(actions)
        after=self.env.state_n.copy()
        gids=self.env.group_ids
        # Classify native events from previously active groups and post-action distances.
        completed=[int(g) for g in range(self.env.n_groups) if active[g] and (after[gids==g]==0).all()]
        failed=[int(g) for g in range(self.env.n_groups) if active[g] and (after[gids==g]==0).any() and not (after[gids==g]==0).all()]
        if len(failed)!=self.env._fail_group-failed_before: raise AssertionError('Failure event mismatch')
        self.events.append(dict(t=len(self.events),positions_before=before.tolist(),actions=np.asarray(actions).tolist(),
            positions_after=after.tolist(),reward=reward,terminated=term,truncated=trunc,
            async_failed_groups=failed,simultaneous_group_penalty=len(completed)>1,
            completed_groups_this_step=completed,won_groups=self.env._win_group,failed_groups=self.env._fail_group,
            active_groups=list(map(bool,self.env.active_group)),success=info['success']))
        return obs,reward,term,trunc,info


def terminal_cause(env):
    if env.env._win_group==env.env.n_groups: return 'success'
    if env.env._fail_group==env.env.n_groups: return 'all_groups_failed'
    if env.env._episode_steps>=env.episode_length: return 'time_limit'
    raise AssertionError('Unexpected episode ending')


def historical_audit(names):
    import torch
    rows=[]
    for name in names:
        p=ROOT/'runs'/name
        meta=json.loads((p/'manifest.json').read_text())
        training=list(csv.DictReader((p/'training.csv').open()))
        saved=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
        all_zero=all(float(r['episode_return'])==0 for r in training)
        source_matches=meta['source_hashes'].get(str(SOURCE))==hash_file(SOURCE)
        env=meta['config']['environment']
        two_groups=sorted(set(env['group_ids']))==[0,1]
        # Native 2-group structure: negative reward requires both groups to finish on
        # the same step, then neither is active. A prior single-group success precludes
        # that branch. Thus +10 and -10 cannot cancel within a default episode.
        inference=all_zero and source_matches and two_groups
        rows.append(dict(run=name,files=sorted(x.name for x in p.iterdir()),checkpoint_keys=list(saved),
            replay_saved=False,checkpoint_has_replay=any('replay' in k.lower() or 'buffer' in k.lower() for k in saved),
            episodes=len(training),transitions=meta['transitions'],return_histogram=dict(Counter(r['episode_return'] for r in training)),
            native_source_hash_matches=source_matches,nonzero_transition_reward_directly_observed=None,
            inferred_no_nonzero_rewards=inference,
            inference_basis='Unchanged default two-group native reward branches cannot cancel +10 with -10 within one episode; all episode returns are zero.' if inference else None))
    return rows


def save_replay(path, records):
    ep_offsets=np.cumsum([0]+[len(r.episode.actions) for r in records])
    obs_offsets=np.cumsum([0]+[len(r.episode.observations) for r in records])
    data={key:np.concatenate([getattr(r.episode,key) for r in records],axis=0) for key in
          ('observations','states','available_actions','actions','rewards','terminated','truncated')}
    data.update({key:np.concatenate([getattr(r,key) for r in records],axis=0) for key in
                 ('attempt_mask','delivery_mask','source_step')})
    data.update(initial_cache=np.stack([r.initial_cache for r in records]),
                initial_source=np.stack([r.initial_source for r in records]),transition_offsets=ep_offsets,observation_offsets=obs_offsets)
    np.savez_compressed(path,**data)


def aggregate(rows, traces):
    rewards=[e['reward'] for tr in traces for e in tr['steps']]
    return dict(episodes=len(rows),transitions=len(rewards),
        reward_histogram=dict(Counter(str(float(v)) for v in rewards)),
        positive_reward_transitions=sum(v>0 for v in rewards),negative_reward_transitions=sum(v<0 for v in rewards),
        positive_reward_episodes=sum(r['has_positive_reward'] for r in rows),
        negative_reward_episodes=sum(r['has_negative_reward'] for r in rows),
        positive_return_episodes=sum(r['episode_return']>0 for r in rows),success_episodes=sum(r['success'] for r in rows),
        mean_return=float(np.mean([r['episode_return'] for r in rows])),mean_length=float(np.mean([r['length'] for r in rows])),
        terminal_causes=dict(Counter(r['terminal_cause'] for r in rows)),
        async_failure_episodes=sum(r['async_failure_groups']>0 for r in rows),
        simultaneous_penalty_episodes=sum(r['simultaneous_penalty'] for r in rows),
        timeout_with_async_failure=sum(r['terminal_cause']=='time_limit' and r['async_failure_groups']>0 for r in rows),
        timeout_with_penalty=sum(r['terminal_cause']=='time_limit' and r['simultaneous_penalty'] for r in rows),
        first_async_failure_step_histogram=dict(Counter(str(r['first_failure_t']) for r in rows if r['first_failure_t'] is not None)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='experiments/deadline_comm/configs/exploration_reward_audit.yaml');ap.add_argument('--run-name')
    a=ap.parse_args();configure_cpu()
    import torch
    from .common import build_policy
    from .gates import METHODS
    from .harness import rollout
    cfg=load_config(a.config);base=load_config(cfg['source_config'])
    out=output_dir(a.run_name or cfg['output_name'])
    meta=dict(status='running',config=cfg,behavior_config=base,torch=torch.__version__,threads=torch.get_num_threads(),
        source_hashes={str(p.resolve()):hash_file(p) for p in [Path(__file__),SOURCE,ROOT/'harness.py',ROOT/'policy.py',ROOT/'adapters/hallway.py',Path(a.config)]},
        updates=0,definition='Fresh frozen INITIAL policies, paired resets, cyclic original gate schedule; not recovery of historical replay.')
    save_json(out/'manifest.json',meta)
    try:
        history=historical_audit(cfg['historical_runs']);save_json(out/'historical_audit.json',history)
        models={}
        for seed in cfg['model_seeds']:
            torch.manual_seed(seed);models[seed]=build_policy(base).eval()
        env=DiagnosticHallway(**base['environment'])
        allrows=[];totals={};steps=0
        for arm,epsilon in cfg['arms'].items():
            rows=[];traces=[];records=[]
            for ep in range(cfg['episodes_per_arm']):
                seed=cfg['environment_seed_start']+ep;ms=cfg['model_seeds'][ep%len(models)];method=METHODS[ep%len(METHODS)]
                record,_,_,summary=rollout(env,models[ms],method,base['harness'],seed,False,epsilon)
                events=env.events
                if not np.array_equal(record.episode.rewards,np.asarray([e['reward'] for e in events],np.float32)):raise AssertionError('Replay reward mismatch')
                failures=[e['t'] for e in events if e['async_failed_groups']]
                row=dict(arm=arm,episode=ep,seed=seed,model_seed=ms,method=method,epsilon=epsilon,
                    episode_return=summary['return'],length=summary['length'],success=summary['success'],
                    terminal_cause=terminal_cause(env),has_positive_reward=bool((record.episode.rewards>0).any()),
                    has_negative_reward=bool((record.episode.rewards<0).any()),async_failure_groups=env.env._fail_group,
                    first_failure_t=min(failures) if failures else None,
                    simultaneous_penalty=any(e['simultaneous_group_penalty'] for e in events))
                rows.append(row);records.append(record);traces.append(dict(**row,steps=events))
                steps+=summary['length']
                if steps>cfg['max_steps']:raise AssertionError('Step budget exceeded')
            save_replay(out/f'{arm}_replay.npz',records)
            with (out/f'{arm}_traces.jsonl').open('w') as f:
                for tr in traces:f.write(json.dumps(tr)+'\n')
            totals[arm]=aggregate(rows,traces)
            totals[arm]['by_model_seed']={str(seed):aggregate([r for r in rows if r['model_seed']==seed],[r for r in traces if r['model_seed']==seed]) for seed in models}
            allrows.extend(rows)
        save_csv(out/'episodes.csv',allrows)
        save_json(out/'summary.json',dict(historical=history,arms=totals,
            limitations=['Historical replay was not persisted; native reward-structure inference is distinguished from direct inspection.',
                        'Frozen initial behavior is not the historical evolving training policy or epsilon=0.2.',
                        'One failed group does not necessarily terminate; penalty events and terminal causes overlap.',
                        'No gradient/TD-path or oracle-injection test was run.']))
        meta.update(status='completed',episodes=len(allrows),transitions=steps)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));raise
    finally:save_json(out/'manifest.json',meta)
    print(out)

if __name__=='__main__':main()
