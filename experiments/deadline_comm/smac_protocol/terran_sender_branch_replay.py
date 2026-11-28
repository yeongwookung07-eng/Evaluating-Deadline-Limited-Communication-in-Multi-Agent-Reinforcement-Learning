"""Matched sender substitutions on three frozen SMACv2 Terran policies.

Run with the previously installed native SC2 environment. The immediate action
comparison shares one in-memory input. The suffix comparison requires exact
factual replay. Logged host CPU time is excluded from trajectory fingerprints.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from google.protobuf.json_format import MessageToDict
from s2clientprotocol import sc2api_pb2 as sc

from ..common import load_config, hash_file
from ..day2.io import atomic_json
from ..standard_benchmarks.smac_adapter import SMACAdapter
from ..timing import configure_cpu
from .evaluate import load_model, make_harness, delivery_ids


class ScopedSMAC(SMACAdapter):
    """Native get_obs mutates visibility bookkeeping and uses Python/NumPy RNG."""
    def observation(self):
        with self.local_rng():
            return super().observation()


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def native_view(response):
    obs = response.observation
    # Preserve order: the upstream wrapper uses the observed iteration order.
    return dict(loop=obs.game_loop, units=[MessageToDict(u, preserving_proto_field_name=True)
                                         for u in obs.raw_data.units])


def visibility(raw):
    return dict(owners=list(raw.enemy_tags), mask=raw.obs_enemies.tolist())


def scalar_diff(a, b, path=''):
    if type(a) is not type(b):
        return (path, str(type(a)), str(type(b)))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return (path, sorted(a.keys()), sorted(b.keys()))
        for k in a:
            v = scalar_diff(a[k], b[k], path+'/'+str(k))
            if v is not None:
                return v
    elif isinstance(a, list):
        if len(a) != len(b):
            return (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a,b)):
            v = scalar_diff(x, y, path+'/'+str(i))
            if v is not None:
                return v
    elif a != b:
        return (path,a,b)
    return None


def first_mismatch(first, repeat):
    for a,b in zip(first, repeat):
        keys=[k for k in a if a[k]!=b[k]]
        if keys:
            return dict(t=a['t'], fields=keys,
                        differences={k:scalar_diff(a[k], b[k]) for k in keys})
    if len(first)!=len(repeat):
        return dict(lengths=[len(first),len(repeat)])
    return None


def frozen_at(h, obs, mask, t, sender=None):
    # SelectionHarness queries select twice (including its post-packing check).
    if sender is not None:
        def select(scores, cap):
            assert cap == 1
            forced=torch.zeros(h.model.n_agents, dtype=torch.bool)
            forced[sender]=True
            return forced
        h.dac.select=select
    try:
        action,row,msg=h.step(obs,mask,t)
    finally:
        if sender is not None:
            del h.dac.select
    ids=delivery_ids(msg)
    if h.method=='rr_cap':
        assert len(ids)==1 and row['K_attempt']==1 and row['wire_bytes']==48
        if sender is not None:
            assert ids==[sender]
    return action,row,ids


def transition(env,h,obs,t,sender=None,detailed=False):
    state=env.get_state()
    mask=env.available_actions()
    raw=env.env.env
    # Snapshot the already-fetched native reply, without another get_obs call.
    pre_raw=native_view(raw._obs) if detailed else compact_native(raw._obs)
    vis=visibility(raw)
    action,row,ids=frozen_at(h,obs,mask,t,sender)
    post,reward,term,trunc,info=env.step(action)
    post_raw=native_view(raw._obs) if detailed else compact_native(raw._obs)
    def arr(x): return x.tolist() if detailed else fingerprint(x)
    def native(x):return x if detailed else fingerprint(x)
    rec=dict(t=t, pre_obs=arr(obs), pre_state=arr(state), pre_mask=arr(mask),
             pre_native=native(pre_raw), wrapper_visibility=native(vis),
             action=action.tolist(), sender=ids, payload=arr(h.cache), ts=arr(h.source),
             hidden=arr(h.hidden), previous=arr(h.previous), reward=reward,
             terminal=term, timeout=trunc, won=bool(info.get('battle_won',False)),
             post_obs=arr(post), post_state=arr(env.get_state()),
             post_mask=arr(env.available_actions()), post_native=native(post_raw))
    return post,rec


def run_episode(env,h,obs,t=0,first_sender=None,detailed=False):
    rows=[]
    while t < env.episode_length and not env.done:
        obs,rec=transition(env,h,obs,t,first_sender if not rows else None,detailed)
        rows.append(rec)
        t+=1
    return rows


def diagnose(spec,base,models):
    path=Path(spec['output'])
    seed=spec['diagnostic_reset']
    results=[]
    for cls in [SMACAdapter,ScopedSMAC]:
        cohorts=[]
        for i in range(spec['historical_repeats']):
            random.seed(197+i)
            np.random.seed(197+i)
            env=cls('terran',seed)
            try:
                obs,_=env.reset(seed)
                h=make_harness(models[spec['diagnostic_checkpoint']],'full',base,spec['condition'],'fixed',seed)
                rows=run_episode(env,h,obs,detailed=True)
            finally:
                env.close()
            # Full per-field comparison and compact raw fingerprints for replay.
            hashes=[{k:fingerprint(v) for k,v in row.items()} for row in rows]
            with gzip.open(path/f'diagnostic_{cls.__name__}_{i}.json.gz','wt') as f:
                json.dump(hashes,f)
            if not cohorts:
                first=rows
            legacy=['pre_obs','pre_state','pre_mask','action','payload','ts','hidden','sender','reward','terminal','timeout']
            summary=dict(adapter=cls.__name__,repeat=i,steps=len(rows),return_=sum(q['reward'] for q in rows),
                         exact=first_mismatch(first,rows),
                         legacy_fields=first_mismatch([{**{k:x[k] for k in legacy},'t':x['t']} for x in first],
                                                      [{**{k:x[k] for k in legacy},'t':x['t']} for x in rows]))
            cohorts.append(summary)
            print('Historical repeat',cls.__name__,i,'length',len(rows),'first',summary['exact'],flush=True)
        results.extend(cohorts)
        atomic_json(path/'diagnosis.json',results)
    return results


def load_inputs(spec):
    base=load_config(spec['evaluation_config'])
    paths=json.loads(Path(spec['checkpoint_map']).read_text())
    models={seed:load_model(paths[str(seed)]) for seed in spec['training_seeds']}
    return base,models,paths



def compact_native(response):
    obs = response.observation
    digest = hashlib.sha256(str(obs.game_loop).encode())
    for unit in obs.raw_data.units:
        value = unit.SerializeToString(deterministic=True)
        digest.update(len(value).to_bytes(4, 'little'))
        digest.update(value)
    return digest.hexdigest()


def write_trace(path, rows):
    with gzip.open(path, 'xt') as stream:
        json.dump(rows, stream, separators=(',', ':'))


def prefix(env, model, base, spec, reset):
    obs, _ = env.reset(reset)
    h = make_harness(model, 'rr_cap', base, spec['condition'], 'fixed', reset)
    rows = []
    while len(rows) < spec['branch_step'] and not env.done:
        obs, row = transition(env, h, obs, len(rows))
        rows.append(row)
    return h, obs, rows


def common_actions(env, h, obs, t):
    mask = env.available_actions()
    natural, natural_row, ids = frozen_at(deepcopy(h), obs, mask, t)
    candidates = []
    for sender in range(env.n_agents):
        branch = deepcopy(h)
        action, row, ids_forced = frozen_at(branch, obs, mask, t, sender)
        assert ids_forced == [sender] and row['K_delivered'] == 1
        assert row['wire_bytes'] == row['delivered_bytes'] == 48
        candidates.append(dict(sender=sender, action=action.tolist(), offered_bytes=row['wire_bytes'],
                               payload_bytes=4*h.model.message_dim, source=branch.source.tolist(),
                               cursor_after=branch.dac.cursor, budget_after=int(branch.dac.credit)))
    factual = ids[0]
    assert candidates[factual]['action'] == natural.tolist()
    assert natural_row['ready_ns'] == 500000
    return dict(factual_sender=factual, natural_action=natural.tolist(), candidates=candidates,
                prefix_hidden=fingerprint(h.hidden), prefix_cache=fingerprint(h.cache),
                prefix_timestamps=fingerprint(h.source), prefix_previous=fingerprint(h.previous),
                available=mask.tolist(), input_observation=fingerprint(obs),
                input_state=fingerprint(env.get_state()))


def quick_snapshot(env):
    raw = env.env.env
    client = raw._controller._client
    client.send(quick_save=sc.RequestQuickSave())
    handles = {'_controller', '_sc2_proc', '_run_config'}
    wrapper = {k:deepcopy(v) for k,v in vars(raw).items() if k not in handles}
    return (wrapper, env.steps, env.done, env.py_rng.getstate(), env.np_rng.get_state(),
            compact_native(raw._controller.observe()))


def quick_restore(env, snapshot):
    raw = env.env.env
    wrapper, env.steps, env.done, py_state, np_state, expected = snapshot
    raw._controller._client.send(quick_load=sc.RequestQuickLoad())
    obtained = compact_native(raw._controller.observe())
    for k,v in wrapper.items():
        setattr(raw, k, deepcopy(v))
    env.py_rng.setstate(py_state)
    env.np_rng.set_state(np_state)
    return expected, obtained


def evaluate_cell(mode, path, model, base, spec, seed, reset):
    cell = path/f'{mode}_s{seed}_r{reset}'
    cell.mkdir()
    env = ScopedSMAC('terran', reset)
    recovered = True
    replay_checks = []
    outcome = dict(seed=seed,reset=reset,mode=mode)
    try:
        h, obs, history = prefix(env, model, base, spec, reset)
        t = len(history)
        write_trace(cell/'prefix.json.gz', history)
        outcome.update(branch_step=t, reached=not env.done, prefix_reward=sum(r['reward'] for r in history))
        if env.done:
            outcome.update(valid=True, branches=[], factual_minus_uniform=0., action_changed=False)
            return outcome
        actions = common_actions(env, h, obs, t)
        outcome.update(actions)
        factual = actions['factual_sender']
        outcome['action_changed'] = any(x['action'] != actions['natural_action'] for x in actions['candidates'])
        # Save physical common-state actions independently of any later native replay failure.
        atomic_json(cell/'common_actions.json', outcome)
        snapshot = quick_snapshot(env) if mode == 'quick' else None
        factual_rows = run_episode(env,deepcopy(h),obs,t)
        write_trace(cell/'natural.json.gz',factual_rows)
        factual_reward = sum(r['reward'] for r in factual_rows)
        outcome.update(factual_suffix=factual_reward, factual_suffix_steps=len(factual_rows))
        if mode == 'fresh':
            env.close()
            env = None
        branches=[]
        for sender in [factual]+[i for i in range(model.n_agents) if i!=factual]:
            if mode == 'quick':
                expected, obtained = quick_restore(env,snapshot)
                restoration = None if expected==obtained else dict(expected=expected, obtained=obtained)
                previous_rows = None
                branch_h, branch_obs = deepcopy(h), obs.copy()
            else:
                env = ScopedSMAC('terran',reset)
                branch_h, branch_obs, previous_rows = prefix(env,model,base,spec,reset)
                write_trace(cell/f's{sender}_prefix.json.gz',previous_rows)
                restoration = first_mismatch(history,previous_rows)
            cursor=branch_h.dac.cursor
            rows = run_episode(env,branch_h,branch_obs,t,sender)
            if mode == 'fresh':
                env.close()
                env = None
            write_trace(cell/f's{sender}.json.gz',rows)
            noop = first_mismatch(factual_rows,rows) if sender==factual else None
            branch_action = rows[0]['action'] if rows else None
            expected_action = actions['candidates'][sender]['action']
            valid = restoration is None and noop is None and branch_action==expected_action
            replay_checks.append(dict(sender=sender,prefix_or_live_state=restoration,
                                      factual_replay=noop,same_as_physical_action=branch_action==expected_action))
            value = sum(r['reward'] for r in rows)
            branches.append(dict(sender=sender,action=branch_action,return_=value,steps=len(rows),
                                 won=rows[-1]['won'] if rows else None,cursor_before=cursor,
                                 offered_bytes=48,delivered_payload_bytes=32))
            recovered &= valid
            if not valid and mode=='quick':
                break
        outcome.update(valid=recovered,replay_checks=replay_checks,branches=branches,
                       factual_minus_uniform=factual_reward-float(np.mean([b['return_'] for b in branches]))
                       if recovered and len(branches)==model.n_agents else None)
        return outcome
    finally:
        if env is not None:
            env.close()


def summarize(rows):
    all_valid = all(x['valid'] for x in rows)
    seeds = []
    for seed in sorted({x['seed'] for x in rows}):
        group = [x for x in rows if x['seed']==seed]
        reached = [x for x in group if x['reached']]
        changed = sum(x['action_changed'] for x in group)
        diff = [x['factual_minus_uniform'] for x in group]
        alt_action = sum(c['action']!=x['natural_action'] for x in reached for c in x['candidates'])
        alt_return = sum(b['return_']!=x['factual_suffix'] for x in reached for b in x['branches'])
        seeds.append(dict(seed=seed,episodes=len(group), reached=len(reached),episodes_action_changed=changed,
                          changed_nonfactual_actions=alt_action,nonfactual_branches=4*len(reached),
                          changed_nonfactual_returns=alt_return if all_valid else None,
                          mean_factual_minus_uniform=float(np.mean(diff)) if all_valid else None))
    out = dict(valid=all_valid, per_seed=seeds)
    if all_valid and len(seeds)==3:
        vals = [s['mean_factual_minus_uniform'] for s in seeds]
        mean, se = np.mean(vals), np.std(vals,ddof=1)/np.sqrt(3)
        out.update(mean_factual_minus_uniform=mean,seed_t95=[mean-4.3026527297*se,mean+4.3026527297*se])
    return out


def run_branches(spec,base,models):
    path=Path(spec['output'])
    manifest = path/'branch_manifest.json'
    assert not manifest.exists()
    atomic_json(manifest,dict(config=spec, config_hash=hash_file(args.config), source_hash=hash_file(__file__),
                              checkpoint_hashes={k:hash_file(v) for k,v in paths.items()}, started=time.time()))
    start=time.monotonic()
    # If native restoration fails its gate, retain the failed route and recollect
    # the entire prespecified cohort with independently launched engines.
    for mode in ['quick','fresh']:
        results=[]
        for seed in spec['training_seeds']:
            for i in range(spec['episodes_per_seed']):
                if time.monotonic()-start > spec['wall_seconds']:
                    raise TimeoutError('registered CPU budget exhausted')
                reset=spec['fresh_reset_base']+i
                try:
                    outcome=evaluate_cell(mode,path,models[seed],base,spec,seed,reset)
                except Exception as exc:
                    if mode!='quick':
                        raise
                    import traceback
                    outcome=dict(seed=seed,reset=reset,mode=mode,valid=False,
                                 failure=repr(exc),traceback=traceback.format_exc())
                results.append(outcome)
                atomic_json(path/f'{mode}_episodes.json',results)
                print('Finished',mode,seed,reset,'replay',outcome['valid'], 'action change',
                      outcome.get('action_changed'),flush=True)
                if mode=='quick' and not outcome['valid']:
                    break
            if mode=='quick' and not results[-1]['valid']:
                break
        if len(results)==len(models)*spec['episodes_per_seed']:
            summary=summarize(results)
            atomic_json(path/'summary.json',dict(mode=mode,elapsed=time.monotonic()-start,**summary))
            return
        assert mode=='quick'


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--phase',choices=['diagnosis','branches'],required=True)
    args=ap.parse_args()
    spec=load_config(args.config)
    configure_cpu()
    base,models,paths=load_inputs(spec)
    out=Path(spec['output'])
    if args.phase=='diagnosis':
        assert not (out/'diagnosis_manifest.json').exists()
        atomic_json(out/'diagnosis_manifest.json',dict(config=spec,config_hash=hash_file(args.config),
                    checkpoint_hashes={k:hash_file(v) for k,v in paths.items()},init_time=time.time()))
        diagnose(spec,base,models)
    else:
        run_branches(spec,base,models)
