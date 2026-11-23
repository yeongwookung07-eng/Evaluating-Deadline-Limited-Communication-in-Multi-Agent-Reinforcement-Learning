"""Bounded supervised follower-head changes evaluated in actual closed loop."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from scipy.stats import t as student_t
from torch import nn

from ..common import load_config
from ..day2.io import atomic_json, JSONLLog
from ..day2.policy import DecayedPolicy
from ..reviewer_validation.hidden_source import HiddenSource
from ..reviewer_validation.train_ten_seeds import configure_arm
from ..reviewer_validation.endpoints import EndpointHarness
from ..timing import configure_cpu
from ..tmlr_revision.experiments import setup, clone
from ..tmlr_revision.robustness.aggregate_probe import ridge_prediction


class SplitUtility(nn.Module):
    """Keep source behavior unchanged while adapting the shared follower head."""
    def __init__(self, source, follower):
        super().__init__()
        self.source = deepcopy(source)
        self.follower = deepcopy(follower)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1] == 5
        # Preserve the original batch shape for exact source-head arithmetic.
        # The doubled head evaluation is charged equally for every arm.
        return torch.cat((self.source(x)[:, :1], self.follower(x)[:, 1:]), dim=1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    a = np.asarray(values, float)
    delta = float(student_t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a)>1 else None
    return dict(mean=float(a.mean()), n=len(a), ci95=None if delta is None else [float(a.mean()-delta), float(a.mean()+delta)], seed_values=a.tolist())


def fit_head(original, x, y, settings, seed, shuffle, check_budget):
    head = deepcopy(original)
    for p in head.parameters():
        p.requires_grad_(True)
    if shuffle:
        # Shuffle episode/time labels, keeping the four receiver copies together.
        y = np.random.default_rng(710000 + seed).permutation(y)
    inputs = torch.as_tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float32)
    labels = torch.as_tensor(np.repeat(y, x.shape[1]), dtype=torch.long)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
    start = time.perf_counter_ns()
    losses = []
    for _ in range(settings['updates']):
        check_budget()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(inputs), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return head.eval(), dict(cpu_wall_ns=time.perf_counter_ns()-start, updates=len(losses), initial_loss=losses[0], final_loss=losses[-1], labeled_timesteps=len(y), receiver_examples=len(labels), parameters=sum(p.numel() for p in head.parameters()))


def rollout(model, settings, seed, scheduler, arm, output, check_budget):
    actual, cfg = configure_arm(setup(settings['condition']), settings['method'], 100000, 2.)
    cfg['fixed_ready'] = True
    captured = {}
    handle = model.utility.register_forward_pre_hook(lambda _, args: captured.update(x=args[0].detach().numpy().copy()))
    episodes, features, labels = [], [], []
    try:
        with torch.inference_mode():
            for episode in range(settings['test_episodes']):
                check_budget()
                env = HiddenSource()
                reset = settings['test_reset_base'] + episode
                obs, _ = env.reset(seed=reset)
                h = EndpointHarness(model, actual, cfg, reset, True)
                trace = []
                for step in range(env.episode_length):
                    chosen = None
                    overhead = 0
                    if scheduler == 'source_aware':
                        start = time.perf_counter_ns()
                        _, factual_row, factual_logs = clone(h).step(obs.copy(), env.available_actions(), step)
                        selected = sorted(m['sender'] for m in factual_logs if m['accepted'])
                        k = len(selected)
                        chosen = sorted(([0] + [i for i in selected if i != 0])[:k]) if k else []
                        overhead = time.perf_counter_ns() - start
                    def select(scores, kmax):
                        mask = torch.zeros(5, dtype=torch.bool)
                        mask[chosen] = True
                        assert len(chosen) <= kmax
                        return mask
                    outer_start = time.perf_counter_ns()
                    if chosen is None:
                        action, row, logs = h.step(obs, env.available_actions(), step)
                    else:
                        with patch.object(h.dac, 'select', side_effect=select):
                            action, row, logs = h.step(obs, env.available_actions(), step)
                    actual_call_ns = time.perf_counter_ns() - outer_start
                    if chosen is not None:
                        assert row['K_delivered'] == factual_row['K_delivered']
                    x = captured['x'][0, 1:].copy()
                    bit = int(env.bits[0])
                    features.append(x)
                    labels.append(bit)
                    obs, reward, done, _, _ = env.step(action)
                    assert reward == float((action[1:] == bit).mean())
                    total_cpu = overhead + actual_call_ns
                    trace.append(dict(t=step, bit=bit, actions=action.tolist(), reward=reward,
                        offered_wire_bytes=row['wire_bytes'], delivered_payload_bytes=32*row['K_delivered'],
                        delivered_senders=sorted(m['sender'] for m in logs if m['accepted']),
                        source_timestamps=h.source[0].tolist(),
                        source_received=any(m['sender']==0 and m['accepted'] for m in logs),
                        cpu_harness_ns=row['cpu_total_ns'], diagnostic_selection_ns=overhead,
                        cpu_total_including_diagnostic_ns=total_cpu, modeled_wait_ns=row['comm_model_ns'],
                        readiness_ns=row['ready_ns'],
                        deadline_miss_including_diagnostic=total_cpu+row['comm_model_ns']>cfg['deadline_ns']))
                    if done:
                        break
                result=dict(seed=seed, scheduler=scheduler, arm=arm, episode=episode, reset=reset,
                    accuracy=float(np.mean([v['reward'] for v in trace])),
                    return_=sum(v['reward'] for v in trace), trace=trace)
                episodes.append(result)
                output.append([result])
    finally:
        handle.remove()
    return episodes, np.asarray(features), np.asarray(labels)


def main():
    configure_cpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    args=parser.parse_args()
    settings=load_config(args.config)
    if args.smoke:
        settings.update(output=settings['output']+'_smoke_v2', seeds=[0], test_episodes=2, updates=10)
    out=Path(settings['output'])
    out.mkdir(parents=True, exist_ok=False)
    settings['entry_sha256']=digest(__file__)
    atomic_json(out/'registration.json', settings)
    start=time.monotonic()
    def check_budget():
        if time.monotonic()-start>settings['wall_seconds']:
            raise TimeoutError('Registered receiver-head adaptation budget exhausted')
    original_gate=json.loads(Path('experiments/deadline_comm/runs/hidden_mechanism_20261119_001/summary.json').read_text())['records']
    gate=[r for r in original_gate if r['task']=='broadcast' and r['method']=='ideal_full' and r['condition']=='medium_2ms']
    assert len(gate)==10 and sum(r['accuracy']>=.75 for r in gate)>=8
    rows, probes, training, hashes = [], [], [], []
    checks=dict(historical_feature_policy_cells=0, unchanged_source_head_checks=0, frozen_parameter_checks=0, matched_delivery_cells=0)
    log=JSONLLog(out/'episodes.jsonl')
    try:
        for seed in settings['seeds']:
            checkpoint=Path(settings['checkpoint_template'].format(seed=seed))
            base=DecayedPolicy(4, 10, 5, 2, hidden_dim=64, message_dim=8, mixer='qmix')
            base.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
            base.eval()
            for p in base.parameters():
                p.requires_grad_(False)
            hashes.append(dict(path=str(checkpoint), sha256=digest(checkpoint)))
            for scheduler in settings['schedulers']:
                path=Path(settings['feature_template'].format(seed=seed, scheduler=scheduler))
                hashes.append(dict(path=str(path), sha256=digest(path)))
                with np.load(path, allow_pickle=False) as data:
                    fit=data['episode']<settings['train_episodes']
                    train_x=data['utility_input'][fit].copy()
                    train_y=data['bit'][fit].copy()
                    assert set(data['episode'][fit]).isdisjoint(set(data['episode'][~fit]))
                    with torch.inference_mode():
                        prediction=base.utility(torch.as_tensor(data['utility_input'],dtype=torch.float32)).argmax(-1).numpy()
                    np.testing.assert_array_equal(prediction==data['bit'][:,None], data['policy_correct'].astype(bool))
                checks['historical_feature_policy_cells']+=1
                fitted={}
                for label, shuffled in [('adapted', False), ('shuffled', True)]:
                    fitted[label], info=fit_head(base.utility, train_x, train_y, settings, seed, shuffled, check_budget)
                    training.append(dict(seed=seed, scheduler=scheduler, arm=label, **info))
                    torch.save(fitted[label].state_dict(), out/f'seed{seed}_{scheduler}_{label}_head.pt')
                reference=None
                # Alternate execution order across seeds to avoid one fixed timing order.
                order=['original','adapted','shuffled'] if seed%2==0 else ['original','shuffled','adapted']
                for arm in order:
                    model=deepcopy(base)
                    model.utility=SplitUtility(base.utility, base.utility if arm=='original' else fitted[arm])
                    model.eval()
                    for p in model.parameters():
                        p.requires_grad_(False)
                    for name, value in base.state_dict().items():
                        if not name.startswith('utility.'):
                            assert torch.equal(value, model.state_dict()[name])
                    checks['frozen_parameter_checks']+=1
                    with torch.inference_mode():
                        sample=torch.zeros(1,5,74)
                        assert torch.equal(model.utility(sample)[:,0],base.utility(sample)[:,0])
                    checks['unchanged_source_head_checks']+=1
                    result, x, y=rollout(model, settings, seed, scheduler, arm, log, check_budget)
                    trace=[t for e in result for t in e['trace']]
                    delivery=[(r['offered_wire_bytes'],r['delivered_payload_bytes'],r['delivered_senders'],r['source_timestamps']) for r in trace]
                    if arm=='original':
                        reference=delivery
                        probe={}
                        for kind in ['aggregate','full_input']:
                            hits=[]
                            for receiver in range(4):
                                tr=train_x[:,receiver,64:72] if kind=='aggregate' else train_x[:,receiver]
                                te=x[:,receiver,64:72] if kind=='aggregate' else x[:,receiver]
                                hits.append(ridge_prediction(tr,train_y,te,settings['ridge'])==y)
                            probe[kind]=float(np.asarray(hits).mean())
                        probes.append(dict(seed=seed, scheduler=scheduler, policy_accuracy=float(np.mean([r['reward'] for r in trace])), **probe))
                    else:
                        assert delivery==reference
                        checks['matched_delivery_cells']+=1
                    rows.append(dict(seed=seed, scheduler=scheduler, arm=arm,
                        accuracy=float(np.mean([r['accuracy'] for r in result])),
                        mean_return=float(np.mean([r['return_'] for r in result])),
                        wire_B_step=float(np.mean([r['offered_wire_bytes'] for r in trace])),
                        payload_B_step=float(np.mean([r['delivered_payload_bytes'] for r in trace])),
                        source_delivery=float(np.mean([r['source_received'] for r in trace])),
                        cpu_total_us=float(np.mean([r['cpu_total_including_diagnostic_ns'] for r in trace]))/1000,
                        diagnostic_selection_us=float(np.mean([r['diagnostic_selection_ns'] for r in trace]))/1000,
                        deadline_miss_fraction=float(np.mean([r['deadline_miss_including_diagnostic'] for r in trace]))))
                atomic_json(out/'status.json',dict(stage='running', seed=seed, scheduler=scheduler, cells=len(rows)))
        aggregate={}
        for scheduler in settings['schedulers']:
            indexed={(r['seed'],r['arm']):r for r in rows if r['scheduler']==scheduler}
            contrasts={}
            for first, second in [('adapted','original'),('adapted','shuffled'),('shuffled','original')]:
                contrasts[first+'_minus_'+second]=interval([100*(indexed[seed,first]['accuracy']-indexed[seed,second]['accuracy']) for seed in settings['seeds']])
            probe_rows=[p for p in probes if p['scheduler']==scheduler]
            contrasts['aggregate_probe_minus_policy']=interval([100*(p['aggregate']-p['policy_accuracy']) for p in probe_rows])
            contrasts['diagnostic_adoption_gate']=all(contrasts[name]['ci95'] is not None and contrasts[name]['ci95'][0]>0 for name in ['adapted_minus_original','adapted_minus_shuffled'])
            aggregate[scheduler]=contrasts
        atomic_json(out/'input_hashes.json',hashes)
        atomic_json(out/'summary.json',dict(rows=rows, probes=probes, training=training, aggregate=aggregate, checks=checks,
            original_learning_gate={'seeds':10,'passed':sum(r['accuracy']>=.75 for r in gate)},
            scope=settings['scope'], wall_seconds=time.monotonic()-start))
        atomic_json(out/'status.json',dict(stage='completed',cells=len(rows),wall_seconds=time.monotonic()-start,checks=checks))
        print(json.dumps(aggregate,indent=2))
    except Exception as exc:
        atomic_json(out/'partial.json',dict(rows=rows,probes=probes,training=training,checks=checks))
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc),cells=len(rows),wall_seconds=time.monotonic()-start))
        raise
    finally:
        log.close()


if __name__=='__main__':
    main()
