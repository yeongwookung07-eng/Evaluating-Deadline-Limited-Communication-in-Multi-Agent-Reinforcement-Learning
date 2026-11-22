"""Explicit non-exact continuation of retained SMACv2 Ideal checkpoints."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..common import load_config
from ..day2.io import atomic_json
from ..day2.policy import DecayedPolicy
from ..batched_learner import BatchedCommLearner
from ..replay import CommReplay
from ..timing import configure_cpu
from ..standard_benchmarks.smac_adapter import SMACAdapter
from ..standard_benchmarks.train import collect, evaluate, fingerprint


def epsilon(spec, total, origin):
    frac=min(1.,max(0.,total-origin)/spec['continuation_epsilon_steps'])
    return spec['continuation_epsilon_start']+(spec['continuation_epsilon_final']-spec['continuation_epsilon_start'])*frac


def run(spec, checkpoint, seed, milestone, device, deadline):
    configure_cpu();torch.manual_seed(seed);np.random.seed(seed)
    out=Path(spec['output'])/f'phase_{milestone}'/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=False)
    start=time.time();saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    metadata=deepcopy(saved['metadata']);dims=metadata['dimensions']
    model=DecayedPolicy(dims['obs'],dims['state'],dims['agents'],dims['actions'],**metadata['policy']).to(device)
    model.decay_lambda=1.;learner=BatchedCommLearner(model,spec['learner'])
    learner.load_checkpoint(checkpoint)
    steps=int(metadata['environment_steps']);origin=int(metadata.get('continuation_origin_steps',steps))
    episode=int(metadata.get('continuation_episodes',0));metadata['continuation_origin_steps']=origin
    initial=fingerprint(model);roll=deepcopy(model).cpu();replay=CommReplay(spec['replay_capacity'],seed+milestone)
    env=SMACAdapter(spec['scenario'],engine_seed=spec['train_seed_base']+seed)
    atomic_json(out/'manifest.json',dict(spec=spec,seed=seed,milestone=milestone,source_checkpoint=str(checkpoint),source_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),source_steps=steps,device=device,started=start,exact_resume=False,replay_reinitialized=True))
    next_progress=steps+10000;metrics={}
    try:
        with (out/'training.jsonl').open('w',buffering=1) as log:
            while steps<milestone:
                if time.time()>deadline:raise TimeoutError('Registered campaign wall budget exceeded')
                roll.load_state_dict(model.state_dict())
                record,row=collect(env,roll,'ideal_full',spec['train_seed_base']+seed*100000+episode,epsilon(spec,steps,origin))
                steps+=len(record.episode.actions);episode+=1;replay.insert(record)
                if len(replay)>=spec['batch_size']:
                    metrics=learner.train_batch(replay.sample(spec['batch_size']))
                log.write(json.dumps(dict(episode=episode,environment_steps=steps,epsilon=epsilon(spec,steps,origin),**row,**metrics,seconds=time.time()-start))+'\n')
                atomic_json(out/'status.json',dict(stage='training',environment_steps=steps,target=milestone,updates=learner.updates,seconds=time.time()-start))
                if steps>=next_progress:
                    metadata.update(environment_steps=steps,updates=learner.updates,continuation_episodes=episode)
                    learner.save_checkpoint(out/'checkpoint_latest.pt',metadata)
                    print('seed',seed,'steps',steps,'td',metrics.get('td_loss'),flush=True);next_progress=steps+10000
        metadata.update(environment_steps=steps,updates=learner.updates,continuation_episodes=episode)
        ckpt=out/'checkpoint_final.pt';learner.save_checkpoint(ckpt,metadata)
        assert fingerprint(model)!=initial
        env.close();env=None;gc.collect()
        roll.load_state_dict(model.state_dict())
        atomic_json(out/'status.json',dict(stage='gate_evaluation',environment_steps=steps,updates=learner.updates))
        gate_spec=dict(spec,eval_seed_base=spec['gate_seed_base'],evaluation_episodes=spec['gate_episodes'])
        result=evaluate(roll,spec['scenario'],'ideal_full',gate_spec)
        result.update(environment_steps=steps,updates=learner.updates,checkpoint=str(ckpt),seconds=time.time()-start,learning_gate=result['win_rate']>=spec['learning_gate_win_rate'])
        atomic_json(out/'gate.json',result)
        atomic_json(out/'status.json',dict(stage='completed',environment_steps=steps,updates=learner.updates,win_rate=result['win_rate'],seconds=time.time()-start))
        print('GATE',seed,milestone,result['win_rate'],result['mean_return'],flush=True)
    except Exception as exc:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc),environment_steps=steps));raise
    finally:
        if env is not None:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--milestone',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();run(load_config(a.config),a.checkpoint,a.seed,a.milestone,a.device,a.deadline)
if __name__=='__main__':main()
