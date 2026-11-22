"""Real-engine adapter/replay/gradient smoke, never reported as policy learning."""
import hashlib,json,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from ..common import load_config
from ..day2.io import atomic_json
from ..day2.policy import DecayedPolicy
from ..batched_learner import BatchedCommLearner
from ..timing import configure_cpu
from .smac_adapter import SMACAdapter
from .train import collect, fingerprint

BASE=Path('experiments/deadline_comm')
def signature(seed,race):
    env=SMACAdapter(race,engine_seed=seed);h=hashlib.sha256();count=0
    try:
        obs,_=env.reset(seed);h.update(obs.tobytes());h.update(env.get_state().tobytes())
        for t in range(8):
            mask=env.available_actions();a=np.array([np.flatnonzero(m)[0] for m in mask]);obs,r,term,trunc,_=env.step(a)
            h.update(obs.tobytes());h.update(env.get_state().tobytes());h.update(np.array([r,term,trunc]).tobytes());count+=1
            if term or trunc:break
        return h.hexdigest(),count
    finally:env.close()
def main():
    configure_cpu();torch.manual_seed(123)
    setup=load_config(BASE/'configs/smacv2_setup.yaml');spec=load_config(BASE/'configs/smacv2_pilot.yaml')
    out=Path(setup['smoke_output']);out.mkdir(exist_ok=False);rows=[];start=time.time()
    atomic_json(out/'status.json',dict(stage='running'))
    try:
        for race in spec['scenarios']:
            a=signature(19000001,race);b=signature(19000001,race)
            assert a==b,('fresh-engine seeded replay mismatch',race,a,b)
            env=SMACAdapter(race,engine_seed=19000002)
            try:
                model=DecayedPolicy(env.obs_dim,env.state_dim,env.n_agents,env.n_actions,hidden_dim=64,message_dim=8,mixer='qmix')
                records=[];episodes=[]
                for ep,arm in enumerate(['ideal_full','no_comm']):
                    record,row=collect(env,model,arm,19000100+ep,1.0);records.append(record);episodes.append(row)
                    if arm=='ideal_full':assert record.delivery_mask.all()
                    else:assert not record.delivery_mask.any()
                gpu=deepcopy(model).to('cuda:0');learner=BatchedCommLearner(gpu,spec['learner']);before=fingerprint(gpu)
                metrics=learner.train_batch(records);assert fingerprint(gpu)!=before
                assert np.isfinite(metrics['td_loss']) and np.isfinite(metrics['grad_norm']) and metrics['grad_norm']>0
                rows.append(dict(race=race,seeded_trace=a[0],trace_steps=a[1],obs_dim=env.obs_dim,state_dim=env.state_dim,n_actions=env.n_actions,episode_limit=env.episode_length,episodes=episodes,learner=metrics))
                atomic_json(out/'progress.json',rows)
            finally:env.close()
        atomic_json(out/'summary.json',dict(stage='completed',results=rows,seconds=time.time()-start,scope='Random actions and one GPU update per scenario. Connectivity/gradient validation, not learning or native reproduction.'))
        atomic_json(out/'status.json',dict(stage='completed'));print(out,flush=True)
    except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
if __name__=='__main__':main()
