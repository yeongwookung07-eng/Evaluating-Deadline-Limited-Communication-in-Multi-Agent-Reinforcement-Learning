"""Sample-budgeted Oracle-free SMACv2 feasibility training, not native reproduction."""
import argparse, gc, hashlib, json, os, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.shared.buffers.episode import Episode
from ..common import load_config
from ..day2.io import atomic_json
from ..day2.policy import DecayedPolicy
from ..batched_learner import BatchedCommLearner
from ..replay import CommEpisode, CommReplay
from ..reviewer_validation.endpoints import EndpointHarness
from ..reviewer_validation.train_ten_seeds import configure_arm
from ..timing import configure_cpu
from .smac_adapter import SMACAdapter

BASE=Path('experiments/deadline_comm')
def harness_config(arm):
    base=load_config(BASE/'configs/day2_dac_marl_main.yaml')['harness']
    return configure_arm(base,arm,100000,2.)

@torch.inference_mode()
def collect(env,model,arm,seed,epsilon):
    model.eval();obs,_=env.reset(seed=seed);initial_hash=hashlib.sha256(env.get_state().tobytes()).hexdigest()
    method,cfg=harness_config(arm);h=EndpointHarness(model,method,cfg,seed,False)
    rng=np.random.default_rng(seed+17000000)
    observations,states,masks=[obs],[env.get_state()],[env.available_actions()]
    actions=[];rewards=[];terms=[];truncs=[];attempts=[];deliveries=[];sources=[]
    env_seconds=0.;infer_seconds=0.;final_info={}
    for t in range(env.episode_length+1):
        start=time.perf_counter();a,row,messages=h.step(obs,masks[-1],t);infer_seconds+=time.perf_counter()-start
        attempt,delivery=np.zeros(env.n_agents,bool),np.zeros(env.n_agents,bool)
        for m in messages:attempt[m['sender']]=True;delivery[m['sender']]=m['accepted']
        attempts.append(attempt);deliveries.append(delivery);sources.append(h.source[0].numpy().copy())
        if t==env.episode_length or (terms and (terms[-1] or truncs[-1])):break
        for i in range(env.n_agents):
            if rng.random()<epsilon:a[i]=rng.choice(np.flatnonzero(masks[-1][i]))
        assert masks[-1][np.arange(env.n_agents),a].all()
        h.previous=F.one_hot(torch.as_tensor(a),model.n_actions).float()[None]
        start=time.perf_counter();obs,reward,term,trunc,final_info=env.step(a);env_seconds+=time.perf_counter()-start
        actions.append(a.copy());rewards.append(reward);terms.append(term);truncs.append(trunc)
        observations.append(obs);states.append(env.get_state());masks.append(env.available_actions())
    ep=Episode(np.array(observations,np.float32),np.array(states,np.float32),np.array(masks,bool),np.array(actions),np.array(rewards,np.float32),np.array(terms),np.array(truncs))
    record=CommEpisode(ep,np.array(attempts),np.array(deliveries),np.array(sources),np.zeros((env.n_agents,model.message_dim),np.float32),np.full(env.n_agents,-1))
    return record,dict(return_=float(ep.rewards.sum()),won=bool(final_info.get('battle_won',False)),length=len(actions),terminated=bool(terms[-1]),truncated=bool(truncs[-1]),initial_state_sha256=initial_hash,scenario_seed=seed,simulator_seconds=env_seconds,rollout_inference_seconds=infer_seconds)

def fingerprint(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def evaluate(model,race,arm,spec):
    env=SMACAdapter(race,engine_seed=spec['eval_seed_base']);rows=[]
    try:
        for i in range(spec['evaluation_episodes']):
            _,r=collect(env,model,arm,spec['eval_seed_base']+i,0.)
            rows.append(dict(episode=i,**r))
    finally:env.close()
    return dict(mean_return=float(np.mean([r['return_'] for r in rows])),win_rate=float(np.mean([r['won'] for r in rows])),episodes=rows)

def run(spec,race,arm,seed,device,deadline):
    configure_cpu();torch.manual_seed(seed);np.random.seed(seed)
    out=Path(spec['output'])/f'{race}_seed{seed}_{arm}';out.mkdir(parents=True,exist_ok=False)
    env=SMACAdapter(race,engine_seed=spec['train_seed_base']+seed);start=time.time()
    try:
        model=DecayedPolicy(env.obs_dim,env.state_dim,env.n_agents,env.n_actions,hidden_dim=spec['hidden_dim'],message_dim=spec['message_dim'],mixer='qmix').to(device)
        model.decay_lambda=1.;initial=fingerprint(model)
        learner=BatchedCommLearner(model,spec['learner']);roll=deepcopy(model).cpu();replay=CommReplay(spec['replay_capacity'],seed)
        metadata=dict(environment=env.config,dimensions=dict(obs=env.obs_dim,state=env.state_dim,agents=env.n_agents,actions=env.n_actions),policy=dict(hidden_dim=spec['hidden_dim'],message_dim=spec['message_dim'],mixer='qmix'),learner=spec['learner'],race=race,arm=arm,seed=seed,oracle=False,initial_parameter_sha256=initial)
        atomic_json(out/'manifest.json',dict(spec=spec,metadata=metadata,device=device,pid=os.getpid(),started=start,engine_seed=env.engine_seed,scope='Local communication-conditioned QMIX adaptation; not native reproduction. No convergence assumed at pilot budget.'))
        steps=0;episode=0;next_eval=spec['evaluation_interval'];last_metrics={}
        def checkpoint_and_eval(tag):
            if time.time()>deadline:raise TimeoutError('campaign wall budget reached')
            atomic_json(out/'status.json',dict(stage='evaluating',checkpoint=tag,environment_steps=steps,updates=learner.updates,seconds=time.time()-start))
            roll.load_state_dict(model.state_dict());metadata['environment_steps']=steps;metadata['updates']=learner.updates
            learner.save_checkpoint(out/f'checkpoint_{tag}.pt',metadata)
            result=evaluate(roll,race,arm,spec);result.update(environment_steps=steps,updates=learner.updates,seconds=time.time()-start)
            atomic_json(out/f'evaluation_{tag}.json',result)
            print(out.name,tag,'steps',steps,'win',result['win_rate'],'return',result['mean_return'],flush=True)
            return result
        initial_eval=checkpoint_and_eval('initial')
        with (out/'training.jsonl').open('w',buffering=1) as log:
            while steps<spec['environment_steps']:
                if time.time()>deadline:raise TimeoutError('campaign wall budget reached')
                roll.load_state_dict(model.state_dict())
                eps=max(spec['epsilon_final'],1-(1-spec['epsilon_final'])*steps/spec['epsilon_anneal_steps'])
                record,metrics=collect(env,roll,arm,spec['train_seed_base']+seed*100000+episode,eps)
                steps+=len(record.episode.actions);episode+=1;replay.insert(record)
                if len(replay)>=spec['batch_size']:
                    for _ in range(spec['updates_per_episode']):last_metrics=learner.train_batch(replay.sample(spec['batch_size']))
                log.write(json.dumps(dict(episode=episode,environment_steps=steps,epsilon=eps,**metrics,**last_metrics,seconds=time.time()-start))+'\n')
                atomic_json(out/'status.json',dict(stage='training',episode=episode,environment_steps=steps,updates=learner.updates,seconds=time.time()-start))
                if steps>=next_eval or steps>=spec['environment_steps']:
                    final=checkpoint_and_eval(str(steps));next_eval+=spec['evaluation_interval']
            changed=fingerprint(model)!=initial
            assert changed and learner.updates>0
            atomic_json(out/'summary.json',dict(stage='completed',environment_steps=steps,updates=learner.updates,initial_win_rate=initial_eval['win_rate'],final_win_rate=final['win_rate'],final_return=final['mean_return'],parameters_changed=changed,learning_gate=final['win_rate']>=spec['learning_gate_win_rate'],seconds=time.time()-start,final_checkpoint=str(out/f'checkpoint_{steps}.pt')))
            atomic_json(out/'status.json',dict(stage='completed',environment_steps=steps,updates=learner.updates))
    except Exception as exc:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise
    finally:env.close();gc.collect()

def main():
    p=argparse.ArgumentParser();p.add_argument('--race',required=True);p.add_argument('--arm',required=True,choices=['ideal_full','no_comm']);p.add_argument('--seed',required=True,type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    spec=load_config(BASE/'configs/smacv2_pilot.yaml');run(spec,args.race,args.arm,args.seed,args.device,args.deadline)
if __name__=='__main__':main()
