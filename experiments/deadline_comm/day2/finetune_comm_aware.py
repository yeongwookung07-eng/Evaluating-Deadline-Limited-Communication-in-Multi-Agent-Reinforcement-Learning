"""Adapt a final Full-Comm checkpoint using recorded DAC delivery histories."""
from copy import deepcopy
import gc
import numpy as np
import torch
from torch.nn import functional as F
from models.shared.buffers.episode import Episode
from ..replay import CommEpisode, StratifiedReplay
from ..oracle_bc_learner import OracleBCLearner
from ..adapters.hallway import Hallway
from ..common import load_config
from .policy import DecayedPolicy
from .followup_harness import Day2Harness
from .io import JSONLLog


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    n = metadata['environment']['n_agents']
    model = DecayedPolicy(2, 2*n, n, 3, **metadata['policy'])
    model.load_state_dict(saved['model'])
    return model, metadata


def trajectory(env, model, cfg, seed, oracle=False):
    from ..hallway_oracle_check import oracle_actions
    model.eval()
    obs, _ = env.reset(seed=seed)
    h = Day2Harness(model, 'dac', cfg, seed, measured=False)
    rng = np.random.default_rng(seed)
    observations, states, masks = [obs], [env.get_state()], [env.available_actions()]
    actions, rewards, terms, truncs, attempts, deliveries, sources = [], [], [], [], [], [], []
    for t in range(env.episode_length+1):
        a, _, messages = h.step(obs, masks[-1], t)
        attempt = np.zeros(env.n_agents, bool)
        delivery = np.zeros(env.n_agents, bool)
        for message in messages:
            attempt[message['sender']] = True
            delivery[message['sender']] = message['accepted']
        attempts.append(attempt)
        deliveries.append(delivery)
        sources.append(h.source[0].numpy().copy())
        if t == env.episode_length or (terms and (terms[-1] or truncs[-1])):
            break
        if oracle:
            a = oracle_actions(obs, env.env.group_ids)
        else:
            for i in range(env.n_agents):
                if rng.random() < .05:
                    a[i] = rng.choice(np.flatnonzero(masks[-1][i]))
        h.previous = F.one_hot(torch.as_tensor(a), model.n_actions).float()[None]
        obs, reward, term, trunc, _ = env.step(a)
        actions.append(a); rewards.append(reward); terms.append(term); truncs.append(trunc)
        observations.append(obs); states.append(env.get_state()); masks.append(env.available_actions())
    episode = Episode(np.array(observations,np.float32),np.array(states,np.float32),np.array(masks,bool),
                      np.array(actions),np.array(rewards,np.float32),np.array(terms),np.array(truncs))
    record = CommEpisode(episode,np.array(attempts),np.array(deliveries),np.array(sources),
                         np.zeros((env.n_agents,model.message_dim),np.float32),np.full(env.n_agents,-1))
    record.oracle_source = oracle
    return record


def finetune(checkpoint, harness, out, seed, updates=500, device="cpu"):
    torch.manual_seed(seed)
    model, metadata = load_model(checkpoint)
    model.decay_lambda = harness['decay_lambda']
    config = load_config('experiments/deadline_comm/configs/timing_rank.yaml')['learner']
    config.update(learning_rate=1e-4, target_update_interval=100, bootstrap_timeouts=False, bc_weight=.5)
    model.to(device)
    learner = OracleBCLearner(model, config)
    rollout_model = deepcopy(model).cpu()
    env = Hallway(**metadata['environment'])
    replay = StratifiedReplay(2000, seed)
    schedules = [(100000,5000000),(20000,5000000),(100000,2000000)]
    def collect(index, oracle=False):
        rollout_model.load_state_dict(model.state_dict())
        cfg = deepcopy(harness)
        cfg['link']['bandwidth_bytes_s'], cfg['deadline_ns'] = schedules[index%3]
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            return trajectory(env,rollout_model,cfg,3000000+seed*10000+index,oracle)
        finally:
            if enabled: gc.enable()
    for i in range(50):
        replay.insert(collect(i,True))
    log = JSONLLog(out/'finetune.jsonl')
    try:
        for update in range(updates):
            record = collect(100+update)
            replay.insert(record)
            batch = replay.sample(16)
            assert sum(r.oracle_source for r in batch)==4
            metrics = learner.train_batch(batch)
            log.append([dict(update=update+1,episode_return=float(record.episode.rewards.sum()),**metrics)])
            if (update+1)%100==0 or update+1==updates:
                learner.save_checkpoint(out/f'checkpoint_{update+1:04}.pt',metadata)
                print('FINETUNE',seed,update+1,metrics,flush=True)
    finally:
        log.close()
    return model.cpu()
