from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.deadline_comm.common import load_config,build_policy
from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.harness import rollout,DeadlineCommHarness
from experiments.deadline_comm.learner import unroll,CommLearner
from experiments.deadline_comm.replay import CommReplay,CommEpisode
from experiments.deadline_comm.gates import METHODS

CFG=Path(__file__).parents[1]/'experiments/deadline_comm/configs/smoke.yaml'

@pytest.mark.parametrize('method',METHODS)
def test_online_replay_q_consistency(method):
    torch.manual_seed(11)
    cfg=load_config(CFG); model=build_policy(cfg).eval()
    record,_,_,_=rollout(Hallway(**cfg['environment']),model,method,cfg['harness'],43,False)
    with torch.no_grad(): qs=unroll(model,record)
    h=DeadlineCommHarness(model,method,cfg['harness'],seed=100043,measured=False)
    for t,obs in enumerate(record.episode.observations):
        _,_,_,event,q=h.step(obs,record.episode.available_actions[t],t)
        np.testing.assert_allclose(q,qs[t][0].numpy(),rtol=1e-5,atol=1e-6)
        assert np.array_equal(event[2],record.source_step[t])
    replay=CommReplay(2); replay.insert(record)
    record.episode.observations[:]=999
    assert not (replay.sample(1)[0].episode.observations==999).all()
    learner=CommLearner(model,cfg['learner'])
    result=learner.train_batch(replay.sample(1))
    assert np.isfinite(result['td_loss'])
    assert model.message.weight.grad is not None

def test_source_history_rejected():
    cfg=load_config(CFG); model=build_policy(cfg)
    record,_,_,_=rollout(Hallway(**cfg['environment']),model,'full',cfg['harness'],1,False)
    record.source_step[0,0]=99
    with pytest.raises(ValueError): record.__post_init__()


def test_target_uses_own_message_weights_and_vdn_updates():
    cfg=load_config(CFG); cfg['policy']['mixer']='vdn'
    torch.manual_seed(8); model=build_policy(cfg)
    record,_,_,_=rollout(Hallway(**cfg['environment']),model,'full',cfg['harness'],14,False)
    learner=CommLearner(model,cfg['learner'])
    with torch.no_grad():
        before=unroll(learner.target,record)
        learner.target.message.bias.add_(10.)
        after=unroll(learner.target,record)
    assert any(not torch.allclose(a,b) for a,b in zip(before,after))
    assert all(not p.requires_grad for p in learner.target.parameters())
    assert np.isfinite(learner.train_batch([record])['td_loss'])
