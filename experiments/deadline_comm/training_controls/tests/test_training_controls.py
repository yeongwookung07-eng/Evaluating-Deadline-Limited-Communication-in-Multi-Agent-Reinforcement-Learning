from copy import deepcopy
import torch
from experiments.deadline_comm.common import load_config
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.day2.followup_harness import Day2Harness
from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.oracle_bc_learner import OracleBCLearner
from experiments.deadline_comm.training_controls.run import configure_arm, trajectory, make_jobs


def setup():
    base=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    env=Hallway(n_agents=5,state_numbers=[4]*5,group_ids=[0,0,1,1,1])
    torch.manual_seed(0)
    model=DecayedPolicy(2,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix')
    return base,env,model


def test_ablations_apply_to_training_and_target():
    base,env,model=setup(); original=deepcopy(base)
    for arm in ('dac_no_decay','dac_no_aoi','dac_no_cap'):
        method,cfg=configure_arm(base,arm,20000,5.)
        model.decay_lambda=cfg['decay_lambda']
        learner=OracleBCLearner(model,dict(optimizer="adam",learning_rate=1e-4,gamma=.99,grad_clip=10.,target_update_interval=100,bootstrap_timeouts=False))
        assert learner.target.decay_lambda==cfg['decay_lambda']
        h=Day2Harness(model,method,cfg,0,False)
        if arm=='dac_no_decay':assert model.decay_lambda==1.
        if arm=='dac_no_aoi':assert h.dac.alpha==0.
        if arm=='dac_no_cap':
            obs,_=env.reset(seed=0)
            _,row,_=h.step(obs,env.available_actions(),0)
            assert row['K_attempt']==2 and row['K_delivered']==1
    assert base==original


def test_oracle_replay_and_update_all_arms():
    base,env,model=setup()
    for arm in ('full_continuation','dac','dac_no_cap','dac_no_aoi','dac_no_decay','full','periodic'):
        method,cfg=configure_arm(base,arm,20000,5.)
        demo=trajectory(env,model,method,cfg,123,True)
        online=trajectory(env,model,method,cfg,124,False)
        assert demo.episode.rewards.sum()>0
        learner=OracleBCLearner(model,dict(optimizer="adam",learning_rate=1e-4,gamma=.99,grad_clip=10.,target_update_interval=100,bootstrap_timeouts=False))
        result=learner.train_batch([demo,online,online,online])
        assert result['updates']==1 and result['grad_norm']>0
        if arm=='full_continuation':assert demo.delivery_mask.all()
        if arm=='full':assert (demo.delivery_mask.sum(-1)<=1).all()


def test_registered_budget():
    cfg=load_config('experiments/deadline_comm/configs/training_controls.yaml')
    jobs=make_jobs(cfg)
    assert len(jobs)==24 and sum(j['updates'] for j in jobs)==21000
    assert len({j['name'] for j in jobs})==24
