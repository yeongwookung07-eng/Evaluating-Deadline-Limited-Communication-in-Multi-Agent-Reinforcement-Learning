import numpy as np
import torch
from experiments.deadline_comm.common import load_config
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.reviewer_validation.endpoints import EndpointHarness
from experiments.deadline_comm.reviewer_validation.stats import paired_bootstrap
from experiments.deadline_comm.reviewer_validation.train_ten_seeds import make_jobs,configure_arm,trajectory


def test_endpoints_and_replay():
    cfg=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    env=Hallway(n_agents=5,state_numbers=[4]*5,group_ids=[0,0,1,1,1])
    torch.manual_seed(0);model=DecayedPolicy(2,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix')
    for name in ('no_comm','ideal_full'):
        method,c=configure_arm(cfg,name,20000,1.5)
        obs,_=env.reset(seed=2);h=EndpointHarness(model,method,c,0,False)
        for t in range(2):
            _,row,_=h.step(obs,env.available_actions(),t)
            assert row['K_delivered']==(5 if name=='ideal_full' else 0)
        r=trajectory(env,model,method,c,123,True)
        assert bool(r.delivery_mask.all())==(name=='ideal_full')
        assert r.episode.rewards.sum()>0


def test_paired_bootstrap_preserves_pairing():
    a={s:float(s+1) for s in range(10)};b={s:float(s) for s in range(10)}
    result=paired_bootstrap(a,b)
    assert result['ci95_low']==result['ci95_high']==1
    assert paired_bootstrap(a,a)['mean_difference']==0


def test_ten_seeds_registered_counts():
    s=load_config('experiments/deadline_comm/configs/reviewer_ten_seeds.yaml');jobs=make_jobs(s)
    assert len(jobs)==180 and len({j['name'] for j in jobs})==180
    assert sum(j['updates'] for j in jobs)==270000


def test_hidden_sources_do_not_leak_and_oracle_solves():
    from experiments.deadline_comm.reviewer_validation.hidden_source import HiddenSource
    for task in ('broadcast','xor'):
        env=HiddenSource(10,task)
        obs,_=env.reset(seed=0)
        follower=obs[env.n_sources:].copy()
        env.bits=1-env.bits
        assert np.array_equal(follower,env.obs()[env.n_sources:])
        for _ in range(20):
            _,reward,_,_,_=env.step(env.oracle_actions())
            assert reward==1.


def test_persistent_queue_and_bounded_retries():
    from experiments.deadline_comm.reviewer_validation.queued_link import QueuedLink
    q=QueuedLink(1000)
    q.submit(0,0,48);q.submit(1,0,48)
    assert q.advance(.02)==[] and q.snapshot()['backlog_packets']==2
    arrivals=q.advance(.05)
    assert len(arrivals)==1 and arrivals[0]['source_step']==0
    assert len(q.advance(.1))==1
    q=QueuedLink(1000,loss_probability=1,max_retries=2)
    q.submit(0,0,48);assert q.advance(1)==[]
    assert q.wire==144 and q.retransmitted_wire==96 and q.failed==1


def test_queued_harness_uses_original_source_step():
    from experiments.deadline_comm.reviewer_validation.queued_harness import QueuedHarness
    cfg=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    cfg.update(loss=0.,retries=0,deadline_ns=1500000)
    cfg['link']['bandwidth_bytes_s']=20000
    env=Hallway(n_agents=5,state_numbers=[4]*5,group_ids=[0,0,1,1,1]);obs,_=env.reset(seed=0)
    model=DecayedPolicy(2,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix')
    h=QueuedHarness(model,'full',cfg,0,False)
    _,first=h.step(obs,env.available_actions(),0)
    assert first['received_payload_bytes']==0 and (h.source<0).all()
    rows=[h.step(obs,env.available_actions(),t)[1] for t in range(1,5)]
    assert any(r['received_old_packets']>0 for r in rows)
    assert (h.source==0).any()  # delayed packet retains generation step, not arrival step
