import copy
import random
import numpy as np
import torch
from experiments.deadline_comm.common import load_config,build_policy
from experiments.deadline_comm.harness import DeadlineCommHarness
from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.gates import Gate
from pathlib import Path

CFG=Path(__file__).parents[1]/'experiments/deadline_comm/configs/smoke.yaml'

def test_drops_cannot_change_cache_or_source():
    cfg=load_config(CFG); model=build_policy(cfg).eval()
    hc=copy.deepcopy(cfg['harness']); hc.update(deadline_ns=100,r_post_ns=0,training_ready_ns=0)
    hc['link'].update(bandwidth_bytes_s=1e15,latency_ns=0)
    harness=DeadlineCommHarness(model,'full',hc,measured=False)
    obs=np.ones((5,2),np.float32); mask=np.ones((5,3),bool)
    _,_,_,(_,delivery,source),_=harness.step(obs,mask,0)
    assert delivery.all() and (source==0).all()
    old=harness.cache.clone()
    harness.link.bandwidth=1
    _,row,_,(_,delivery,source),q=harness.step(obs*2,mask,1)
    assert not delivery.any() and (source==0).all()
    assert torch.equal(old,harness.cache)
    with torch.no_grad(): expected=model.q(harness.hidden,old,harness.source,1)[0].numpy()
    np.testing.assert_allclose(q,expected)
    assert row['mean_age']==1
    # TMC sender memory advances independently of receiver success.
    assert all(v is not None for v in harness.gate.last)

def test_initial_invalid_and_compute_miss():
    cfg=load_config(CFG); model=build_policy(cfg)
    hc=copy.deepcopy(cfg['harness']); hc.update(deadline_ns=1,r_post_ns=1)
    h=DeadlineCommHarness(model,'full',hc)
    _,row,_,_,_=h.step(np.zeros((5,2),np.float32),np.ones((5,3),bool),0)
    assert row['action_miss'] and row['K_delivered']==0
    assert row['mean_age'] is None and row['invalid_cache_fraction']==1
    assert row['cpu_total_ns']==sum(row[k] for k in ('encode_ns','gate_ns','cache_ns','act_ns','other_ns'))

def test_hallway_deterministic_without_global_rng_mutation():
    np.random.seed(41); random.seed(23)
    state=np.random.get_state(); py=random.getstate()
    a=Hallway(seed=7); b=Hallway(seed=99)
    oa,_=a.reset(seed=9); ob,_=b.reset(seed=9)
    assert np.array_equal(oa,ob)
    for _ in range(5):
        aa=a.step([0]*5); bb=b.step([0]*5)
        assert np.array_equal(aa[0],bb[0]) and aa[1:]==bb[1:]
    after=np.random.get_state()
    assert np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    assert random.getstate()==py

def test_gates_boundary_and_no_ack():
    m=torch.zeros(3); q=torch.tensor([1.,.95,0]); mask=torch.ones(3,dtype=torch.bool)
    assert Gate('vbc',1,q_threshold=.1).decide(0,0,m,q,mask)
    assert not Gate('vbc',1).decide(0,0,m,q,torch.tensor([True,False,False]))
    g=Gate('tmc',1,message_threshold=.1)
    assert g.decide(0,0,m,q,mask)
    g.record_attempt(0,m)
    assert not g.decide(0,1,m,q,mask)
    assert Gate('periodic',5,period=2).decide(1,1,m,q,mask)
    assert not Gate('random',1,p=0).decide(0,0,m,q,mask)
    assert Gate('random',1,p=1).decide(0,0,m,q,mask)


def test_horizon_is_truncation_not_natural_termination():
    env=Hallway(seed=0); env.reset(seed=42)
    for i in range(env.episode_length):
        _,_,term,trunc,_=env.step([0]*env.n_agents)
        assert not term
        assert trunc==(i==env.episode_length-1)
