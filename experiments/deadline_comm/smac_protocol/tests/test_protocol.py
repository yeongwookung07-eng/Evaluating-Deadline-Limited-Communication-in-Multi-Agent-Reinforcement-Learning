from copy import deepcopy
import numpy as np
import torch
from experiments.deadline_comm.common import load_config
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.smac_protocol.evaluate import make_harness,compare_states,seed_summary,bootstrap_pairs
from experiments.deadline_comm.smac_protocol.campaign import gate_pass
from experiments.deadline_comm.smac_protocol.train import epsilon


def fixture():
    torch.manual_seed(17)
    spec=load_config('experiments/deadline_comm/configs/smacv2_protocol.yaml')
    m=DecayedPolicy(4,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix');m.eval()
    return spec,m


def test_slow_full_rr_exact_history_with_fewer_attempts():
    spec,m=fixture();a=make_harness(m,'full',spec,'slow_5ms','fixed',0);b=make_harness(m,'rr_cap',spec,'slow_5ms','fixed',0)
    rng=np.random.default_rng(2)
    for t in range(20):
        obs=rng.normal(size=(5,4));mask=np.ones((5,3),bool);mask[0,1:]=False
        aa,ra,ma=a.step(obs,mask,t);ab,rb,mb=b.step(obs,mask,t)
        assert all(compare_states(a,b,aa,ab,ma,mb).values())
        assert ra['wire_bytes']==240 and rb['wire_bytes']==48
        assert ra['K_delivered']==rb['K_delivered']==1


def test_no_comm_and_ideal_endpoints():
    spec,m=fixture()
    for arm,k in [('no_comm',0),('ideal_full',5)]:
        h=make_harness(m,arm,spec,'slow_5ms','fixed',1)
        _,r,_=h.step(np.ones((5,4)),np.ones((5,3),bool),0)
        assert r['K_delivered']==k
        assert int((h.source>=0).sum())==k


def test_medium_identity_mismatch_retained():
    spec,m=fixture();a=make_harness(m,'full',spec,'medium_2ms','fixed',0);b=make_harness(m,'rr_cap',spec,'medium_2ms','fixed',0)
    checks=[]
    for t in range(3):
        aa,_,ma=a.step(np.ones((5,4)),np.ones((5,3),bool),t);ab,_,mb=b.step(np.ones((5,4)),np.ones((5,3),bool),t)
        checks.append(compare_states(a,b,aa,ab,ma,mb))
    assert checks[0]['delivery'] and not checks[1]['delivery']


def test_learning_gate_requires_all_seeds():
    spec,_=fixture()
    assert gate_pass([dict(win_rate=.25)]*3,spec)
    assert not gate_pass([dict(win_rate=.5),dict(win_rate=.5),dict(win_rate=.19)],spec)
    assert not gate_pass([dict(win_rate=.5)]*2,spec)


def test_paired_ci_uses_training_seed_units():
    r=bootstrap_pairs([1,1,1]);assert r['ci95']==[1,1] and r['n_training_seeds']==3


def test_continuation_epsilon_not_reset_each_phase():
    spec,_=fixture()
    assert epsilon(spec,100000,100000)==.30
    assert abs(epsilon(spec,300000,100000)-.05)<1e-9
    assert abs(epsilon(spec,600000,100000)-.05)<1e-9
