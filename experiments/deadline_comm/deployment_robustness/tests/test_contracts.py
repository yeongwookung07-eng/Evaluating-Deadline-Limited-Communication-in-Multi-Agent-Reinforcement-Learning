import numpy as np
from experiments.deadline_comm.deployment_robustness.mnist_uncertain_readiness import admission_probability, evaluate_cell
from experiments.deadline_comm.deployment_robustness.broadcast_hard_deadline import executed_action


def test_known_readiness_preserves_full_with_same_cost():
    ready=np.array([100,300,600,900]);full=np.array([.8,.7,.6,.9]);zero=np.full(4,.1)
    for d in [250,500,750,1000,1250]:
        for cost in [0,50,100,200,400]:
            r=evaluate_cell(ready,full,zero,d,0,0,cost)
            np.testing.assert_array_equal(r['informed'],r['full_same_cost'])


def test_estimate_noise_boundary_and_cost():
    assert admission_probability(np.array([530]),750,100,0,0)[0]==.5
    assert admission_probability(np.array([530]),750,0,0,0)[0]==1
    assert admission_probability(np.array([530]),750,0,0,1)[0]==0
    p=admission_probability(np.array([100,300,600,900]),750,100,0,100)
    assert np.all(np.diff(p)<0)


def test_overhead_can_defeat_cheap_random():
    r=evaluate_cell(np.array([300,600]),np.ones(2),np.zeros(2),750,100,-1000,400)
    assert r['informed'].mean()<r['random_no_extra_cost'].mean()


def test_no_information_equals_same_cost_random():
    r=evaluate_cell(np.full(4,300),np.array([.1,.3,.6,.8]),np.zeros(4),750,100,0,100)
    np.testing.assert_array_equal(r['informed'],r['random_same_cost'])


def test_actuator_rejects_late_candidate_and_holds_actual_history():
    candidate=np.array([1,0,1]);previous=np.array([0,1,0])
    np.testing.assert_array_equal(executed_action(candidate,previous,True,'hold'),previous)
    np.testing.assert_array_equal(executed_action(candidate,previous,True,'zero'),[0,0,0])
    for fb in ['ordinary','hold','zero']:
        np.testing.assert_array_equal(executed_action(candidate,previous,False,fb),candidate)
    np.testing.assert_array_equal(executed_action(candidate,previous,True,'ordinary'),candidate)


def test_late_executed_actions_reach_next_recurrent_encode():
    import torch
    from unittest.mock import patch
    from experiments.deadline_comm.day2.policy import DecayedPolicy
    from experiments.deadline_comm.deployment_robustness.broadcast_hard_deadline import rollout
    torch.manual_seed(17)
    model=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix').eval()
    history=[];original=model.encode
    def capture(obs,previous,hidden):
        history.append(previous.detach().clone())
        return original(obs,previous,hidden)
    with patch.object(model,'encode',side_effect=capture):
        tr=rollout(model,901,'normal','hold',0,fixed=True,forced_miss=True)
    assert len(history)==len(tr)==20
    for t in range(1,20):
        np.testing.assert_array_equal(history[t].argmax(-1)[0].numpy(),tr[t-1]['actions'])
        assert history[t].sum().item()==5
    assert all(r['actions']==[0]*5 for r in tr)


def test_no_misses_preserve_behavior_through_full_rollout():
    import torch
    from experiments.deadline_comm.day2.policy import DecayedPolicy
    from experiments.deadline_comm.deployment_robustness.broadcast_hard_deadline import rollout, behavioral
    torch.manual_seed(19)
    model=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix').eval()
    original=rollout(model,902,'normal','ordinary',0,fixed=True,forced_miss=False)
    for fallback in ('hold','zero'):
        other=rollout(model,902,'normal',fallback,0,fixed=True,forced_miss=False)
        assert behavioral(original)==behavioral(other)
