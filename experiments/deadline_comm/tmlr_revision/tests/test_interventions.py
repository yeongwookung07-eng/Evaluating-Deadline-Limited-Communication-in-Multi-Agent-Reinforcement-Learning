from copy import deepcopy
import torch
from experiments.deadline_comm.tmlr_revision.experiments import setup,clone,step,suffix
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.selection_validation.harness import SelectionHarness
from experiments.deadline_comm.reviewer_validation.endpoints import EndpointHarness
from experiments.deadline_comm.reviewer_validation.train_ten_seeds import configure_arm
from experiments.deadline_comm.reviewer_validation.hidden_source import HiddenSource
from experiments.deadline_comm.adapters.hallway import Hallway

def test_forced_fresh_subset_and_hold():
 torch.set_num_threads(1);torch.manual_seed(12)
 m=DecayedPolicy(2,10,5,3,hidden_dim=16,message_dim=8);env=Hallway();obs,_=env.reset(seed=1);h=SelectionHarness(m,'dac',setup(['slow',5.]),0,False)
 with torch.inference_mode():
  for sender in range(5):
   hh=clone(h);_,r=step(deepcopy(env),hh,obs.copy(),0,[sender]);assert r['senders']==[sender]
   assert hh.source[0,sender]==0 and int((hh.source>=0).sum())==1
   expected=m.message(hh.hidden)[0,sender];assert torch.equal(expected,hh.cache[0,sender])
  hh=clone(h);_,r=step(deepcopy(env),hh,obs.copy(),0,[2],hold=2);assert r['senders']==[] and bool((hh.source<0).all())
 assert bool((h.source<0).all()) and h.dac.credit==0

def test_original_subset_reproduces_full_suffix():
 torch.manual_seed(10);m=DecayedPolicy(2,10,5,3,hidden_dim=16,message_dim=8);env=Hallway();obs,_=env.reset(seed=3);h=SelectionHarness(m,'rr_cap',setup(['medium',2.]),0,False)
 with torch.inference_mode():
  a,t=suffix(deepcopy(env),clone(h),obs.copy(),0);b,u=suffix(deepcopy(env),clone(h),obs.copy(),0,t[0]['senders']);assert a==b and t==u

def test_endpoint_clone_preserves_ideal_reference():
 torch.manual_seed(5);m=DecayedPolicy(4,10,5,2,hidden_dim=16,message_dim=8);env=HiddenSource();obs,_=env.reset(seed=4);method,cfg=configure_arm(setup(['slow',5.]),'ideal_full',20000,5.)
 h=EndpointHarness(m,method,cfg,0,False)
 with torch.inference_mode():
  c=clone(h);_,r=step(deepcopy(env),c,obs.copy(),0);assert r['senders']==list(range(5));assert (h.source<0).all()

def test_environment_snapshots_replay_and_branch_are_independent():
 import numpy as np
 from experiments.deadline_comm.reviewer_validation.train_ten_seeds import trajectory
 from experiments.deadline_comm.replay import CommReplay
 env=Hallway(group_ids=[0,0,1,1,1]);obs,_=env.reset(seed=8);state=env.get_state();saved_obs=obs.copy();saved_state=state.copy()
 branch=deepcopy(env);branch.step(np.ones(5,int));assert np.array_equal(env.get_state(),state)
 env.step(np.ones(5,int));assert np.array_equal(obs,saved_obs) and np.array_equal(state,saved_state)
 model=DecayedPolicy(2,10,5,3,hidden_dim=16,message_dim=8)
 with torch.inference_mode():record=trajectory(env,model,'dac',setup(['slow',5.]),8)
 original=record.episode.observations.copy();replay=CommReplay();replay.insert(record)
 env.reset(seed=9);env.step(np.ones(5,int));assert np.array_equal(record.episode.observations,original)
 record.episode.observations.fill(999);assert np.array_equal(replay.items[0].episode.observations,original)
 sampled=replay.sample(1)[0];sampled.episode.observations.fill(777);assert np.array_equal(replay.items[0].episode.observations,original)

def test_forced_selection_updates_scheduler_state():
 torch.manual_seed(12);m=DecayedPolicy(2,10,5,3,hidden_dim=16,message_dim=8);env=Hallway();obs,_=env.reset(seed=1)
 h=SelectionHarness(m,'rr_cap',setup(['slow',5.]),0,False)
 with torch.inference_mode():
  c=clone(h);step(deepcopy(env),c,obs.copy(),0,[3])
 assert c.dac.age.tolist()==[1,1,1,0,1] and c.dac.credit==72 and c.dac.cursor==1
 assert h.dac.credit==0 and h.dac.age.tolist()==[0]*5 and h.dac.cursor==0
 with torch.inference_mode():c.cache.fill_(10)
 assert (h.cache==0).all()
