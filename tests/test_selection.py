import torch
from experiments.deadline_comm.selection_validation.harness import SelectionHarness
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.common import load_config
from experiments.deadline_comm.adapters.hallway import Hallway

def test_common_slots_and_round_robin():
 cfg=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness'];cfg['fixed_ready']=True
 cfg['link']['bandwidth_bytes_s']=20000
 env=Hallway();obs,_=env.reset(seed=0)
 model=DecayedPolicy(2,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix')
 histories={}
 for method in ['dac','rr_cap','random_cap','age_cap']:
  h=SelectionHarness(model,method,cfg,0,False);history=[]
  for t in range(5):
   _,row,msg=h.step(obs,env.available_actions(),t)
   assert row['K_attempt']==row['K_delivered']==1
   assert h.dac.spent<=h.dac.steps*120
   history.append([m['sender'] for m in msg if m['accepted']])
  histories[method]=history
 assert histories['rr_cap']==[[0],[1],[2],[3],[4]]

def test_rr_skips_q_head():
 cfg=load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
 model=DecayedPolicy(2,10,5,3,hidden_dim=64,message_dim=8,mixer='qmix');env=Hallway();obs,_=env.reset(seed=0)
 def fail(*a):raise AssertionError('unnecessary Q-head computation')
 model.local_q=fail
 h=SelectionHarness(model,'rr_cap',cfg,0,False);h.step(obs,env.available_actions(),0)
